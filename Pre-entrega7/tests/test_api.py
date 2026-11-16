"""API: encolado sin bloqueo, consulta de estado y endpoint de aprobación (Redis falso)."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fakeredis import FakeAsyncRedis

from app.jobs import JobStatus, JobStore
from app.main import create_app


class FakeQueue:
    """Registra lo que la API encola en lugar de enviarlo a Arq."""

    def __init__(self, fail: bool = False) -> None:
        self.jobs: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail = fail

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> None:
        if self.fail:
            raise ConnectionError("cola caída")
        self.jobs.append((function, args, kwargs))


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
async def client(redis: FakeAsyncRedis, queue: FakeQueue) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(redis=redis, queue=queue)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_create_task_enqueues_and_returns_job_id(client: httpx.AsyncClient, queue: FakeQueue) -> None:
    response = await client.post("/tasks", json={"query": "¿Cuántos días de vacaciones hay?"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "PENDING" and body["status_url"] == f"/tasks/{body['job_id']}"
    assert queue.jobs == [("run_task", (body["job_id"],), {"_job_id": body["job_id"]})]

    status = (await client.get(body["status_url"])).json()
    assert status["status"] == "PENDING" and status["query"] == "¿Cuántos días de vacaciones hay?"
    assert status["publish"] is False


async def test_unknown_task_returns_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/tasks/no-existe")).status_code == 404
    assert (await client.post("/tasks/no-existe/approve", json={"approved": True})).status_code == 404


async def test_invalid_payload_returns_422(client: httpx.AsyncClient) -> None:
    assert (await client.post("/tasks", json={"query": ""})).status_code == 422


async def test_approve_requires_waiting_approval(client: httpx.AsyncClient) -> None:
    job_id = (await client.post("/tasks", json={"query": "consulta"})).json()["job_id"]
    response = await client.post(f"/tasks/{job_id}/approve", json={"approved": True})
    assert response.status_code == 409 and "PENDING" in response.json()["detail"]


async def test_approve_enqueues_resume_once(client: httpx.AsyncClient, redis: FakeAsyncRedis, queue: FakeQueue) -> None:
    job_id = (await client.post("/tasks", json={"query": "Publica el informe", "publish": True})).json()["job_id"]
    await JobStore(redis).update(job_id, status=JobStatus.WAITING_APPROVAL, approval_request={"motivos": ["x"]})

    response = await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "adelante"})
    assert response.status_code == 202
    assert response.json()["status"] == "PENDING"
    assert response.json()["approval"] == {"approved": True, "comment": "adelante"}
    function, args, kwargs = queue.jobs[-1]
    assert function == "resume_task" and args == (job_id, {"approved": True, "comment": "adelante"})
    assert kwargs["_job_id"].startswith(f"{job_id}:resume:")

    # Una segunda aprobación ya no encuentra el job en pausa
    assert (await client.post(f"/tasks/{job_id}/approve", json={"approved": True})).status_code == 409
    assert len(queue.jobs) == 2


async def test_queue_failure_marks_job_failed(redis: FakeAsyncRedis) -> None:
    app = create_app(redis=redis, queue=FakeQueue(fail=True))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.post("/tasks", json={"query": "consulta"})).status_code == 503
    keys = [k async for k in redis.scan_iter("job:*")]
    assert len(keys) == 1
    job = await JobStore(redis).get(keys[0].removeprefix("job:"))
    assert job is not None and job["status"] == "FAILED" and "cola caída" in job["error"]


async def test_health(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok", "redis": "ok"}
