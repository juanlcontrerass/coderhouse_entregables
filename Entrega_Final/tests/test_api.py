"""API: encolado sin bloqueo, conversaciones, estado y aprobación (Redis falso, sin worker)."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver

from app.api.main import create_app
from app.graph.turn import graph_config, turn_input
from app.storage.jobs import JobStatus, JobStore
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _graph, _models, vacation_store  # noqa: F401


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
def checkpointer() -> InMemorySaver:
    return InMemorySaver()


@pytest.fixture
async def client(redis: FakeAsyncRedis, queue: FakeQueue, checkpointer: InMemorySaver) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(redis=redis, queue=queue, checkpointer=checkpointer)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def test_create_task_enqueues_and_returns_ids(client: httpx.AsyncClient, queue: FakeQueue) -> None:
    response = await client.post("/tasks", json={"query": "¿Cuántos días de vacaciones hay?"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "PENDING" and body["status_url"] == f"/tasks/{body['job_id']}"
    assert body["conversation_url"] == f"/conversations/{body['conversation_id']}"
    assert queue.jobs == [("run_task", (body["job_id"],), {"_job_id": body["job_id"]})]

    status = (await client.get(body["status_url"])).json()
    assert status["status"] == "PENDING" and status["query"] == "¿Cuántos días de vacaciones hay?"
    assert status["publish"] is False and status["conversation_id"] == body["conversation_id"]


async def test_conversation_accepts_one_turn_at_a_time(client: httpx.AsyncClient, redis: FakeAsyncRedis) -> None:
    created = await client.post("/conversations")
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]
    url = created.json()["messages_url"]

    first = await client.post(url, json={"query": "Primera pregunta"})
    assert first.status_code == 202
    busy = await client.post(url, json={"query": "Segunda pregunta"})
    assert busy.status_code == 409 and first.json()["job_id"] in busy.json()["detail"]

    # El worker termina el primer turno: la conversación vuelve a aceptar mensajes
    await JobStore(redis).update(first.json()["job_id"], status=JobStatus.DONE)
    second = await client.post(url, json={"query": "Segunda pregunta"})
    assert second.status_code == 202 and second.json()["conversation_id"] == conversation_id

    record = (await client.get(f"/conversations/{conversation_id}")).json()
    assert record["job_ids"] == [first.json()["job_id"], second.json()["job_id"]]
    assert record["active_job_id"] == second.json()["job_id"]


async def test_get_conversation_reads_history_from_the_checkpointer(
    client: httpx.AsyncClient, checkpointer: InMemorySaver
) -> None:
    conversation_id = (await client.post("/conversations")).json()["conversation_id"]
    graph = _graph(_models(HAPPY_SUPERVISOR, [GOOD_ANSWER]), checkpointer)
    await graph.ainvoke(turn_input(QUERY), graph_config(conversation_id))

    body = (await client.get(f"/conversations/{conversation_id}")).json()
    assert body["turns"] == [{"query": QUERY, "answer": GOOD_ANSWER.content}]


async def test_unknown_resources_return_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/tasks/no-existe")).status_code == 404
    assert (await client.post("/tasks/no-existe/approve", json={"approved": True})).status_code == 404
    assert (await client.get("/conversations/no-existe")).status_code == 404
    assert (await client.post("/conversations/no-existe/messages", json={"query": "hola"})).status_code == 404


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "x" * 2001}, {"publish": True}])
async def test_invalid_payload_returns_422(client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
    assert (await client.post("/tasks", json=payload)).status_code == 422


async def test_approve_requires_waiting_approval(client: httpx.AsyncClient) -> None:
    job_id = (await client.post("/tasks", json={"query": "consulta"})).json()["job_id"]
    response = await client.post(f"/tasks/{job_id}/approve", json={"approved": True})
    assert response.status_code == 409 and "PENDING" in response.json()["detail"]
    assert (await client.post(f"/tasks/{job_id}/approve", json={"comment": "sin approved"})).status_code == 422


async def test_approve_enqueues_resume_once(client: httpx.AsyncClient, redis: FakeAsyncRedis, queue: FakeQueue) -> None:
    job_id = (await client.post("/tasks", json={"query": "Publica el informe", "publish": True})).json()["job_id"]
    request = {"motivos": ["x"], "accion_pendiente": "publicar_informe", "tokens_usados": 10, "vista_previa": "..."}
    await JobStore(redis).update(job_id, status=JobStatus.WAITING_APPROVAL, approval_request=request)

    waiting = (await client.get(f"/tasks/{job_id}")).json()
    assert waiting["approval_request"]["tipo"] == "aprobacion_requerida"

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


async def test_queue_failure_marks_job_failed_and_frees_the_conversation(redis: FakeAsyncRedis) -> None:
    app = create_app(redis=redis, queue=FakeQueue(fail=True), checkpointer=InMemorySaver())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.post("/tasks", json={"query": "consulta"})).status_code == 503
    keys = [k async for k in redis.scan_iter("job:*")]
    assert len(keys) == 1
    job = await JobStore(redis).get(keys[0].removeprefix("job:"))
    assert job is not None and job["status"] == "FAILED" and "cola caída" in job["error"]
    assert await redis.hget(f"conversation:{job['conversation_id']}", "active_job_id") == ""


async def test_health_and_docs(client: httpx.AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert body["status"] == "ok" and body["redis"] == "ok"
    schema = (await client.get("/openapi.json")).json()
    assert {"/conversations", "/conversations/{conversation_id}/messages", "/tasks", "/tasks/{job_id}/approve"} <= set(schema["paths"])
