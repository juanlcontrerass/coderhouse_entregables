"""Worker: transiciones de estado en Redis, incluido FAILED ante excepciones."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver

import app.tools.publisher as publisher_module
import app.worker as worker
from app.graph import build_graph
from app.jobs import JobStore
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _models, vacation_store  # noqa: F401


@pytest.fixture
def store() -> JobStore:
    return JobStore(FakeAsyncRedis(decode_responses=True))


@pytest.fixture(autouse=True)
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(publisher_module, "REPORTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_runtime() -> Iterator[None]:
    yield
    worker.set_runtime(None)


def _graph() -> Any:
    models = _models(HAPPY_SUPERVISOR, [GOOD_ANSWER])
    return build_graph(
        models["supervisor"],
        checkpointer=InMemorySaver(),
        researcher_llm=models["researcher"],
        analyst_llm=models["analyst"],
        synthesizer_llm=models["synthesizer"],
    )


class ExplodingGraph:
    async def ainvoke(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("el LLM no responde")


async def test_successful_run_ends_in_done(store: JobStore) -> None:
    worker.set_runtime(worker.Runtime(graph=_graph(), store=store))
    await store.create("j1", QUERY)
    outcome = await worker.run_task({}, "j1")

    job = await store.get("j1")
    assert job is not None and outcome["status"] == job["status"] == "DONE"
    assert job["result"]["final_answer"] == GOOD_ANSWER.content
    assert job["result"]["route"] == ["researcher", "analyst", "synthesizer"]
    assert job["started_at"] and job["finished_at"]


async def test_exception_marks_job_failed(store: JobStore) -> None:
    worker.set_runtime(worker.Runtime(graph=ExplodingGraph(), store=store))
    await store.create("j2", QUERY)
    outcome = await worker.run_task({}, "j2")

    job = await store.get("j2")
    assert job is not None and outcome["status"] == job["status"] == "FAILED"
    assert job["error"] == "RuntimeError: el LLM no responde"


async def test_timeout_marks_job_failed(store: JobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class SlowGraph:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(5)

    monkeypatch.setattr(worker, "JOB_TIMEOUT_SECONDS", 0.05)
    worker.set_runtime(worker.Runtime(graph=SlowGraph(), store=store))
    await store.create("j3", QUERY)
    await worker.run_task({}, "j3")
    job = await store.get("j3")
    assert job is not None and job["status"] == "FAILED" and job["error"].startswith("TimeoutError")


async def test_critical_task_waits_then_resumes(store: JobStore, reports_dir: Path) -> None:
    worker.set_runtime(worker.Runtime(graph=_graph(), store=store))
    await store.create("j4", QUERY, publish=True)
    await worker.run_task({}, "j4")

    job = await store.get("j4")
    assert job is not None and job["status"] == "WAITING_APPROVAL"
    assert job["approval_request"]["accion_pendiente"] == "publicar_informe"

    await worker.resume_task({}, "j4", {"approved": True, "comment": "ok"})
    job = await store.get("j4")
    assert job is not None and job["status"] == "DONE" and job["approval_request"] is None
    assert job["result"]["published_path"] == str(reports_dir / "informe_j4.md")


async def test_rejected_task_ends_in_rejected(store: JobStore, reports_dir: Path) -> None:
    worker.set_runtime(worker.Runtime(graph=_graph(), store=store))
    await store.create("j5", QUERY, publish=True)
    await worker.run_task({}, "j5")
    await worker.resume_task({}, "j5", {"approved": False, "comment": "no publicar"})

    job = await store.get("j5")
    assert job is not None and job["status"] == "REJECTED"
    assert not list(reports_dir.iterdir())


async def test_missing_job_is_discarded(store: JobStore) -> None:
    worker.set_runtime(worker.Runtime(graph=ExplodingGraph(), store=store))
    assert (await worker.run_task({}, "no-existe"))["status"] == "DESCARTADO"
