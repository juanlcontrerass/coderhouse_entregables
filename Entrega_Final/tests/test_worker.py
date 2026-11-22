"""Worker: transiciones de estado en Redis, FAILED ante excepciones y turno de la conversación."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from langgraph.checkpoint.memory import InMemorySaver

import app.worker.worker as worker
from app.core.config import get_settings
from app.storage.conversations import ConversationStore
from app.storage.jobs import JobStore
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _graph, _models, vacation_store  # noqa: F401


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def store(redis: FakeAsyncRedis) -> JobStore:
    return JobStore(redis)


@pytest.fixture
def conversations(redis: FakeAsyncRedis, store: JobStore) -> ConversationStore:
    return ConversationStore(redis, store)


@pytest.fixture(autouse=True)
def reset_runtime() -> Iterator[None]:
    yield
    worker.set_runtime(None)


def _install(graph: Any, store: JobStore, conversations: ConversationStore) -> None:
    worker.set_runtime(worker.Runtime(graph=graph, store=store, conversations=conversations))


def _real_graph() -> Any:
    return _graph(_models(HAPPY_SUPERVISOR, [GOOD_ANSWER]), InMemorySaver())


async def _new_job(store: JobStore, conversations: ConversationStore, job_id: str, publish: bool = False) -> None:
    await conversations.create("conv")
    await conversations.acquire("conv", job_id)
    await store.create(job_id, "conv", QUERY, publish=publish)


class ExplodingGraph:
    async def ainvoke(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("el LLM no responde")


async def test_successful_run_ends_in_done_and_releases_the_turn(store: JobStore, conversations: ConversationStore) -> None:
    _install(_real_graph(), store, conversations)
    await _new_job(store, conversations, "j1")
    outcome = await worker.run_task({}, "j1")

    job = await store.get("j1")
    assert job is not None and outcome["status"] == job["status"] == "DONE"
    assert job["result"]["final_answer"] == GOOD_ANSWER.content
    assert job["result"]["route"] == ["researcher", "analyst", "synthesizer"]
    assert job["result"]["sources"] == ["politica_vacaciones.md"]
    assert job["started_at"] and job["finished_at"]
    assert (await conversations.get("conv"))["active_job_id"] == ""  # type: ignore[index]


async def test_exception_marks_job_failed(store: JobStore, conversations: ConversationStore) -> None:
    _install(ExplodingGraph(), store, conversations)
    await _new_job(store, conversations, "j2")
    outcome = await worker.run_task({}, "j2")

    job = await store.get("j2")
    assert job is not None and outcome["status"] == job["status"] == "FAILED"
    assert job["error"] == "RuntimeError: el LLM no responde"
    assert (await conversations.get("conv"))["active_job_id"] == ""  # type: ignore[index]


async def test_timeout_marks_job_failed(store: JobStore, conversations: ConversationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowGraph:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(5)

    monkeypatch.setattr(get_settings(), "job_timeout_seconds", 0.05)
    _install(SlowGraph(), store, conversations)
    await _new_job(store, conversations, "j3")
    await worker.run_task({}, "j3")
    job = await store.get("j3")
    assert job is not None and job["status"] == "FAILED" and job["error"].startswith("TimeoutError")


async def test_critical_task_waits_keeping_the_turn_then_resumes(
    store: JobStore, conversations: ConversationStore, reports_dir: Path
) -> None:
    _install(_real_graph(), store, conversations)
    await _new_job(store, conversations, "j4", publish=True)
    await worker.run_task({}, "j4")

    job = await store.get("j4")
    assert job is not None and job["status"] == "WAITING_APPROVAL"
    assert job["approval_request"]["accion_pendiente"] == "publicar_informe"
    assert (await conversations.get("conv"))["active_job_id"] == "j4"  # type: ignore[index]

    await worker.resume_task({}, "j4", {"approved": True, "comment": "ok"})
    job = await store.get("j4")
    assert job is not None and job["status"] == "DONE" and job["approval_request"] is None
    assert job["result"]["published_path"] == str(reports_dir / "informe_j4.md")
    assert (await conversations.get("conv"))["active_job_id"] == ""  # type: ignore[index]


async def test_rejected_task_ends_in_rejected(store: JobStore, conversations: ConversationStore, reports_dir: Path) -> None:
    _install(_real_graph(), store, conversations)
    await _new_job(store, conversations, "j5", publish=True)
    await worker.run_task({}, "j5")
    await worker.resume_task({}, "j5", {"approved": False, "comment": "no publicar"})

    job = await store.get("j5")
    assert job is not None and job["status"] == "REJECTED"
    assert not reports_dir.exists() or not list(reports_dir.iterdir())


async def test_missing_job_is_discarded(store: JobStore, conversations: ConversationStore) -> None:
    _install(ExplodingGraph(), store, conversations)
    assert (await worker.run_task({}, "no-existe"))["status"] == "DESCARTADO"
