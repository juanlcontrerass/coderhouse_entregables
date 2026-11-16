"""Human-in-the-loop: pausa en tareas críticas, reanudación y efecto lateral controlado."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app.hitl as hitl
import app.tools.publisher as publisher_module
from app.graph import build_graph
from app.runner import initial_state
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _models, vacation_store  # noqa: F401

USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


@pytest.fixture(autouse=True)
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(publisher_module, "REPORTS_DIR", tmp_path)
    return tmp_path


def _graph(answer: AIMessage = GOOD_ANSWER) -> Any:
    models = _models(HAPPY_SUPERVISOR, [answer])
    return build_graph(
        models["supervisor"],
        checkpointer=InMemorySaver(),
        researcher_llm=models["researcher"],
        analyst_llm=models["analyst"],
        synthesizer_llm=models["synthesizer"],
    )


def _config(job_id: str = "job-1") -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}}


async def test_regular_query_finishes_without_pause() -> None:
    graph = _graph()
    await graph.ainvoke(initial_state(QUERY), _config())
    snapshot = await graph.aget_state(_config())
    assert not snapshot.interrupts and not snapshot.next
    assert snapshot.values["task_completed"] and snapshot.values["published_path"] == ""


async def test_publish_request_pauses_before_the_side_effect(reports_dir: Path) -> None:
    graph = _graph()
    await graph.ainvoke(initial_state(QUERY, publish=True), _config())
    snapshot = await graph.aget_state(_config())

    assert snapshot.next == ("human_approval",)
    request = snapshot.interrupts[0].value
    assert request["accion_pendiente"] == "publicar_informe"
    assert any("Efecto lateral" in r for r in request["motivos"])
    assert GOOD_ANSWER.content in request["vista_previa"]
    assert not list(reports_dir.iterdir())  # nada se publicó todavía


async def test_approval_resumes_and_publishes(reports_dir: Path) -> None:
    graph = _graph()
    await graph.ainvoke(initial_state(QUERY, publish=True), _config("job-ok"))
    final = await graph.ainvoke(Command(resume={"approved": True, "comment": "ok de RR. HH."}), _config("job-ok"))

    assert final["approval"] == {"approved": True, "comment": "ok de RR. HH."}
    report = reports_dir / "informe_job-ok.md"
    assert final["published_path"] == str(report)
    text = report.read_text(encoding="utf-8")
    assert GOOD_ANSWER.content in text and "ok de RR. HH." in text


async def test_rejection_ends_without_publishing(reports_dir: Path) -> None:
    graph = _graph()
    await graph.ainvoke(initial_state(QUERY, publish=True), _config())
    final = await graph.ainvoke(Command(resume={"approved": False, "comment": "no"}), _config())

    assert final["approval"]["approved"] is False
    assert final["published_path"] == "" and not list(reports_dir.iterdir())
    assert not (await graph.aget_state(_config())).next


async def test_publish_verb_in_the_query_is_detected() -> None:
    state: Any = initial_state("Calcula el promedio y envía el informe a RR. HH.")
    assert hitl.wants_publication(state)
    assert not hitl.wants_publication(initial_state("¿Cuántos días de vacaciones hay?"))  # type: ignore[arg-type]


async def test_tokens_are_counted_and_high_cost_requires_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hitl, "COST_APPROVAL_TOKEN_THRESHOLD", 100)
    answer = AIMessage(GOOD_ANSWER.content, usage_metadata=USAGE)
    graph = _graph(answer)
    await graph.ainvoke(initial_state(QUERY), _config())
    snapshot = await graph.aget_state(_config())

    assert snapshot.values["tokens_used"] == 150
    request = snapshot.interrupts[0].value
    assert request["accion_pendiente"] == "entregar_respuesta"
    assert any("Costo alto" in r for r in request["motivos"])

    final = await graph.ainvoke(Command(resume={"approved": True}), _config())
    assert final["published_path"] == ""  # aprobar un costo alto no publica nada
