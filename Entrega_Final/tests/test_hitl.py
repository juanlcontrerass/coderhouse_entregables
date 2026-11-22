"""Human-in-the-loop: pausa en tareas críticas, reanudación y efecto lateral controlado."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app.graph.hitl as hitl
from app.core.config import get_settings
from app.graph.turn import graph_config, turn_input
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _graph, _models, vacation_store  # noqa: F401

USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


def _hitl_graph(answer: AIMessage = GOOD_ANSWER) -> Any:
    return _graph(_models(HAPPY_SUPERVISOR, [answer]), InMemorySaver())


def _files(directory: Path) -> list[Path]:
    return list(directory.iterdir()) if directory.exists() else []


async def test_regular_query_finishes_without_pause() -> None:
    graph = _hitl_graph()
    config = graph_config("conv", "job-1")
    await graph.ainvoke(turn_input(QUERY), config)
    snapshot = await graph.aget_state(config)
    assert not snapshot.interrupts and not snapshot.next
    assert snapshot.values["task_completed"] and snapshot.values["published_path"] == ""


async def test_publish_request_pauses_before_the_side_effect(reports_dir: Path) -> None:
    graph = _hitl_graph()
    config = graph_config("conv", "job-1")
    await graph.ainvoke(turn_input(QUERY, publish=True), config)
    snapshot = await graph.aget_state(config)

    assert snapshot.next == ("human_approval",)
    request = hitl.ApprovalRequest.model_validate(snapshot.interrupts[0].value)
    assert request.accion_pendiente == "publicar_informe"
    assert any("Efecto lateral" in r for r in request.motivos)
    assert GOOD_ANSWER.content in request.vista_previa
    assert not _files(reports_dir)  # nada se publicó todavía


async def test_approval_resumes_and_publishes_with_the_job_id(reports_dir: Path) -> None:
    graph = _hitl_graph()
    config = graph_config("conv-ok", "job-ok")
    await graph.ainvoke(turn_input(QUERY, publish=True), config)
    final = await graph.ainvoke(Command(resume={"approved": True, "comment": "ok de RR. HH."}), config)

    assert final["approval"] == {"approved": True, "comment": "ok de RR. HH."}
    report = reports_dir / "informe_job-ok.md"
    assert final["published_path"] == str(report)
    text = report.read_text(encoding="utf-8")
    assert GOOD_ANSWER.content in text and "ok de RR. HH." in text


async def test_rejection_ends_without_publishing(reports_dir: Path) -> None:
    graph = _hitl_graph()
    config = graph_config("conv", "job-1")
    await graph.ainvoke(turn_input(QUERY, publish=True), config)
    final = await graph.ainvoke(Command(resume={"approved": False, "comment": "no"}), config)

    assert final["approval"]["approved"] is False
    assert final["published_path"] == "" and not _files(reports_dir)
    assert not (await graph.aget_state(config)).next


async def test_invalid_resume_payload_is_rejected() -> None:
    graph = _hitl_graph()
    config = graph_config("conv", "job-1")
    await graph.ainvoke(turn_input(QUERY, publish=True), config)
    with pytest.raises(Exception, match="approved"):
        await graph.ainvoke(Command(resume={"comment": "falta approved"}), config)


async def test_publish_verb_in_the_query_is_detected() -> None:
    state: Any = turn_input("Calcula el promedio y envía el informe a RR. HH.")
    assert hitl.wants_publication(state)
    assert not hitl.wants_publication(turn_input("¿Cuántos días de vacaciones hay?"))  # type: ignore[arg-type]


async def test_tokens_are_counted_and_high_cost_requires_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "cost_approval_token_threshold", 100)
    graph = _hitl_graph(AIMessage(GOOD_ANSWER.content, usage_metadata=USAGE))
    config = graph_config("conv", "job-1")
    await graph.ainvoke(turn_input(QUERY), config)
    snapshot = await graph.aget_state(config)

    assert snapshot.values["tokens_used"] == 150
    request = snapshot.interrupts[0].value
    assert request["accion_pendiente"] == "entregar_respuesta"
    assert any("Costo alto" in r for r in request["motivos"])

    final = await graph.ainvoke(Command(resume={"approved": True}), config)
    assert final["published_path"] == ""  # aprobar un costo alto no publica nada
