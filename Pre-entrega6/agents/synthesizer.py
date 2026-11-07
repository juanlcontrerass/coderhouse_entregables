"""Síntesis final y nodo de Validación previo al END."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from agents.common import contributions_by, message_text, user_query
from config import MAX_STEPS, MAX_VALIDATION_RETRIES
from prompts import SYNTHESIZER_PROMPT
from state import Contribution, OrchestratorState
from tools.retrieval import extract_sources


def build_synthesis_context(state: OrchestratorState) -> str:
    parts = [f"Consulta del usuario:\n{user_query(state)}", ""]
    for agent in ("researcher", "analyst"):
        for c in contributions_by(state, agent):
            parts.append(f"Contribución de {agent} (paso {c['iteration']}):\n{c['content']}\n")
    if state.get("current_instruction"):
        parts.append(f"Indicación del supervisor: {state['current_instruction']}")
    if state.get("validation_feedback"):
        parts.append(f"Corrige lo que observó el validador: {state['validation_feedback']}")
    return "\n".join(parts)


def make_synthesizer_node(llm: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    async def synthesizer(state: OrchestratorState) -> dict[str, Any]:
        response = await llm.ainvoke([SystemMessage(SYNTHESIZER_PROMPT), HumanMessage(build_synthesis_context(state))])
        answer = message_text(response)
        contribution: Contribution = {
            "agent": "synthesizer",
            "instruction": state.get("current_instruction", ""),
            "content": answer,
            "sources": extract_sources(answer),
            "iteration": state.get("step_count", 0),
        }
        return {"final_answer": answer, "contributions": [contribution]}

    return synthesizer


def check_answer(state: OrchestratorState) -> list[str]:
    """Rúbrica determinista de la respuesta final. Devuelve los problemas encontrados."""
    answer = state.get("final_answer", "").strip()
    if not answer:
        return ["La respuesta final está vacía."]
    issues = []
    retrieved = {s for c in contributions_by(state, "researcher") for s in c["sources"]}
    cited = set(extract_sources(answer))
    invented = sorted(cited - retrieved)
    if invented:
        issues.append(f"Cita fuentes que ningún agente recuperó: {', '.join(invented)}.")
    if retrieved and not cited:
        issues.append("No cita ninguna fuente aunque hubo evidencia recuperada; agrega [Fuente: archivo].")
    return issues


def validator(state: OrchestratorState) -> dict[str, Any]:
    issues = check_answer(state)
    attempts = state.get("validation_attempts", 0) + 1
    passed = not issues
    can_retry = attempts <= MAX_VALIDATION_RETRIES and state.get("step_count", 0) < MAX_STEPS
    completed = passed or not can_retry
    feedback = " ".join(issues)
    answer = state.get("final_answer", "")
    if not passed and not can_retry:
        answer += f"\n\n> Aviso de validación: {feedback}"
    verdict = "aprobada" if passed else ("rechazada, vuelve al supervisor" if can_retry else "rechazada, sin reintentos")
    contribution: Contribution = {
        "agent": "validator",
        "instruction": "Validar la respuesta final",
        "content": f"Respuesta {verdict}. {feedback}".strip(),
        "sources": [],
        "iteration": state.get("step_count", 0),
    }
    update: dict[str, Any] = {
        "validation_feedback": feedback,
        "validation_attempts": attempts,
        "task_completed": completed,
        "final_answer": answer,
        "contributions": [contribution],
    }
    if completed:
        update["messages"] = [AIMessage(answer, name="synthesizer")]
    return update


def route_from_validator(state: OrchestratorState) -> Literal["supervisor", "__end__"]:
    return END if state.get("task_completed") else "supervisor"
