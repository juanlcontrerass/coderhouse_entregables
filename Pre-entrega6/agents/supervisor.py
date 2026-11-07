"""Supervisor: router inteligente y controlador de flujo.

El LLM decide quién interviene (salida estructurada); las condiciones de parada se aplican
en código, después de la decisión, para que un modelo indeciso no pueda generar un bucle.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.common import contributions_by, user_query
from config import MAX_REVISIONS, MAX_STEPS
from prompts import SUPERVISOR_PROMPT
from state import Decision, OrchestratorState, SpecialistName


class SupervisorDecision(BaseModel):
    """Decisión de enrutamiento del supervisor."""

    reasoning: str = Field(description="Justificación breve, aplicando la rúbrica de suficiencia")
    next_agent: Literal["researcher", "analyst", "synthesizer"] = Field(
        description="Quién interviene ahora; synthesizer solo si se cumple la rúbrica"
    )
    instruction: str = Field(description="Instrucción autocontenida para el agente elegido")


def build_supervisor_context(state: OrchestratorState) -> str:
    """Lo que ve el supervisor: consulta, contribuciones de los especialistas y feedback.

    No recibe los mensajes internos de cada agente (búsquedas, llamadas a herramientas).
    """
    lines = [f"Consulta del usuario:\n{user_query(state)}", "", "Contribuciones registradas:"]
    specialists = [c for c in state.get("contributions", []) if c["agent"] in ("researcher", "analyst")]
    if not specialists:
        lines.append("(ninguna todavía)")
    for i, c in enumerate(specialists, 1):
        lines.append(f"[{i}] {c['agent']} (paso {c['iteration']}) — instrucción: {c['instruction']}")
        lines.append(f"Resultado:\n{c['content']}")
        if c["sources"]:
            lines.append(f"Fuentes recuperadas: {', '.join(c['sources'])}")
        lines.append("")
    if state.get("validation_feedback"):
        lines.append(f"Feedback del validador sobre la respuesta anterior: {state['validation_feedback']}")
    lines.append(f"Pasos usados: {state.get('step_count', 0)} de {MAX_STEPS}.")
    return "\n".join(lines)


def apply_guards(decision: SupervisorDecision, state: OrchestratorState) -> tuple[SupervisorDecision, bool, dict[str, int]]:
    """Condiciones de parada deterministas. Devuelve (decisión final, ¿forzada?, revisiones)."""
    revisions = dict(state.get("revisions") or {})
    if state.get("step_count", 0) >= MAX_STEPS:
        return _force_synthesis(f"Límite de {MAX_STEPS} pasos alcanzado."), True, revisions
    agent = decision.next_agent
    if agent != "synthesizer" and contributions_by(state, agent):
        # Ya intervino: esto es un refinamiento y tiene un cupo
        if revisions.get(agent, 0) >= MAX_REVISIONS:
            return _force_synthesis(f"{agent} alcanzó el máximo de {MAX_REVISIONS} refinamientos."), True, revisions
        revisions[agent] = revisions.get(agent, 0) + 1
    return decision, False, revisions


def _force_synthesis(reason: str) -> SupervisorDecision:
    return SupervisorDecision(
        reasoning=f"{reason} Se fuerza la síntesis.",
        next_agent="synthesizer",
        instruction="Sintetiza con la información disponible e indica qué quedó sin resolver.",
    )


def make_supervisor_node(llm: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    router = llm.with_structured_output(SupervisorDecision)

    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        step = state.get("step_count", 0)
        if step >= MAX_STEPS:
            proposed = _force_synthesis(f"Límite de {MAX_STEPS} pasos alcanzado.")
        else:
            raw = await router.ainvoke(
                [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(build_supervisor_context(state))]
            )
            proposed = raw if isinstance(raw, SupervisorDecision) else SupervisorDecision.model_validate(raw)
        decision, forced, revisions = apply_guards(proposed, state)
        record: Decision = {
            "step": step + 1,
            "next_agent": decision.next_agent,
            "instruction": decision.instruction,
            "reasoning": decision.reasoning,
            "forced": forced,
        }
        return {
            "next_agent": decision.next_agent,
            "current_instruction": decision.instruction,
            "step_count": step + 1,
            "revisions": revisions,
            "decisions": [record],
            "messages": [AIMessage(f"[→ {decision.next_agent}] {decision.instruction}", name="supervisor")],
        }

    return supervisor


def route_from_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "synthesizer"]:
    """Arista condicional: el nombre devuelto es el nodo destino."""
    next_agent: SpecialistName = state["next_agent"]
    return next_agent
