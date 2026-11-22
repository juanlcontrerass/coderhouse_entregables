from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.supervisor import SupervisorDecision, apply_guards, build_supervisor_context, route_from_supervisor
from app.agents.synthesizer import check_answer, route_from_validator
from app.core.config import get_settings
from app.graph.turn import turn_input


def _state(**overrides: Any) -> Any:
    state = turn_input("¿Cuántos días de vacaciones hay?")
    state.update(contributions=[], decisions=[], tokens_used=0)
    state.update(overrides)
    return state


def _contribution(agent: str, content: str = "dato", sources: list[str] | None = None) -> dict[str, Any]:
    return {"agent": agent, "instruction": "x", "content": content, "sources": sources or [], "iteration": 1}


def test_first_call_to_a_specialist_is_not_a_revision() -> None:
    decision = SupervisorDecision(reasoning="r", next_agent="researcher", instruction="buscar")
    final, forced, revisions = apply_guards(decision, _state())
    assert final.next_agent == "researcher" and not forced and revisions == {}


def test_redelegation_counts_as_revision_until_the_limit() -> None:
    max_revisions = get_settings().max_revisions
    decision = SupervisorDecision(reasoning="r", next_agent="researcher", instruction="refinar")
    state = _state(contributions=[_contribution("researcher")], revisions={"researcher": max_revisions - 1})
    final, forced, revisions = apply_guards(decision, state)
    assert final.next_agent == "researcher" and not forced and revisions["researcher"] == max_revisions

    state["revisions"] = revisions
    final, forced, _ = apply_guards(decision, state)
    assert final.next_agent == "synthesizer" and forced


def test_step_limit_forces_synthesis() -> None:
    decision = SupervisorDecision(reasoning="r", next_agent="analyst", instruction="calcular")
    final, forced, _ = apply_guards(decision, _state(step_count=get_settings().max_steps))
    assert final.next_agent == "synthesizer" and forced


def test_supervisor_context_shows_query_and_contributions() -> None:
    state = _state(contributions=[_contribution("researcher", "14 días", ["politica_vacaciones.md"])])
    context = build_supervisor_context(state)
    assert "¿Cuántos días" in context and "14 días" in context and "politica_vacaciones.md" in context
    assert "Historial" not in context  # primer turno: sin historial


def test_supervisor_context_includes_previous_turns_only() -> None:
    messages = [
        HumanMessage("Primera pregunta"),
        AIMessage("[→ researcher] buscar", name="supervisor"),
        AIMessage("Primera respuesta", name="synthesizer"),
        HumanMessage("¿Y en porcentaje?"),
    ]
    context = build_supervisor_context(_state(messages=messages))
    assert "Primera pregunta" in context and "Primera respuesta" in context
    assert "[→ researcher]" not in context  # los mensajes internos no forman parte del historial
    assert context.index("Historial") < context.index("Consulta del usuario:\n¿Y en porcentaje?")


def test_routes() -> None:
    assert route_from_supervisor(_state(next_agent="analyst")) == "analyst"
    assert route_from_validator(_state(task_completed=True)) == "__end__"
    assert route_from_validator(_state(task_completed=False)) == "supervisor"


def test_check_answer_detects_invented_and_missing_sources() -> None:
    researched = [_contribution("researcher", sources=["politica_vacaciones.md"])]
    assert check_answer(_state(contributions=researched, final_answer="14 días [Fuente: politica_vacaciones.md]")) == []
    invented = check_answer(_state(contributions=researched, final_answer="14 días [Fuente: otro.md]"))
    assert "otro.md" in invented[0]
    uncited = check_answer(_state(contributions=researched, final_answer="14 días"))
    assert "No cita" in uncited[0]
    assert check_answer(_state(final_answer="  ")) == ["La respuesta final está vacía."]
