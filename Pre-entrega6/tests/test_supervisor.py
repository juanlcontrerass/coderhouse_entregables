from typing import Any

from agents.supervisor import SupervisorDecision, apply_guards, build_supervisor_context, route_from_supervisor
from agents.synthesizer import check_answer, route_from_validator
from config import MAX_REVISIONS, MAX_STEPS
from runner import initial_state


def _state(**overrides: Any) -> Any:
    state = initial_state("¿Cuántos días de vacaciones hay?")
    state.update(overrides)
    return state


def _contribution(agent: str, content: str = "dato", sources: list[str] | None = None) -> dict[str, Any]:
    return {"agent": agent, "instruction": "x", "content": content, "sources": sources or [], "iteration": 1}


def test_first_call_to_a_specialist_is_not_a_revision() -> None:
    decision = SupervisorDecision(reasoning="r", next_agent="researcher", instruction="buscar")
    final, forced, revisions = apply_guards(decision, _state())
    assert final.next_agent == "researcher" and not forced and revisions == {}


def test_redelegation_counts_as_revision_until_the_limit() -> None:
    decision = SupervisorDecision(reasoning="r", next_agent="researcher", instruction="refinar")
    state = _state(contributions=[_contribution("researcher")], revisions={"researcher": MAX_REVISIONS - 1})
    final, forced, revisions = apply_guards(decision, state)
    assert final.next_agent == "researcher" and not forced and revisions["researcher"] == MAX_REVISIONS

    state["revisions"] = revisions
    final, forced, _ = apply_guards(decision, state)
    assert final.next_agent == "synthesizer" and forced


def test_step_limit_forces_synthesis() -> None:
    decision = SupervisorDecision(reasoning="r", next_agent="analyst", instruction="calcular")
    final, forced, _ = apply_guards(decision, _state(step_count=MAX_STEPS))
    assert final.next_agent == "synthesizer" and forced


def test_supervisor_context_shows_query_and_contributions() -> None:
    state = _state(contributions=[_contribution("researcher", "14 días", ["politica_vacaciones.md"])])
    context = build_supervisor_context(state)
    assert "¿Cuántos días" in context and "14 días" in context and "politica_vacaciones.md" in context


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
