"""Conversaciones multi-turno sobre el checkpointer: memoria entre turnos y estado por turno."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.common import conversation_turns
from app.graph.state import add_or_reset, extend_or_reset
from app.graph.turn import graph_config, turn_input
from app.storage.checkpointer import load_messages
from tests.fakes import decide, tool_call
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _graph, _models, vacation_store  # noqa: F401

FOLLOW_UP = "¿Y cuántos días serían en total sumando los tres casos?"
SECOND_ANSWER = AIMessage("En total son 63 días [Fuente: politica_vacaciones.md]")
USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_reducers_accumulate_and_reset() -> None:
    assert extend_or_reset([1], [2]) == [1, 2]
    assert extend_or_reset([1, 2], None) == []
    assert add_or_reset(5, 3) == 8 and add_or_reset(5, None) == 0


async def test_second_turn_keeps_history_and_resets_turn_state() -> None:
    supervisor = HAPPY_SUPERVISOR + [
        decide("researcher", "Busca de nuevo los días por antigüedad"),
        decide("analyst", "Suma 14, 21 y 28"),
        decide("synthesizer", "Responde la repregunta"),
    ]
    models = _models(supervisor, [AIMessage(GOOD_ANSWER.content, usage_metadata=USAGE), SECOND_ANSWER])
    search, finding = models["researcher"].responses  # el investigador vuelve a buscar en el turno 2
    models["researcher"].responses = [search, finding, tool_call("buscar_documentos", "call_2", consulta="vacaciones"), finding]
    checkpointer = InMemorySaver()
    graph = _graph(models, checkpointer)
    config = graph_config("conv-1", "job-1")

    first = await graph.ainvoke(turn_input(QUERY), config)
    second = await graph.ainvoke(turn_input(FOLLOW_UP), graph_config("conv-1", "job-2"))

    # Estado del turno: se reinicia (no arrastra contribuciones ni decisiones del turno anterior)
    assert [c["agent"] for c in second["contributions"]] == ["researcher", "analyst", "synthesizer", "validator"]
    assert len(second["decisions"]) == 3 and second["final_answer"] == SECOND_ANSWER.content
    assert first["tokens_used"] == 15 and second["tokens_used"] == 0
    assert second["total_tokens"] == 15  # acumulado de la conversación

    # Memoria: la conversación tiene los dos turnos, en orden, con sus respuestas
    turns = conversation_turns(second["messages"])
    assert turns == [(QUERY, GOOD_ANSWER.content), (FOLLOW_UP, SECOND_ANSWER.content)]
    assert conversation_turns(await load_messages(checkpointer, "conv-1")) == turns
    assert await load_messages(checkpointer, "otra") == []

    # El supervisor y el sintetizador ven el historial; los especialistas no
    supervisor_context = str(models["supervisor"].received[3][-1].content)
    assert "Historial reciente" in supervisor_context and GOOD_ANSWER.content in supervisor_context
    assert FOLLOW_UP in supervisor_context
    assert "Historial reciente" in str(models["synthesizer"].received[-1][-1].content)
    assert "Historial reciente" not in str(models["researcher"].received[-1][1].content)


async def test_history_survives_a_new_graph_instance() -> None:
    """Simula el reinicio del worker: un grafo nuevo sobre el mismo checkpointer conserva la conversación."""
    checkpointer = InMemorySaver()
    await _graph(_models(HAPPY_SUPERVISOR, [GOOD_ANSWER]), checkpointer).ainvoke(turn_input(QUERY), graph_config("conv-2"))

    restarted = _graph(_models(HAPPY_SUPERVISOR, [SECOND_ANSWER]), checkpointer)
    state = await restarted.ainvoke(turn_input(FOLLOW_UP), graph_config("conv-2"))
    assert [q for q, _ in conversation_turns(state["messages"])] == [QUERY, FOLLOW_UP]
