import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fakes import ScriptedChatModel, tool_call
from react_agent.graph import build_graph
from react_agent.runner import ask, open_checkpointer


async def ask_once(
    llm: ScriptedChatModel, db: Path, thread_id: str, question: str, recursion_limit: int = 10
):
    """Abre y cierra SQLite en cada turno, como si fuera otra ejecución del programa."""
    async with open_checkpointer(db) as checkpointer:
        return await ask(build_graph(llm, checkpointer=checkpointer), thread_id, question, recursion_limit)


def test_ask_returns_answer_and_counts_tool_calls(tmp_path: Path):
    llm = ScriptedChatModel(
        responses=[
            tool_call("buscar_cliente", nombre="Lucía Fernández"),
            tool_call("listar_pedidos", call_id="call_2", cliente_id=102),
            AIMessage("3 pedidos"),
        ]
    )
    turn = asyncio.run(ask_once(llm, tmp_path / "c.sqlite", "t1", "¿Cuántos pedidos?"))
    assert turn.answer == "3 pedidos"
    assert turn.tool_call_count == 2
    assert not turn.limit_reached


def test_same_thread_remembers_after_reopening_sqlite(tmp_path: Path):
    db = tmp_path / "c.sqlite"
    llm = ScriptedChatModel(responses=[AIMessage("uno"), AIMessage("dos")])
    asyncio.run(ask_once(llm, db, "t1", "primera"))
    asyncio.run(ask_once(llm, db, "t1", "segunda"))
    contents = [m.content for m in llm.received[1] if isinstance(m, HumanMessage)]
    assert contents == ["primera", "segunda"]


def test_other_thread_does_not_see_the_conversation(tmp_path: Path):
    db = tmp_path / "c.sqlite"
    llm = ScriptedChatModel(responses=[AIMessage("uno"), AIMessage("dos")])
    asyncio.run(ask_once(llm, db, "t1", "primera"))
    asyncio.run(ask_once(llm, db, "t2", "segunda"))
    contents = [m.content for m in llm.received[1] if isinstance(m, HumanMessage)]
    assert contents == ["segunda"]


def test_turn_only_contains_messages_of_that_turn(tmp_path: Path):
    db = tmp_path / "c.sqlite"
    llm = ScriptedChatModel(responses=[AIMessage("uno"), AIMessage("dos")])
    asyncio.run(ask_once(llm, db, "t1", "primera"))
    turn = asyncio.run(ask_once(llm, db, "t1", "segunda"))
    assert [m.content for m in turn.messages] == ["segunda", "dos"]


def test_recursion_limit_is_reported_instead_of_raising(tmp_path: Path):
    llm = ScriptedChatModel(responses=[tool_call("buscar_cliente", nombre="Lucía Fernández")])
    turn = asyncio.run(ask_once(llm, tmp_path / "c.sqlite", "t1", "pregunta"))
    assert turn.limit_reached
    assert "límite" in turn.answer
    assert len(llm.received) == 5  # techo de 10 pasos: 5 modelo + 5 herramientas


def test_thread_stays_usable_when_limit_leaves_a_tool_call_unanswered(tmp_path: Path):
    db = tmp_path / "c.sqlite"
    looping = [tool_call("buscar_cliente", call_id=f"call_{i}", nombre="Lucía Fernández") for i in range(5)]
    llm = ScriptedChatModel(responses=[*looping, AIMessage("sigo acá")])

    # Con un techo impar el grafo se detiene justo después del modelo, con la tool_call pendiente
    first = asyncio.run(ask_once(llm, db, "t1", "pregunta", recursion_limit=9))
    assert first.limit_reached
    second = asyncio.run(ask_once(llm, db, "t1", "otra"))

    assert second.answer == "sigo acá"
    sent = llm.received[-1]
    assert "call_4" in {tc["id"] for m in sent if isinstance(m, AIMessage) for tc in m.tool_calls}
    answered = {m.tool_call_id for m in sent if isinstance(m, ToolMessage)}
    requested = {tc["id"] for m in sent if isinstance(m, AIMessage) for tc in m.tool_calls}
    assert requested <= answered
