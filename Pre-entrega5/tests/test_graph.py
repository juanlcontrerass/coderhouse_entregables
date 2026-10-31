import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState

from fakes import ScriptedChatModel, tool_call
from react_agent.graph import build_context, build_graph
from react_agent.state import AgentState


def run_graph(llm: ScriptedChatModel, question: str = "pregunta") -> list:
    app = build_graph(llm)
    result = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 10})
    )
    return result["messages"]


def test_state_inherits_from_messages_state():
    assert MessagesState in AgentState.__orig_bases__


def test_llm_is_bound_to_all_tools():
    llm = ScriptedChatModel(responses=[AIMessage("hola")])
    build_graph(llm)
    assert llm.bound_tools == ["buscar_cliente", "listar_pedidos", "detalle_pedido"]


def test_answer_without_tools_ends_after_one_model_call():
    llm = ScriptedChatModel(responses=[AIMessage("hola")])
    messages = run_graph(llm)
    assert len(llm.received) == 1
    assert messages[-1].content == "hola"


def test_cycle_runs_two_tools_before_answering():
    llm = ScriptedChatModel(
        responses=[
            tool_call("buscar_cliente", nombre="Lucía Fernández"),
            tool_call("listar_pedidos", call_id="call_2", cliente_id=102),
            AIMessage("3 pedidos por $14.500"),
        ]
    )
    messages = run_graph(llm)
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["buscar_cliente", "listar_pedidos"]
    assert '"total": 14500' in tool_messages[1].content
    assert messages[-1].content == "3 pedidos por $14.500"


def test_model_sees_tool_error_and_can_retry():
    llm = ScriptedChatModel(
        responses=[
            tool_call("buscar_cliente", nombre="Lusia Fernandes"),
            tool_call("buscar_cliente", call_id="call_2", nombre="Lucía Fernández"),
            AIMessage("listo"),
        ]
    )
    run_graph(llm)
    seen_on_retry = llm.received[1][-1]
    assert isinstance(seen_on_retry, ToolMessage)
    assert "Lucía Fernández" in seen_on_retry.content
    assert "error" in seen_on_retry.content


def test_invalid_tool_arguments_return_to_the_model_instead_of_crashing():
    llm = ScriptedChatModel(
        responses=[tool_call("listar_pedidos", cliente_id="no-es-un-numero"), AIMessage("perdón")]
    )
    messages = run_graph(llm)
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].status == "error"
    assert messages[-1].content == "perdón"


def test_system_prompt_reaches_the_model_but_is_not_stored_in_state():
    llm = ScriptedChatModel(responses=[AIMessage("hola")])
    messages = run_graph(llm)
    assert isinstance(llm.received[0][0], SystemMessage)
    assert not any(isinstance(m, SystemMessage) for m in messages)


def test_context_drops_oldest_turns_when_over_the_limit():
    history = []
    for i in range(50):
        history += [HumanMessage(f"pregunta {i} " + "x" * 400), AIMessage(f"respuesta {i}")]
    history.append(HumanMessage("última"))

    context = build_context(history, max_tokens=500)

    assert isinstance(context[0], SystemMessage)
    assert isinstance(context[1], HumanMessage)
    assert context[-1].content == "última"
    assert len(context) < len(history)


def test_context_never_starts_on_an_orphan_tool_message():
    history = [
        HumanMessage("vieja " + "x" * 2000),
        tool_call("buscar_cliente", nombre="x" * 2000),
        ToolMessage("resultado", tool_call_id="call_1", name="buscar_cliente"),
        AIMessage("respuesta"),
        HumanMessage("nueva"),
    ]
    context = build_context(history, max_tokens=300)
    assert [type(m) for m in context] == [SystemMessage, HumanMessage]
    assert context[1].content == "nueva"


def test_context_keeps_current_turn_when_it_alone_exceeds_the_limit():
    history = [HumanMessage("x" * 5000)]
    context = build_context(history, max_tokens=10)
    assert context[-1] is history[0]


def test_trimming_does_not_remove_messages_from_state():
    llm = ScriptedChatModel(responses=[AIMessage("ok")])
    app = build_graph(llm, max_context_tokens=100)
    history = [HumanMessage("vieja " + "x" * 2000), AIMessage("r"), HumanMessage("nueva")]
    result = asyncio.run(app.ainvoke({"messages": history}, {"recursion_limit": 10}))
    assert len(llm.received[0]) < len(history) + 1
    assert result["messages"][0].content.startswith("vieja")
