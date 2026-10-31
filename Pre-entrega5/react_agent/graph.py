"""Grafo ReAct: modelo → (¿herramienta?) → herramienta → modelo → ... → fin."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from react_agent.config import MAX_CONTEXT_TOKENS
from react_agent.prompts import SYSTEM_PROMPT
from react_agent.state import AgentState
from react_agent.tools import TOOLS


def build_context(
    messages: Sequence[BaseMessage], max_tokens: int = MAX_CONTEXT_TOKENS
) -> list[BaseMessage]:
    """Arma lo que se envía al modelo: el system prompt y los turnos más recientes.

    Empieza siempre en un mensaje de usuario para no dejar resultados de herramientas
    sin su llamada, que OpenAI rechazaría. Si el turno actual ya supera el tope, se
    envía completo: sin él, el modelo no tendría qué responder.
    """
    trimmed = trim_messages(
        list(messages),
        max_tokens=max_tokens,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
    )
    return [SystemMessage(SYSTEM_PROMPT), *(trimmed or messages)]


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool] = TOOLS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = await llm_with_tools.ainvoke(build_context(state["messages"], max_context_tokens))
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")  # el ciclo: el modelo razona sobre el resultado
    return graph.compile(checkpointer=checkpointer)
