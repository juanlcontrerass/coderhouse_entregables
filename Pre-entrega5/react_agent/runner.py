"""Ejecución de turnos del agente sobre un hilo persistente."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from react_agent.config import CHECKPOINT_DB, RECURSION_LIMIT

LIMIT_ANSWER = "Alcancé el límite de pasos sin llegar a una conclusión. ¿Podés reformular o acotar la pregunta?"
CANCELLED_TOOL_RESULT = "Llamada cancelada: se alcanzó el límite de pasos."


@dataclass(frozen=True)
class TurnResult:
    thread_id: str
    question: str
    answer: str
    messages: list[BaseMessage]
    limit_reached: bool = False

    @property
    def tool_call_count(self) -> int:
        return sum(isinstance(message, ToolMessage) for message in self.messages)


@asynccontextmanager
async def open_checkpointer(path: Path = CHECKPOINT_DB) -> AsyncIterator[AsyncSqliteSaver]:
    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


async def _close_pending_tool_calls(app: CompiledStateGraph[Any, Any, Any, Any], config: RunnableConfig) -> None:
    """Responde las tool_calls sin resultado: OpenAI rechaza un historial que las tenga."""
    state = await app.aget_state(config)
    last = state.values["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        results = [
            ToolMessage(CANCELLED_TOOL_RESULT, tool_call_id=call["id"], name=call["name"], status="error")
            for call in last.tool_calls
        ]
        await app.aupdate_state(config, {"messages": results}, as_node="tools")


async def ask(
    app: CompiledStateGraph[Any, Any, Any, Any],
    thread_id: str,
    question: str,
    recursion_limit: int = RECURSION_LIMIT,
) -> TurnResult:
    """Ejecuta un turno. Con el mismo thread_id, el agente recuerda los turnos anteriores."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    previous = await app.aget_state(config)
    previous_count = len(previous.values.get("messages", []))

    try:
        result = await app.ainvoke({"messages": [HumanMessage(question)]}, config)
    except GraphRecursionError:
        await _close_pending_tool_calls(app, config)
        state = await app.aget_state(config)
        return TurnResult(thread_id, question, LIMIT_ANSWER, state.values["messages"][previous_count:], True)

    messages = result["messages"]
    return TurnResult(thread_id, question, messages[-1].text, messages[previous_count:])
