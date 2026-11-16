"""Utilidades compartidas por los nodos de agentes."""

import warnings
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from app.config import AGENT_RECURSION_LIMIT
from app.state import Contribution, OrchestratorState

with warnings.catch_warnings():
    # La consigna pide create_react_agent; en LangGraph 1.x su sucesor es
    # langchain.agents.create_agent, pero sigue siendo funcional.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langgraph.prebuilt import create_react_agent


def message_text(message: BaseMessage) -> str:
    """Texto plano de un mensaje (OpenAI puede devolver el contenido en bloques)."""
    return str(message.text).strip()


def user_query(state: OrchestratorState) -> str:
    """La última consulta del usuario."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    """Suma los tokens (entrada + salida) informados por el proveedor en los mensajes de IA."""
    total = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            total += int(usage.get("total_tokens", 0))
    return total


def contributions_by(state: OrchestratorState, agent: str) -> list[Contribution]:
    return [c for c in state.get("contributions", []) if c["agent"] == agent]


def build_specialist(llm: BaseChatModel, tools: Sequence[BaseTool], prompt: str, name: str) -> Any:
    """Agente ReAct con herramientas acotadas a su rol."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_react_agent(llm, tools=list(tools), prompt=prompt, name=name)


async def run_specialist(agent: Any, task: str) -> tuple[str, list[BaseMessage]]:
    """Ejecuta el agente con un único mensaje (la tarea) y devuelve su respuesta final.

    Si agota sus pasos, devuelve un aviso en lugar de romper el grafo: el supervisor lo
    verá como contribución incompleta y podrá decidir.
    """
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(task)]}, config={"recursion_limit": AGENT_RECURSION_LIMIT}
        )
    except GraphRecursionError:
        return "No pude completar la tarea dentro del límite de pasos.", []
    messages: list[BaseMessage] = result["messages"]
    return message_text(messages[-1]) or "(sin respuesta)", messages
