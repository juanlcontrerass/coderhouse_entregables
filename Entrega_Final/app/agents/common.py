"""Utilidades compartidas por los nodos de agentes."""

import warnings
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from app.core.config import get_settings
from app.graph.state import Contribution, OrchestratorState

with warnings.catch_warnings():
    # En LangGraph 1.x el sucesor de create_react_agent es langchain.agents.create_agent;
    # se mantiene el del Módulo 6 porque sigue siendo funcional.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langgraph.prebuilt import create_react_agent

FINAL_ANSWER_AUTHOR = "synthesizer"  # nombre del AIMessage que cierra cada turno


def message_text(message: BaseMessage) -> str:
    """Texto plano de un mensaje (OpenAI puede devolver el contenido en bloques)."""
    return str(message.text).strip()


def user_query(state: OrchestratorState) -> str:
    """La consulta del turno actual (el último mensaje humano)."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def conversation_turns(messages: Sequence[BaseMessage]) -> list[tuple[str, str | None]]:
    """Pares (pregunta, respuesta final) en orden; la respuesta es None si el turno no terminó."""
    turns: list[tuple[str, str | None]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append((message_text(message), None))
        elif isinstance(message, AIMessage) and message.name == FINAL_ANSWER_AUTHOR and turns:
            turns[-1] = (turns[-1][0], message_text(message))
    return turns


def history_block(state: OrchestratorState, max_turns: int | None = None) -> str:
    """Turnos anteriores al actual, ya respondidos, como texto para el contexto del LLM."""
    limit = get_settings().history_turns if max_turns is None else max_turns
    previous = [t for t in conversation_turns(state["messages"])[:-1] if t[1]]
    if not limit or not previous:
        return ""
    lines = ["Historial reciente de la conversación:"]
    for question, answer in previous[-limit:]:
        lines.append(f"- Usuario: {question}\n  Respuesta: {answer}")
    return "\n".join(lines)


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    """Suma los tokens (entrada + salida) informados por el proveedor en los mensajes de IA."""
    total = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            total += int(usage.get("total_tokens", 0))
    return total


def usage(tokens: int) -> dict[str, int]:
    """Actualización de estado con los tokens del nodo (turno y conversación)."""
    return {"tokens_used": tokens, "total_tokens": tokens}


def contributions_by(state: OrchestratorState, agent: str) -> list[Contribution]:
    return [c for c in state.get("contributions") or [] if c["agent"] == agent]


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
            {"messages": [HumanMessage(task)]}, config={"recursion_limit": get_settings().agent_recursion_limit}
        )
    except GraphRecursionError:
        return "No se pudo completar la tarea dentro del límite de pasos.", []
    messages: list[BaseMessage] = result["messages"]
    return message_text(messages[-1]) or "(sin respuesta)", messages
