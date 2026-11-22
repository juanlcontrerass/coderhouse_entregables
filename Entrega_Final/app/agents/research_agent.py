"""Agente de Investigación: busca evidencia con el RAG híbrido."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import ValidationError

from app.agents.common import build_specialist, count_tokens, run_specialist, usage
from app.graph.prompts import RESEARCH_PROMPT
from app.graph.state import Contribution, OrchestratorState
from app.tools import RESEARCH_TOOLS
from app.tools.retrieval import extract_sources
from app.tools.schemas import ResultadoBusqueda


def retrieved_sources(messages: list[BaseMessage]) -> list[str]:
    """Fuentes que la herramienta devolvió de verdad (del artifact validado, no del LLM)."""
    sources: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "buscar_documentos":
            continue
        try:
            sources += ResultadoBusqueda.model_validate(message.artifact).fuentes
        except ValidationError:  # sin artifact (p. ej., error de validación): se lee el texto
            sources += extract_sources(str(message.content))
    return list(dict.fromkeys(sources))


def make_research_node(llm: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    agent = build_specialist(llm, RESEARCH_TOOLS, RESEARCH_PROMPT, name="researcher")

    async def researcher(state: OrchestratorState) -> dict[str, Any]:
        # Contexto mínimo: solo la instrucción del supervisor, no el historial del sistema
        instruction = state["current_instruction"]
        content, messages = await run_specialist(agent, instruction)
        contribution: Contribution = {
            "agent": "researcher",
            "instruction": instruction,
            "content": content,
            "sources": retrieved_sources(messages),
            "iteration": state.get("step_count", 0),
        }
        return {
            "contributions": [contribution],
            "messages": [AIMessage(content, name="researcher")],
            **usage(count_tokens(messages)),
        }

    return researcher
