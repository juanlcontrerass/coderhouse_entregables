"""Agente de Investigación: busca evidencia en la Vector DB local."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agents.common import build_specialist, run_specialist
from prompts import RESEARCH_PROMPT
from state import Contribution, OrchestratorState
from tools import RESEARCH_TOOLS
from tools.retrieval import extract_sources


def make_research_node(llm: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    agent = build_specialist(llm, RESEARCH_TOOLS, RESEARCH_PROMPT, name="researcher")

    async def researcher(state: OrchestratorState) -> dict[str, Any]:
        # Contexto mínimo: solo la instrucción del supervisor, no el historial del sistema
        instruction = state["current_instruction"]
        content, messages = await run_specialist(agent, instruction)
        # Las fuentes se toman de lo que la herramienta devolvió, no de lo que dice el LLM
        retrieved = [
            source
            for m in messages
            if isinstance(m, ToolMessage)
            for source in extract_sources(str(m.content))
        ]
        contribution: Contribution = {
            "agent": "researcher",
            "instruction": instruction,
            "content": content,
            "sources": list(dict.fromkeys(retrieved)),
            "iteration": state.get("step_count", 0),
        }
        return {"contributions": [contribution], "messages": [AIMessage(content, name="researcher")]}

    return researcher
