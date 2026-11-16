"""Agente de Investigación: busca evidencia en la Vector DB local."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.common import build_specialist, count_tokens, run_specialist
from app.prompts import RESEARCH_PROMPT
from app.state import Contribution, OrchestratorState
from app.tools import RESEARCH_TOOLS
from app.tools.retrieval import extract_sources


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
        return {
            "contributions": [contribution],
            "messages": [AIMessage(content, name="researcher")],
            "tokens_used": count_tokens(messages),
        }

    return researcher
