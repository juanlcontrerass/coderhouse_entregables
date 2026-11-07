"""Agente de Análisis/Cómputo: cálculos y sentimiento sobre los datos investigados."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from agents.common import build_specialist, contributions_by, run_specialist
from prompts import ANALYST_PROMPT
from state import Contribution, OrchestratorState
from tools import ANALYST_TOOLS


def build_analyst_task(state: OrchestratorState) -> str:
    """Instrucción del supervisor + solo los datos aportados por el investigador."""
    findings = contributions_by(state, "researcher")
    data = "\n\n".join(f"[Aporte {i}]\n{c['content']}" for i, c in enumerate(findings, 1))
    return (
        f"Tarea: {state['current_instruction']}\n\n"
        f"Datos aportados por el Agente de Investigación:\n{data or '(todavía no hay datos)'}"
    )


def make_analyst_node(llm: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    agent = build_specialist(llm, ANALYST_TOOLS, ANALYST_PROMPT, name="analyst")

    async def analyst(state: OrchestratorState) -> dict[str, Any]:
        content, _ = await run_specialist(agent, build_analyst_task(state))
        contribution: Contribution = {
            "agent": "analyst",
            "instruction": state["current_instruction"],
            "content": content,
            "sources": [],
            "iteration": state.get("step_count", 0),
        }
        return {"contributions": [contribution], "messages": [AIMessage(content, name="analyst")]}

    return analyst
