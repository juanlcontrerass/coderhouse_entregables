"""Entrada de cada turno y ejecución directa del grafo (sin API), usada por los tests y scripts."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from app.core.config import get_settings

EventHandler = Callable[[str, dict[str, Any]], None]


def turn_input(query: str, publish: bool = False) -> dict[str, Any]:
    """Actualización con la que empieza un turno.

    `messages` y `total_tokens` se acumulan (son de la conversación); el resto se reinicia.
    `None` vacía los campos con reducer acumulativo (ver `app/graph/state.py`).
    """
    return {
        "messages": [HumanMessage(query)],
        "current_instruction": "",
        "contributions": None,
        "decisions": None,
        "step_count": 0,
        "revisions": {},
        "validation_feedback": "",
        "validation_attempts": 0,
        "final_answer": "",
        "task_completed": False,
        "tokens_used": None,
        "total_tokens": 0,
        "publish_requested": publish,
        "risk_reasons": [],
        "approval": {},
        "published_path": "",
    }


def graph_config(conversation_id: str, job_id: str | None = None, **metadata: Any) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": conversation_id, "job_id": job_id or conversation_id},
        "recursion_limit": get_settings().recursion_limit,
        "metadata": {"conversation_id": conversation_id, "job_id": job_id or conversation_id, **metadata},
    }


async def run_turn(
    graph: Any,
    query: str,
    on_event: EventHandler | None = None,
    *,
    publish: bool = False,
    conversation_id: str = "local",
) -> dict[str, Any]:
    """Ejecuta un turno; `on_event(nodo, actualización)` se llama a medida que avanza."""
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        turn_input(query, publish),
        config=graph_config(conversation_id),
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk
        elif on_event:
            for node, update in chunk.items():
                if not node.startswith("__"):  # omite el evento __interrupt__
                    on_event(node, update or {})
    return final
