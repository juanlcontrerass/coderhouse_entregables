"""Estado inicial del grafo y ejecución directa (sin API), usada por los tests."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import RECURSION_LIMIT

EventHandler = Callable[[str, dict[str, Any]], None]


def initial_state(query: str, publish: bool = False) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(query)],
        "current_instruction": "",
        "contributions": [],
        "decisions": [],
        "step_count": 0,
        "revisions": {},
        "validation_feedback": "",
        "validation_attempts": 0,
        "final_answer": "",
        "task_completed": False,
        "tokens_used": 0,
        "publish_requested": publish,
        "risk_reasons": [],
        "approval": {},
        "published_path": "",
    }


async def run_query(
    graph: Any,
    query: str,
    on_event: EventHandler | None = None,
    *,
    publish: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta el grafo; `on_event(nodo, actualización)` se llama a medida que avanza."""
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        initial_state(query, publish),
        config={"recursion_limit": RECURSION_LIMIT, **(config or {})},
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk
        elif on_event:
            for node, update in chunk.items():
                if not node.startswith("__"):  # omite el evento __interrupt__
                    on_event(node, update or {})
    return final
