"""Ejecución de una consulta, traza de delegación y exportación del diagrama."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from config import BASE_DIR, RECURSION_LIMIT, TRACES_DIR

EventHandler = Callable[[str, dict[str, Any]], None]


def initial_state(query: str) -> dict[str, Any]:
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
    }


async def run_query(graph: Any, query: str, on_event: EventHandler | None = None) -> dict[str, Any]:
    """Ejecuta el grafo; `on_event(nodo, actualización)` se llama a medida que avanza."""
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        initial_state(query),
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk
        elif on_event:
            for node, update in chunk.items():
                on_event(node, update or {})
    return final


def _shorten(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def describe_event(node: str, update: dict[str, Any]) -> str:
    """Una línea legible por cada paso del grafo."""
    if node == "supervisor":
        d = update["decisions"][-1]
        forced = " (forzado por guarda)" if d["forced"] else ""
        return (
            f"[supervisor · paso {d['step']}] → {d['next_agent']}{forced}\n"
            f"    motivo: {_shorten(d['reasoning'], 250)}\n"
            f"    instrucción: {_shorten(d['instruction'], 250)}"
        )
    contribution = (update.get("contributions") or [{}])[-1]
    content = _shorten(contribution.get("content", ""))
    if node == "researcher":
        sources = ", ".join(contribution.get("sources", [])) or "ninguna"
        return f"[researcher] {content}\n    fuentes recuperadas: {sources}"
    if node == "validator":
        return f"[validator] {content}"
    return f"[{node}] {content}"


def build_trace(query: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "route": [d["next_agent"] for d in state.get("decisions", [])],
        "steps": state.get("step_count", 0),
        "task_completed": state.get("task_completed", False),
        "decisions": state.get("decisions", []),
        "contributions": state.get("contributions", []),
        "final_answer": state.get("final_answer", ""),
    }


def save_trace(query: str, state: dict[str, Any], traces_dir: Path = TRACES_DIR) -> Path:
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"trace_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(build_trace(query, state), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_diagram(graph: Any, out_dir: Path = BASE_DIR) -> list[Path]:
    """Guarda el Mermaid del grafo y, si hay internet (usa mermaid.ink), el PNG."""
    drawable = graph.get_graph()
    written = [out_dir / "graph.mmd"]
    written[0].write_text(drawable.draw_mermaid(), encoding="utf-8")
    try:
        png = out_dir / "graph.png"
        png.write_bytes(drawable.draw_mermaid_png())
        written.append(png)
    except Exception as exc:  # sin red o servicio caído: el .mmd alcanza
        print(f"No se pudo generar graph.png ({exc}); queda graph.mmd.")
    return written
