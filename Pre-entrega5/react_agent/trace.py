"""Exporta los turnos del agente como traza ReAct en JSON y en texto."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from react_agent.runner import TurnResult


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    entry: dict[str, Any] = {"tipo": type(message).__name__, "contenido": message.text}
    if isinstance(message, AIMessage) and message.tool_calls:
        entry["tool_calls"] = [{"nombre": call["name"], "argumentos": call["args"]} for call in message.tool_calls]
    if isinstance(message, ToolMessage):
        entry["herramienta"] = message.name
    return entry


def turn_to_dict(turn: TurnResult) -> dict[str, Any]:
    return {
        "thread_id": turn.thread_id,
        "pregunta": turn.question,
        "respuesta": turn.answer,
        "herramientas_invocadas": turn.tool_call_count,
        "limite_alcanzado": turn.limit_reached,
        "mensajes": [_message_to_dict(message) for message in turn.messages],
    }


def format_turn_log(turn: TurnResult) -> str:
    lines: list[str] = []
    for message in turn.messages:
        if isinstance(message, HumanMessage):
            lines.append(f'Usuario: "{message.text}"')
        elif isinstance(message, ToolMessage):
            lines.append(f"-> La herramienta devuelve: {message.text}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            if text := message.text:
                lines.append(f"-> El agente razona: {text}")
            for call in message.tool_calls:
                args = ", ".join(f"{name}={value!r}" for name, value in call["args"].items())
                lines.append(f"-> El agente decide usar la herramienta: {call['name']}({args})")
    lines.append(f"Respuesta: {turn.answer}")
    return "\n".join(lines)


def write_trace(scenarios: Mapping[str, Sequence[TurnResult]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: [turn_to_dict(turn) for turn in turns] for name, turns in scenarios.items()}
    (out_dir / "traza_ejecucion.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    sections = []
    for name, turns in scenarios.items():
        logs = "\n\n".join(f"[thread_id={turn.thread_id}]\n{format_turn_log(turn)}" for turn in turns)
        sections.append(f"=== {name} ===\n{logs}")
    (out_dir / "traza_ejecucion.log").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
