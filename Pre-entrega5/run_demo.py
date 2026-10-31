"""Corre los escenarios de la consigna contra OpenAI y guarda la traza en traces/.

Uso:
    python run_demo.py
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from react_agent.config import CHECKPOINT_DB, TRACES_DIR
from react_agent.graph import build_graph
from react_agent.llm import get_llm
from react_agent.runner import TurnResult, ask, open_checkpointer
from react_agent.trace import format_turn_log, write_trace

# (escenario, hilo, pregunta). "memoria" comparte hilo con "multi_paso".
SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("multi_paso", "pedidos", "¿Cuántos pedidos tuvo Lucía Fernández y cuál fue el total?"),
    ("memoria", "pedidos", "¿Y el último?"),
    ("reintento", "reintento", "¿Cuánto gastó en total Lusia Fernandes?"),
    ("aclaracion", "aclaracion", "¿Cuántos pedidos tuvo Roberto Sánchez?"),
)


async def run_scenarios(llm: BaseChatModel, db_path: Path = CHECKPOINT_DB) -> dict[str, list[TurnResult]]:
    """Ejecuta cada turno con su propia conexión a SQLite.

    Así el escenario "memoria" solo puede recordar lo que quedó guardado en disco.
    """
    run_id = uuid4().hex[:8]
    results: dict[str, list[TurnResult]] = {}
    for scenario, thread, question in SCENARIOS:
        async with open_checkpointer(db_path) as checkpointer:
            app = build_graph(llm, checkpointer=checkpointer)
            results.setdefault(scenario, []).append(await ask(app, f"demo-{run_id}-{thread}", question))
    return results


async def main() -> None:
    try:
        llm = get_llm()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    scenarios = await run_scenarios(llm)

    for name, turns in scenarios.items():
        for turn in turns:
            print(f"\n=== {name} ===\n{format_turn_log(turn)}")

    multi_step_calls = scenarios["multi_paso"][0].tool_call_count
    if multi_step_calls < 2:
        print(f"\n⚠️  El escenario multi_paso usó {multi_step_calls} herramienta(s); la consigna pide al menos 2.")

    write_trace(scenarios, TRACES_DIR)
    print(f"\n✅ Traza guardada en {TRACES_DIR}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    asyncio.run(main())
