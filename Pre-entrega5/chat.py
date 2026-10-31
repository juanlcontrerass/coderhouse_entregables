"""Chat interactivo con el agente.

Uso:
    python chat.py                       # sesión nueva
    python chat.py --thread-id mi-hilo   # crea o retoma esa sesión
"""

import argparse
import asyncio
import sys
from uuid import uuid4

from react_agent.graph import build_graph
from react_agent.llm import get_llm
from react_agent.runner import ask, open_checkpointer
from react_agent.trace import format_turn_log

EXIT_WORDS = ("salir", "exit", "quit", "")


async def main(thread_id: str) -> None:
    try:
        llm = get_llm()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(f"💬 Sesión {thread_id} (retomala con --thread-id {thread_id})")
    print("   (escribí 'salir' para terminar)\n")

    async with open_checkpointer() as checkpointer:
        app = build_graph(llm, checkpointer=checkpointer)
        while True:
            try:
                question = (await asyncio.to_thread(input, "🧑 Tu: ")).strip()
            except EOFError:
                break
            if question.lower() in EXIT_WORDS:
                break
            turn = await ask(app, thread_id, question)
            print(f"\n{format_turn_log(turn)}\n")

    print("👋 Listo, la sesión quedó guardada.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Chat con el agente ReAct")
    parser.add_argument("--thread-id", default=f"chat-{uuid4().hex[:8]}", help="sesión a crear o retomar")
    args = parser.parse_args()
    asyncio.run(main(args.thread_id))
