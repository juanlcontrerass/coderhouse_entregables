"""Orquestador multi-agente por consola.

Uso:
    python main.py "tu consulta"     # una consulta, muestra la delegación y guarda la traza
    python main.py                   # modo interactivo
    python main.py --demo            # corre las consultas de demostración
    python main.py --graph           # exporta el diagrama (graph.mmd y graph.png)
"""

import argparse
import asyncio
import sys
import uuid
from typing import Any

from graph import build_graph
from observability import setup_tracing, shutdown_tracing
from runner import describe_event, export_diagram, run_query, save_trace

DEMO_QUERIES = [
    "Según la política de vacaciones, ¿cuántos días corresponden a empleados con 3, 7 y 12 "
    "años de antigüedad? Calcula el promedio y en qué porcentaje supera el de 12 años al de 3.",
    "Busca los comentarios abiertos de la encuesta de clima 2025 y analiza su sentimiento. "
    "Además, calcula la variación porcentual del eNPS respecto de 2024.",
]


def print_event(node: str, update: dict[str, Any]) -> None:
    print(describe_event(node, update), flush=True)


# Agrupa en una misma sesión de Phoenix las consultas de esta ejecución (demo o modo interactivo)
SESSION_ID = f"cli-{uuid.uuid4().hex[:8]}"


async def answer(graph: Any, query: str) -> None:
    print(f"\n=== Consulta: {query}\n")
    state = await run_query(graph, query, on_event=print_event, session_id=SESSION_ID)
    print("\n=== Respuesta final\n")
    print(state.get("final_answer", "(sin respuesta)"))
    print(f"\nDelegaciones del supervisor: {' → '.join(d['next_agent'] for d in state['decisions'])}")
    print(f"Traza guardada en {save_trace(query, state)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="consulta a resolver")
    parser.add_argument("--demo", action="store_true", help="ejecuta las consultas de demostración")
    parser.add_argument("--graph", action="store_true", help="exporta el diagrama del grafo y termina")
    args = parser.parse_args()

    if args.graph:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        # Compilar el grafo no llama al modelo: el diagrama no necesita una API key real
        offline_llm = ChatOpenAI(model="gpt-5.4-mini", api_key=SecretStr("sin-clave"))
        for path in export_diagram(build_graph(offline_llm)):
            print(f"Generado {path}")
        return

    from llm import get_llm

    # Antes de crear el LLM y el grafo, para que los instrumentores envuelvan los clientes
    setup_tracing()
    try:
        graph = build_graph(get_llm())
        if args.demo:
            for query in DEMO_QUERIES:
                await answer(graph, query)
        elif args.query:
            await answer(graph, args.query)
        else:
            print("Orquestador multi-agente (Supervisor + Investigador + Analista). Pulsa Enter sin escribir nada para salir.")
            while query := input("\nConsulta> ").strip():
                await answer(graph, query)
    finally:
        shutdown_tracing()  # vacía el BatchSpanProcessor: sin esto se pierden los últimos spans


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos en la consola de Windows
    asyncio.run(main())
