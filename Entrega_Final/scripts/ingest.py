"""(Re)construye el índice de Chroma y prueba la búsqueda híbrida.

El worker indexa el corpus automáticamente al arrancar si el índice está vacío o
desactualizado; este script sirve para forzar la reconstrucción o inspeccionar resultados.

Uso:
    python scripts/ingest.py --rebuild
    python scripts/ingest.py --query "días de vacaciones con 7 años de antigüedad"
    docker compose exec worker python scripts/ingest.py --rebuild
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.retriever import build_retriever  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="borra y vuelve a crear el índice")
    parser.add_argument("--query", help="consulta de prueba")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    retriever = await build_retriever(rebuild=args.rebuild)
    print(f"Índice listo: {len(retriever.chunks)} fragmentos.")
    if args.query:
        for hit in await retriever.asearch(args.query, args.k):
            print(f"\n[{hit.score:.4f}] {hit.chunk_id}\n{hit.document.page_content[:300]}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
