"""Indexa los documentos de data/ en la Vector DB local (Chroma persistente en vectorstore/).

Uso:
    python ingest.py            # indexa si la colección está vacía
    python ingest.py --rebuild  # borra la colección y vuelve a indexar
"""

import argparse
import sys

from tools.retrieval import build_chroma, buscar_documentos, set_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="reindexa desde cero")
    args = parser.parse_args()

    store = build_chroma(rebuild=args.rebuild)
    set_vectorstore(store)
    count = len(store.get()["ids"])
    print(f"Colección lista: {count} fragmentos indexados.")
    print("\nPrueba de búsqueda ('días de vacaciones por antigüedad'):\n")
    print(buscar_documentos.invoke({"consulta": "días de vacaciones por antigüedad", "k": 2}))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos en la consola de Windows
    main()
