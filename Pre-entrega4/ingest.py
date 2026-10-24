"""Pipeline de ingesta: carga data/ (Markdown, PDF, JSON), fragmenta y sube a Pinecone.

Uso:
    python ingest.py            # indexa solo si el namespace está vacío
    python ingest.py --rebuild  # vacía el namespace y vuelve a indexar todo
"""

import argparse

from rag_pinecone.ingest import run_ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa data/ en Pinecone Serverless")
    parser.add_argument("--rebuild", action="store_true", help="vacía el namespace y reindexa")
    args = parser.parse_args()
    run_ingest(rebuild=args.rebuild)
