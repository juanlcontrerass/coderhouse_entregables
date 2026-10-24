"""Consulta interactiva (o de una sola pregunta) al recuperador híbrido.

Uso:
    python query.py                                   # modo interactivo
    python query.py "¿Qué significa un error 422?"    # una consulta y sale
    python query.py --mode bm25 --category seguridad "token expirado"
"""

import argparse

from langchain_core.documents import Document

from rag_pinecone.rag_system import RAGSystem


def print_results(docs: list[Document]) -> None:
    if not docs:
        print("⚠️  No se recuperó ningún fragmento.\n")
        return
    print(f"\n📎 Top {len(docs)} fragmentos recuperados:")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        page = f" p.{m['page']}" if m.get("page") else ""
        print(f"\n  {i}. [{m.get('source')}{page} | {m.get('category')} | {m.get('chunk_id')}]")
        excerpt = " ".join(d.page_content.split())
        print(f"     {excerpt[:220]}{'...' if len(excerpt) > 220 else ''}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consulta el RAG híbrido (BM25 + Pinecone)")
    parser.add_argument("query", nargs="?", help="consulta; si se omite, modo interactivo")
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    parser.add_argument("--category", help="filtra por la metadata 'category'")
    args = parser.parse_args()

    rag = RAGSystem()

    if args.query:
        print_results(rag.search(args.query, mode=args.mode, category=args.category))
    else:
        print(f"💬 Modo interactivo ({args.mode}) — categorías: {', '.join(rag.categories())}")
        print("   (escribí 'salir' para terminar)\n")
        while (question := input("🧑 Vos: ").strip()).lower() not in ("salir", "exit", "quit", ""):
            print_results(rag.search(question, mode=args.mode, category=args.category))
        print("\n👋 Listo, terminamos la sesión.")
