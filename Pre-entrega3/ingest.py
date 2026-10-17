"""Módulo de ingesta: lee /data, fragmenta en chunks y los persiste en ChromaDB.

Uso:
    python ingest.py            # indexa solo si la colección no existe o está vacía
    python ingest.py --rebuild  # borra la colección y vuelve a indexar todo
"""

import argparse

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    get_embeddings,
)


def load_documents() -> list[Document]:
    """Carga todos los .txt y .md de /data como Documents de LangChain."""
    paths = sorted(p for p in DATA_DIR.iterdir() if p.suffix in (".txt", ".md"))
    if not paths:
        raise FileNotFoundError(f"No hay archivos .txt o .md en {DATA_DIR}")
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in paths
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunking medido en tokens (tiktoken), que es la unidad del límite de contexto del LLM."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    # Índice del chunk dentro de su archivo, útil para depurar qué fragmento se recuperó
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk"] = counters.get(source, 0)
        counters[source] = chunk.metadata["chunk"] + 1
    return chunks


def collection_count() -> int:
    """Cantidad de fragmentos ya persistidos (0 si la colección no existe)."""
    if not PERSIST_DIR.exists():
        return 0
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        return 0
    return client.get_collection(COLLECTION_NAME).count()


def ingest(rebuild: bool = False) -> Chroma:
    embeddings = get_embeddings()

    if rebuild and PERSIST_DIR.exists():
        client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Colección '{COLLECTION_NAME}' eliminada")

    existing = collection_count()
    if existing > 0:
        print(f"♻️  Índice existente con {existing} fragmentos — se omite la reindexación")
        print("   (usá --rebuild si cambiaste los documentos de /data)")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    docs = load_documents()
    chunks = split_documents(docs)
    print(f"📄 Documentos cargados: {len(docs)}")
    print(f"✂️  Fragmentos generados: {len(chunks)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} tokens)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        # Guardamos el modelo usado para que la cadena RAG pueda verificar que consulta con el mismo
        collection_metadata={"embedding_model": EMBEDDING_MODEL},
    )
    print(f"📦 {vectorstore._collection.count()} fragmentos persistidos en {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa /data en ChromaDB")
    parser.add_argument("--rebuild", action="store_true", help="borra la colección y reindexa")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)
