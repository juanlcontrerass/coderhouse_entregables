"""Pipeline de ingesta: data/ → chunks con metadata → embeddings → Pinecone (namespace).

El texto original de cada chunk viaja dentro de la metadata del vector (clave TEXT_KEY),
así una sola consulta a Pinecone devuelve vector + texto + fuente, sin base relacional aparte.
"""

import time

from langchain_pinecone import PineconeVectorStore

from rag_pinecone.chunking import split_documents
from rag_pinecone.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_NAME,
    NAMESPACE,
    TEXT_KEY,
    get_embeddings,
)
from rag_pinecone.index import ensure_index, namespace_count
from rag_pinecone.loaders import load_documents


def _wait_for_count(index, namespace: str, expected: int, timeout: int = 60) -> int:
    """describe_index_stats es eventualmente consistente: esperamos a que refleje el upsert."""
    deadline = time.monotonic() + timeout
    count = namespace_count(index, namespace)
    while count != expected and time.monotonic() < deadline:
        time.sleep(2)
        count = namespace_count(index, namespace)
    return count


def run_ingest(rebuild: bool = False, namespace: str = NAMESPACE) -> int:
    """Indexa data/ en Pinecone. Devuelve la cantidad de vectores en el namespace."""
    index = ensure_index(index_name=INDEX_NAME)

    existing = namespace_count(index, namespace)
    if existing and rebuild:
        index.delete(delete_all=True, namespace=namespace)
        print(f"🗑️  Namespace '{namespace}' vaciado ({existing} vectores)")
        _wait_for_count(index, namespace, 0)
    elif existing:
        print(f"♻️  El namespace '{namespace}' ya tiene {existing} vectores — se omite la ingesta")
        print("   (usá --rebuild si cambiaste los documentos de data/)")
        return existing

    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    sources = sorted({d.metadata["source"] for d in docs})
    print(f"📄 Documentos cargados: {len(docs)} unidades de {len(sources)} archivos ({', '.join(sources)})")
    print(f"✂️  Chunks generados: {len(chunks)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} tokens)")
    print(f"   Metadata de ejemplo: {chunks[0].metadata}")

    vectorstore = PineconeVectorStore(
        index=index, embedding=get_embeddings(), text_key=TEXT_KEY, namespace=namespace
    )
    vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks], namespace=namespace)

    count = _wait_for_count(index, namespace, len(chunks))
    print(f"📦 {count} vectores en '{INDEX_NAME}' / namespace '{namespace}'")
    return count
