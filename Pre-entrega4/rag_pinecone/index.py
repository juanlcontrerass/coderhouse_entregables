"""Setup del índice Pinecone Serverless: lo crea si no existe y valida su dimensión."""

import time

from pinecone import Pinecone, ServerlessSpec

from rag_pinecone.config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    require_env,
)


def get_client() -> Pinecone:
    return Pinecone(api_key=require_env("PINECONE_API_KEY"))


def ensure_index(pc: Pinecone | None = None, index_name: str = INDEX_NAME, timeout: int = 120):
    """Devuelve el Index listo para usar, creándolo en modo Serverless si hace falta.

    Si el índice ya existe con otra dimensión falla de inmediato: subir vectores de 1536
    a un índice de 768 (u otra) es el "mismatch de dimensiones" que hay que evitar.
    """
    pc = pc or get_client()

    if pc.has_index(index_name):
        description = pc.describe_index(index_name)
        if description.dimension != EMBEDDING_DIM:
            raise ValueError(
                f"El índice '{index_name}' tiene dimensión {description.dimension}, pero los "
                f"embeddings generan {EMBEDDING_DIM}. Usá otro INDEX_NAME o borrá el índice."
            )
        print(f"♻️  El índice '{index_name}' ya existe (dimensión {description.dimension}) — no se recrea")
    else:
        print(f"🆕 Creando índice Serverless '{index_name}' (dim={EMBEDDING_DIM}, {PINECONE_CLOUD}/{PINECONE_REGION})...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    deadline = time.monotonic() + timeout
    while not pc.describe_index(index_name).status.ready:
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice '{index_name}' no quedó listo en {timeout}s")
        time.sleep(2)

    return pc.Index(index_name)


def namespace_count(index, namespace: str) -> int:
    """Cantidad de vectores en un namespace (0 si todavía no existe)."""
    stats = index.describe_index_stats()
    ns = stats.namespaces.get(namespace)
    return ns.vector_count if ns else 0
