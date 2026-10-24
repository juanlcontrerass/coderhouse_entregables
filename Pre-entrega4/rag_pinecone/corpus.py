"""Reconstruye el corpus de chunks leyendo la metadata guardada en Pinecone.

BM25 necesita el texto de todos los chunks en memoria. En vez de volver a leer y
fragmentar data/, lo traemos del propio índice: Pinecone es la única fuente de verdad
y BM25 y el retriever vectorial trabajan exactamente sobre los mismos chunks e IDs.
"""

from langchain_core.documents import Document

from rag_pinecone.config import NAMESPACE, TEXT_KEY

FETCH_BATCH = 100
INT_FIELDS = ("page", "chunk_index")


def metadata_to_document(vector_id: str, metadata: dict) -> Document:
    meta = dict(metadata)
    text = meta.pop(TEXT_KEY, "")
    # Pinecone devuelve los números como float: los normalizamos a int
    for field in INT_FIELDS:
        if field in meta:
            meta[field] = int(meta[field])
    meta.setdefault("chunk_id", vector_id)
    return Document(id=vector_id, page_content=text, metadata=meta)


def load_corpus_from_pinecone(index, namespace: str = NAMESPACE) -> list[Document]:
    ids = [vector_id for page in index.list(namespace=namespace) for vector_id in page]
    docs = []
    for start in range(0, len(ids), FETCH_BATCH):
        response = index.fetch(ids=ids[start : start + FETCH_BATCH], namespace=namespace)
        docs.extend(metadata_to_document(vid, vec.metadata or {}) for vid, vec in response.vectors.items())
    return sorted(docs, key=lambda d: d.id)
