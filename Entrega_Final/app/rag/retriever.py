"""Búsqueda híbrida (BM25 + densa) con Reciprocal Rank Fusion.

Es la estrategia de la pre-entrega del Módulo 4 (`EnsembleRetriever` con pesos 0.5/0.5 y
deduplicación por `chunk_id`), reimplementada en pocas líneas para controlar la asincronía:

- **Densa:** Chroma con embeddings multilingües locales (`asimilarity_search`, que ejecuta la
  búsqueda en un hilo y no bloquea el event loop).
- **Léxica:** BM25 (`rank_bm25`) sobre los mismos fragmentos, con el tokenizador del Módulo 4
  (minúsculas y sin tildes). Captura coincidencias exactas que los embeddings diluyen:
  "eNPS", "2FA", números de artículo, nombres propios.
- **Fusión:** `score(doc) = Σ peso_i / (rrf_k + rank_i)`. No depende de la escala de cada
  puntaje, solo del orden, por eso combina bien BM25 y similitud coseno.

El índice de Chroma se construye una sola vez (en un hilo, al arrancar el worker) y se
reconstruye solo si el corpus cambió.
"""

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.rag.corpus import load_documents, split_documents

logger = logging.getLogger(__name__)


def bm25_tokenize(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"\w+", plain)


@dataclass(frozen=True)
class ScoredChunk:
    document: Document
    score: float

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", ""))


def rrf_fuse(rankings: list[list[Document]], weights: list[float], k: int) -> list[ScoredChunk]:
    """Fusiona listas ordenadas (la mejor primero) y deduplica por `chunk_id`."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc in enumerate(ranking, start=1):
            key = str(doc.metadata.get("chunk_id") or doc.page_content)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            docs.setdefault(key, doc)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [ScoredChunk(docs[key], round(scores[key], 6)) for key in ordered]


class HybridRetriever:
    def __init__(self, store: VectorStore, chunks: list[Document]) -> None:
        self.store = store
        self.chunks = chunks
        self._bm25 = BM25Okapi([bm25_tokenize(c.page_content) for c in chunks])

    def lexical_search(self, query: str, n: int) -> list[Document]:
        scores = self._bm25.get_scores(bm25_tokenize(query))
        best = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i] for i in best if scores[i] > 0]

    async def asearch(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        settings = get_settings()
        k = k or settings.top_k
        n = max(k, settings.retrieval_candidates)
        dense = await self.store.asimilarity_search(query, k=n)
        lexical = self.lexical_search(query, n)  # CPU puro sobre unas decenas de fragmentos: microsegundos
        fused = rrf_fuse(
            [lexical, dense], [settings.hybrid_bm25_weight, settings.hybrid_dense_weight], settings.rrf_k
        )
        return fused[:k]


def _open_chroma(chunks: list[Document], rebuild: bool) -> VectorStore:
    """Abre la colección persistente y la (re)indexa si está vacía o desactualizada. Bloqueante."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=HuggingFaceEmbeddings(model_name=settings.embedding_model),
        persist_directory=str(settings.vectorstore_dir),
    )
    expected = {c.metadata["chunk_id"] for c in chunks}
    indexed = set(store.get(include=[])["ids"])
    if rebuild or indexed != expected:
        if indexed:
            store.reset_collection()
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        logger.info("Índice %s construido: %s fragmentos.", settings.chroma_collection, len(chunks))
    return store


async def build_retriever(rebuild: bool = False) -> HybridRetriever:
    chunks = split_documents(load_documents())
    store = await asyncio.to_thread(_open_chroma, chunks, rebuild)
    return HybridRetriever(store, chunks)


_retriever: HybridRetriever | None = None
_lock = asyncio.Lock()


async def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        async with _lock:
            if _retriever is None:
                _retriever = await build_retriever()
    return _retriever


def set_retriever(retriever: HybridRetriever | None) -> None:
    """Los tests inyectan uno con embeddings deterministas."""
    global _retriever
    _retriever = retriever
