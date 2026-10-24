"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) combinados con EnsembleRetriever.

La búsqueda vectorial resuelve bien los parafraseos ("¿cómo corro tareas después de
responder?" ≈ BackgroundTasks), pero es débil con términos exactos como "HTTP 422" u
"OAuth2PasswordBearer". BM25 premia la coincidencia literal. El EnsembleRetriever fusiona
ambos rankings con Reciprocal Rank Fusion ponderada.
"""

import re
import unicodedata
from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore

from rag_pinecone.config import (
    ENSEMBLE_WEIGHTS,
    INDEX_NAME,
    NAMESPACE,
    TEXT_KEY,
    TOP_K,
    get_embeddings,
)
from rag_pinecone.corpus import load_corpus_from_pinecone
from rag_pinecone.index import ensure_index

Mode = Literal["hybrid", "vector", "bm25"]

_TOKEN_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes y sin puntuación: 'Autenticación,' y 'autenticacion' coinciden."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return _TOKEN_RE.findall(without_accents)


class RAGSystem:
    """Encapsula el EnsembleRetriever y devuelve el top-k para una consulta."""

    def __init__(
        self,
        k: int = TOP_K,
        weights: tuple[float, float] = ENSEMBLE_WEIGHTS,
        namespace: str = NAMESPACE,
        index_name: str = INDEX_NAME,
    ):
        self.k = k
        self.weights = list(weights)
        self.namespace = namespace
        self.index = ensure_index(index_name=index_name)
        self.vectorstore = PineconeVectorStore(
            index=self.index, embedding=get_embeddings(), text_key=TEXT_KEY, namespace=namespace
        )
        self.corpus = load_corpus_from_pinecone(self.index, namespace)
        if not self.corpus:
            raise RuntimeError(f"El namespace '{namespace}' está vacío: ejecutá primero python ingest.py")
        self._retrievers: dict[tuple[Mode, str | None], BaseRetriever] = {}

    # ---- construcción de retrievers -------------------------------------------------

    def _bm25(self, category: str | None) -> BM25Retriever:
        docs = [d for d in self.corpus if category is None or d.metadata.get("category") == category]
        retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenize)
        retriever.k = self.k
        return retriever

    def _vector(self, category: str | None) -> BaseRetriever:
        search_kwargs: dict = {"k": self.k, "namespace": self.namespace}
        if category:
            search_kwargs["filter"] = {"category": {"$eq": category}}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_retriever(self, mode: Mode = "hybrid", category: str | None = None) -> BaseRetriever:
        key = (mode, category)
        if key not in self._retrievers:
            match mode:
                case "bm25":
                    self._retrievers[key] = self._bm25(category)
                case "vector":
                    self._retrievers[key] = self._vector(category)
                case "hybrid":
                    self._retrievers[key] = EnsembleRetriever(
                        retrievers=[self._bm25(category), self._vector(category)],
                        weights=self.weights,
                        id_key="chunk_id",  # deduplica por ID de chunk, no por texto
                    )
                case _:
                    raise ValueError(f"Modo desconocido: {mode}")
        return self._retrievers[key]

    # ---- API pública ----------------------------------------------------------------

    def search(self, query: str, mode: Mode = "hybrid", category: str | None = None) -> list[Document]:
        """Top-k Documents para la consulta (por defecto, fusión léxica + semántica)."""
        return self.get_retriever(mode, category).invoke(query)[: self.k]

    def categories(self) -> list[str]:
        return sorted({d.metadata.get("category", "general") for d in self.corpus})
