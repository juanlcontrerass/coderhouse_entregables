"""Entorno de tests: sin red, sin API key, sin Redis y sin Phoenix.

- Variables de entorno mínimas (se fijan antes de importar `app`).
- RAG híbrido real (BM25 + vector store en memoria) con embeddings deterministas: no descarga
  el modelo de HuggingFace.
"""

import os

os.environ.update(
    {
        "OPENAI_MODEL": "modelo-de-prueba",
        "CHROMA_COLLECTION": "tests",
        "EMBEDDING_MODEL": "embeddings-de-prueba",
        "PHOENIX_ENABLED": "false",  # los tests nunca envían trazas
    }
)
os.environ.pop("OPENAI_API_KEY", None)

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from app.core.config import get_settings
from app.rag.corpus import load_documents, split_documents
from app.rag.retriever import HybridRetriever, set_retriever


def make_retriever(docs: list[Document] | None = None) -> HybridRetriever:
    chunks = split_documents(docs if docs is not None else load_documents())
    store = InMemoryVectorStore.from_documents(chunks, DeterministicFakeEmbedding(size=64))
    return HybridRetriever(store, chunks)


@pytest.fixture(autouse=True)
def memory_retriever() -> Iterator[HybridRetriever]:
    retriever = make_retriever()
    set_retriever(retriever)
    yield retriever
    set_retriever(None)


@pytest.fixture(autouse=True)
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_settings(), "reports_dir", tmp_path / "reports")
    return tmp_path / "reports"
