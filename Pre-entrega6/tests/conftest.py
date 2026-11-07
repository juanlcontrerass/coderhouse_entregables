"""Vector store en memoria con embeddings deterministas: la búsqueda real sin descargas."""

from collections.abc import Iterator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from tools.retrieval import load_documents, set_vectorstore, split_documents


@pytest.fixture(autouse=True)
def memory_store() -> Iterator[InMemoryVectorStore]:
    store = InMemoryVectorStore.from_documents(split_documents(load_documents()), DeterministicFakeEmbedding(size=64))
    set_vectorstore(store)
    yield store
    set_vectorstore(None)
