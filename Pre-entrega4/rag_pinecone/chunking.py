"""Chunking medido en tokens y enriquecimiento de metadata para Pinecone."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pinecone.config import CHUNK_OVERLAP, CHUNK_SIZE


def make_chunk_id(doc_id: str, section: str, index: int) -> str:
    """ID determinista: re-ingestar el mismo chunk lo sobrescribe en vez de duplicarlo."""
    return f"{doc_id}::{section}::c{index}"


def split_documents(
    docs: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # tokenizer de text-embedding-3-small
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Índice del chunk dentro de cada (archivo, sección)
    counters: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["doc_id"], chunk.metadata["section"])
        index = counters.get(key, 0)
        counters[key] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = make_chunk_id(*key, index)
        chunk.id = chunk.metadata["chunk_id"]
    return chunks
