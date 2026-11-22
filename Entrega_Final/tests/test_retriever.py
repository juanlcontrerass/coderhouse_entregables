"""RAG híbrido: chunking, BM25, fusión RRF y búsqueda asíncrona."""

from langchain_core.documents import Document

from app.rag.corpus import load_documents, split_documents
from app.rag.retriever import HybridRetriever, bm25_tokenize, rrf_fuse


def _doc(chunk_id: str) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})


def test_chunks_keep_source_size_and_stable_ids() -> None:
    chunks = split_documents(load_documents())
    assert {c.metadata["source"] for c in chunks} >= {"politica_vacaciones.md", "encuesta_clima_2025.md"}
    assert all(len(c.page_content) <= 1000 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)) and ids == [c.metadata["chunk_id"] for c in split_documents(load_documents())]


def test_bm25_tokenizer_ignores_case_and_accents() -> None:
    assert bm25_tokenize("Política de VACACIONES: días") == ["politica", "de", "vacaciones", "dias"]


def test_rrf_rewards_documents_found_by_both_retrievers() -> None:
    lexical = [_doc("a"), _doc("b"), _doc("c")]
    dense = [_doc("c"), _doc("d"), _doc("a")]
    fused = rrf_fuse([lexical, dense], [0.5, 0.5], k=60)
    assert [h.chunk_id for h in fused][:2] == ["a", "c"]  # aparecen en ambas listas
    assert len(fused) == 4  # deduplicado por chunk_id


def test_rrf_weights_shift_the_ranking() -> None:
    lexical, dense = [_doc("a")], [_doc("b")]
    assert rrf_fuse([lexical, dense], [1.0, 0.0], k=60)[0].chunk_id == "a"
    assert rrf_fuse([lexical, dense], [0.0, 1.0], k=60)[0].chunk_id == "b"


async def test_exact_keyword_is_found_by_the_lexical_side(memory_retriever: HybridRetriever) -> None:
    """Los embeddings de prueba no tienen semántica: si la encuesta aparece primero, es por BM25."""
    hits = await memory_retriever.asearch("eNPS encuesta de clima", k=3)
    assert len(hits) == 3
    assert hits[0].document.metadata["source"] == "encuesta_clima_2025.md"
    assert hits[0].score >= hits[-1].score


def test_lexical_search_skips_chunks_without_matches(memory_retriever: HybridRetriever) -> None:
    assert memory_retriever.lexical_search("zzzz qqqq", 5) == []
