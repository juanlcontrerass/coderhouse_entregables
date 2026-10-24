"""Tests offline (sin Pinecone ni OpenAI) de loaders, chunking y metadata."""

import json

from rag_pinecone.chunking import split_documents
from rag_pinecone.config import DATA_DIR, GOLDEN_SET_PATH
from rag_pinecone.corpus import metadata_to_document
from rag_pinecone.loaders import load_documents, parse_front_matter
from rag_pinecone.rag_system import bm25_tokenize

PINECONE_TYPES = (str, int, float, bool)


def _chunks():
    return split_documents(load_documents(DATA_DIR))


def test_loads_all_three_formats():
    file_types = {d.metadata["file_type"] for d in load_documents(DATA_DIR)}
    assert file_types == {"md", "pdf", "json"}


def test_pdf_keeps_page_numbers():
    pages = [d.metadata["page"] for d in load_documents(DATA_DIR) if d.metadata["file_type"] == "pdf"]
    assert pages == [1, 2, 3]


def test_front_matter_is_parsed_and_removed():
    meta, body = parse_front_matter("---\ntitle: X\ntags: a, b\n---\n# Cuerpo")
    assert meta == {"title": "X", "tags": "a, b"}
    assert body == "# Cuerpo"


def test_chunk_ids_are_unique_and_deterministic():
    ids = [c.metadata["chunk_id"] for c in _chunks()]
    assert len(ids) == len(set(ids))
    assert ids == [c.metadata["chunk_id"] for c in _chunks()]


def test_metadata_only_uses_pinecone_compatible_types():
    for chunk in _chunks():
        for key, value in chunk.metadata.items():
            ok = isinstance(value, PINECONE_TYPES) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value)
            )
            assert ok, f"{key}={value!r} no es un tipo válido para Pinecone"
        assert {"source", "doc_id", "page", "category", "tags", "chunk_id"} <= chunk.metadata.keys()


def test_markdown_docs_produce_several_chunks():
    """Docs largos → varios chunks por documento, para que Precision@5 sea informativa."""
    per_doc: dict[str, int] = {}
    for c in _chunks():
        per_doc[c.metadata["doc_id"]] = per_doc.get(c.metadata["doc_id"], 0) + 1
    assert all(n >= 2 for doc, n in per_doc.items() if doc.endswith(".md"))


def test_golden_set_points_to_existing_documents():
    doc_ids = {d.metadata["doc_id"] for d in load_documents(DATA_DIR)}
    cases = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    assert len(cases) >= 5
    for case in cases:
        expected = case["documento_id_esperado"]
        assert set([expected] if isinstance(expected, str) else expected) <= doc_ids


def test_metadata_roundtrip_from_pinecone():
    doc = metadata_to_document("x::full::c0", {"text": "hola", "page": 2.0, "chunk_index": 0.0})
    assert doc.page_content == "hola"
    assert doc.metadata == {"page": 2, "chunk_index": 0, "chunk_id": "x::full::c0"}


def test_bm25_tokenizer_normalizes_case_accents_and_punctuation():
    assert bm25_tokenize("Autenticación, HTTP 422!") == ["autenticacion", "http", "422"]
