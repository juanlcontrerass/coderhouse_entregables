import pytest

from rag_pinecone.metrics import precision_at_k, recall_at_k, reciprocal_rank

RETRIEVED = ["a.md", "b.md", "a.md", "c.md", "d.md"]


def test_precision_counts_relevant_chunks_over_k():
    assert precision_at_k(RETRIEVED, {"a.md"}, k=5) == pytest.approx(0.4)


def test_precision_penalizes_fewer_than_k_results():
    assert precision_at_k(["a.md"], {"a.md"}, k=5) == pytest.approx(0.2)


def test_recall_single_relevant_is_binary():
    assert recall_at_k(RETRIEVED, {"c.md"}) == 1.0
    assert recall_at_k(RETRIEVED, {"z.md"}) == 0.0


def test_recall_multiple_relevant_is_partial():
    assert recall_at_k(RETRIEVED, {"a.md", "z.md"}) == pytest.approx(0.5)


def test_recall_only_looks_at_top_k():
    assert recall_at_k(RETRIEVED, {"d.md"}, k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, {"b.md"}) == pytest.approx(0.5)
    assert reciprocal_rank(RETRIEVED, {"z.md"}) == 0.0
