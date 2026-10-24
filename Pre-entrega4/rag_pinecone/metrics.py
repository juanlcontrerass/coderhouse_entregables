"""Métricas de recuperación a nivel documento fuente.

Un chunk recuperado es relevante si su doc_id pertenece al conjunto de documentos esperados.
"""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Fracción de los k recuperados que pertenecen a un documento relevante.

    Se divide por k (no por la cantidad devuelta): si el retriever trae menos de k, se penaliza.
    """
    top = retrieved[:k]
    return sum(doc_id in relevant for doc_id in top) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Fracción de los documentos relevantes que aparecen al menos una vez en el top-k."""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """1 / posición del primer resultado relevante (0 si no hay ninguno en el top-k)."""
    for position, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1 / position
    return 0.0
