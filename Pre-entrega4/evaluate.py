"""Evaluación del recuperador con un Golden Set: Precision@k, Recall@k (y MRR como extra).

Corre cada pregunta en los tres modos (híbrido, solo vectorial, solo BM25) para mostrar
qué aporta la combinación.

Uso:
    python evaluate.py                 # detalle del modo híbrido + tabla comparativa
    python evaluate.py --verbose       # detalle de los tres modos
"""

import argparse
import json
from pathlib import Path

from rag_pinecone.config import GOLDEN_SET_PATH, TOP_K
from rag_pinecone.metrics import precision_at_k, recall_at_k, reciprocal_rank
from rag_pinecone.rag_system import RAGSystem

MODES = ("hybrid", "vector", "bm25")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Normaliza 'documento_id_esperado' a set (acepta un string o una lista)."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        expected = case["documento_id_esperado"]
        case["esperados"] = {expected} if isinstance(expected, str) else set(expected)
    return cases


def evaluate(rag: RAGSystem, cases: list[dict], mode: str, k: int = TOP_K) -> dict:
    detail = []
    for case in cases:
        retrieved = [d.metadata["doc_id"] for d in rag.search(case["pregunta"], mode=mode)]
        detail.append(
            {
                "pregunta": case["pregunta"],
                "esperados": sorted(case["esperados"]),
                "recuperados": retrieved,
                "precision": precision_at_k(retrieved, case["esperados"], k),
                "recall": recall_at_k(retrieved, case["esperados"], k),
                "rr": reciprocal_rank(retrieved, case["esperados"], k),
            }
        )
    n = len(detail)
    return {
        "detalle": detail,
        "precision": sum(r["precision"] for r in detail) / n,
        "recall": sum(r["recall"] for r in detail) / n,
        "mrr": sum(r["rr"] for r in detail) / n,
    }


def print_detail(mode: str, report: dict, k: int) -> None:
    print("=" * 80)
    print(f"🔎 Modo: {mode}")
    print("=" * 80)
    for r in report["detalle"]:
        status = "✅" if r["recall"] == 1.0 else ("🟡" if r["recall"] > 0 else "❌")
        print(f"{status} {r['pregunta']}")
        print(f"   Esperado(s): {r['esperados']}")
        print(f"   Recuperados: {r['recuperados']}")
        print(f"   Recall@{k}: {r['recall']:.0%} | Precision@{k}: {r['precision']:.0%}\n")


def print_summary(reports: dict[str, dict], k: int, n: int) -> None:
    print("=" * 80)
    print(f"📊 RESUMEN ({n} preguntas, top-{k})")
    print("=" * 80)
    print(f"{'Modo':<10}{'Recall@' + str(k):>12}{'Precision@' + str(k):>15}{'MRR':>10}")
    for mode, rep in reports.items():
        print(f"{mode:<10}{rep['recall']:>12.1%}{rep['precision']:>15.1%}{rep['mrr']:>10.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa Precision@k y Recall@k con el Golden Set")
    parser.add_argument("--verbose", action="store_true", help="muestra el detalle de los tres modos")
    args = parser.parse_args()

    cases = load_golden_set()
    rag = RAGSystem(k=TOP_K)
    print(f"🧪 Golden Set: {len(cases)} preguntas | corpus: {len(rag.corpus)} chunks en Pinecone\n")

    reports = {mode: evaluate(rag, cases, mode) for mode in MODES}
    for mode in MODES if args.verbose else ("hybrid",):
        print_detail(mode, reports[mode], TOP_K)
    print_summary(reports, TOP_K, len(cases))
