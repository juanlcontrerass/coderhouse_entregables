"""Análisis de sentimiento por léxico para textos en español.

Es determinista y explicable: cuenta palabras positivas y negativas, invierte la polaridad
tras una negación ("no hay reconocimiento") y pondera los intensificadores ("muy alto").
"""

import json
import re
import unicodedata
from typing import TypedDict

from langchain_core.tools import tool

POSITIVE = {
    "bueno", "buena", "buenos", "buenas", "bien", "excelente", "excelentes", "genial", "positivo",
    "positiva", "mejor", "mejoro", "mejora", "satisfecho", "satisfecha", "contento", "contenta",
    "feliz", "agradable", "apoyo", "flexible", "flexibilidad", "claro", "clara", "valoro", "valoramos",
    "encanta", "gusta", "recomiendo", "comodo", "util", "eficiente", "rapido", "rapida", "motivado",
    "motivada", "orgulloso", "orgullosa", "crecimiento", "oportunidades", "reconocimiento",
    "transparente", "justo", "justa", "bienvenido", "bienvenida", "beneficios",
}
NEGATIVE = {
    "malo", "mala", "malos", "malas", "mal", "pesimo", "pesima", "terrible", "horrible", "peor",
    "empeoro", "lento", "lenta", "lentos", "lentas", "problema", "problemas", "frustrante",
    "frustrado", "frustrada", "estres", "estresante", "agotado", "agotada", "cansado", "cansada",
    "injusto", "injusta", "confuso", "confusa", "confusos", "confusas", "desorganizado", "caos",
    "falta", "faltan", "escaso", "escasa", "insuficiente", "dificil", "complicado", "sobrecarga",
    "queja", "quejas", "preocupa", "preocupante", "desmotivado", "desmotivada", "demora", "demoras",
    "debajo",
}
NEGATORS = {"no", "nunca", "ni", "tampoco", "sin", "nadie"}
INTENSIFIERS = {"muy", "super", "bastante", "demasiado", "realmente", "mucho", "totalmente"}
NEGATION_WINDOW = 3  # palabras afectadas por una negación
THRESHOLD = 0.2


class TextSentiment(TypedDict):
    texto: str
    puntaje: float
    etiqueta: str
    positivas: list[str]
    negativas: list[str]


def _normalize(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z]+", plain)


def score_text(text: str) -> TextSentiment:
    """Puntaje en [-1, 1] = (positivo - negativo) / (positivo + negativo), ponderado."""
    pos_weight = neg_weight = 0.0
    positives: list[str] = []
    negatives: list[str] = []
    negation_left = 0
    boost = 1.0
    for token in _normalize(text):
        if token in NEGATORS:
            negation_left = NEGATION_WINDOW
            continue
        if token in INTENSIFIERS:
            boost = 1.5
            continue
        polarity = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
        if polarity and negation_left:
            polarity = -polarity
        if polarity > 0:
            pos_weight += boost
            positives.append(token)
        elif polarity < 0:
            neg_weight += boost
            negatives.append(token)
        boost = 1.0
        negation_left = max(0, negation_left - 1)
    total = pos_weight + neg_weight
    score = round((pos_weight - neg_weight) / total, 3) if total else 0.0
    label = "positivo" if score > THRESHOLD else "negativo" if score < -THRESHOLD else "neutral"
    return {"texto": text, "puntaje": score, "etiqueta": label, "positivas": positives, "negativas": negatives}


def analyze(texts: list[str]) -> dict[str, object]:
    results = [score_text(t) for t in texts]
    distribution = {label: sum(r["etiqueta"] == label for r in results) for label in ("positivo", "neutral", "negativo")}
    n = len(results)
    return {
        "n": n,
        "puntaje_promedio": round(sum(r["puntaje"] for r in results) / n, 3) if n else 0.0,
        "distribucion": distribution,
        "porcentaje_positivo": round(distribution["positivo"] / n * 100, 1) if n else 0.0,
        "detalle": results,
    }


@tool
def analizar_sentimiento(textos: list[str]) -> str:
    """Analiza el sentimiento de una lista de textos en español. Devuelve, por texto, un
    puntaje entre -1 y 1 con su etiqueta (positivo/neutral/negativo) y las palabras que lo
    explican, más la distribución y el puntaje promedio del conjunto."""
    if not textos:
        return "Error: la lista de textos está vacía."
    return json.dumps(analyze(textos), ensure_ascii=False)
