import json

import pytest

from tools import ANALYST_TOOLS, RESEARCH_TOOLS
from tools.calculator import calcular, estadisticas, safe_eval, variacion_porcentual
from tools.retrieval import buscar_documentos, extract_sources, load_documents, split_documents
from tools.sentiment import analizar_sentimiento, score_text


def test_tools_are_scoped_by_role() -> None:
    assert {t.name for t in RESEARCH_TOOLS} == {"buscar_documentos", "listar_documentos"}
    assert {t.name for t in ANALYST_TOOLS} == {
        "calcular",
        "estadisticas",
        "variacion_porcentual",
        "analizar_sentimiento",
    }


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(28 - 14) / 14 * 100", 100), ("(14 + 21 + 28) / 3", 21), ("-2 ** 2", -4), ("max(3, 9) + sqrt(16)", 13)],
)
def test_safe_eval(expression: str, expected: float) -> None:
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('dir')", "open('x')", "2 ** 1000", "a + 1", "[1, 2]", "(1).real"]
)
def test_safe_eval_rejects_code(expression: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calcular_reports_errors_instead_of_raising() -> None:
    assert calcular.invoke({"expresion": "(14 + 21 + 28) / 3"}) == "(14 + 21 + 28) / 3 = 21"
    assert calcular.invoke({"expresion": "1 / 0"}).startswith("Error")
    assert calcular.invoke({"expresion": "__import__('os')"}).startswith("Error")


def test_estadisticas() -> None:
    result = json.loads(estadisticas.invoke({"valores": [14, 21, 28]}))
    assert result["promedio"] == 21 and result["mediana"] == 21 and result["desvio_estandar"] == 7


def test_variacion_porcentual() -> None:
    assert variacion_porcentual.invoke({"valor_inicial": 12, "valor_final": 21}) == "Variación de 12 a 21: +75.00%"
    assert variacion_porcentual.invoke({"valor_inicial": 0, "valor_final": 5}).startswith("Error")


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("La flexibilidad del teletrabajo es excelente", "positivo"),
        ("En Soporte la sobrecarga de tickets es terrible", "negativo"),
        ("No hay reconocimiento por el trabajo extra", "negativo"),  # la negación invierte
        ("No hay problemas con el equipo", "positivo"),
        ("La reunión es los martes", "neutral"),
    ],
)
def test_score_text(text: str, label: str) -> None:
    assert score_text(text)["etiqueta"] == label


def test_analizar_sentimiento_aggregates() -> None:
    result = json.loads(analizar_sentimiento.invoke({"textos": ["Excelente equipo", "Procesos lentos y confusos"]}))
    assert result["distribucion"] == {"positivo": 1, "neutral": 0, "negativo": 1}
    assert result["porcentaje_positivo"] == 50.0


def test_chunks_keep_source_and_size() -> None:
    chunks = split_documents(load_documents())
    assert {c.metadata["source"] for c in chunks} >= {"politica_vacaciones.md", "encuesta_clima_2025.md"}
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_buscar_documentos_tags_every_fragment_with_its_source() -> None:
    result = buscar_documentos.invoke({"consulta": "vacaciones", "k": 3})
    assert result.count("[Fuente:") == 3
    assert extract_sources(result)


def test_extract_sources_supports_lists() -> None:
    assert extract_sources("Dato [Fuente: a.md] y otro [Fuente: b.md, a.md]") == ["a.md", "b.md"]
