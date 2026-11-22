"""Herramientas: lógica, contratos Pydantic de entrada y artifacts de salida."""

import json

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.tools import ANALYST_TOOLS, RESEARCH_TOOLS
from app.tools.calculator import calcular, estadisticas, safe_eval, variacion_porcentual
from app.tools.publisher import publicar_informe
from app.tools.retrieval import buscar_documentos, extract_sources
from app.tools.schemas import BuscarDocumentosInput, ResultadoBusqueda, SentimientoInput, VariacionInput
from app.tools.sentiment import analizar_sentimiento, score_text


def _call(name: str, **args: object) -> dict[str, object]:
    return {"type": "tool_call", "id": "c1", "name": name, "args": args}


def test_tools_are_scoped_by_role() -> None:
    assert {t.name for t in RESEARCH_TOOLS} == {"buscar_documentos", "listar_documentos"}
    assert {t.name for t in ANALYST_TOOLS} == {"calcular", "estadisticas", "variacion_porcentual", "analizar_sentimiento"}


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(28 - 14) / 14 * 100", 100), ("(14 + 21 + 28) / 3", 21), ("-2 ** 2", -4), ("max(3, 9) + sqrt(16)", 13)],
)
def test_safe_eval(expression: str, expected: float) -> None:
    assert safe_eval(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('dir')", "open('x')", "2 ** 1000", "a + 1", "[1, 2]", "(1).real"])
def test_safe_eval_rejects_code(expression: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calcular_returns_text_and_validated_artifact() -> None:
    assert calcular.invoke({"expresion": "(14 + 21 + 28) / 3"}) == "(14 + 21 + 28) / 3 = 21"
    message = calcular.invoke(_call("calcular", expresion="(14 + 21 + 28) / 3"))
    assert isinstance(message, ToolMessage) and message.artifact == {"expresion": "(14 + 21 + 28) / 3", "resultado": 21.0}
    assert calcular.invoke({"expresion": "1 / 0"}).startswith("Error")
    assert calcular.invoke({"expresion": "__import__('os')"}).startswith("Error")


def test_estadisticas() -> None:
    result = json.loads(estadisticas.invoke({"valores": [14, 21, 28]}))
    assert result["promedio"] == 21 and result["mediana"] == 21 and result["desvio_estandar"] == 7


def test_variacion_porcentual() -> None:
    assert variacion_porcentual.invoke({"valor_inicial": 12, "valor_final": 21}) == "Variación de 12 a 21: +75.00%"


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (VariacionInput, {"valor_inicial": 0, "valor_final": 5}),
        (BuscarDocumentosInput, {"consulta": "  a ", "k": 3}),
        (BuscarDocumentosInput, {"consulta": "vacaciones", "k": 50}),
        (SentimientoInput, {"textos": ["   "]}),
    ],
)
def test_input_schemas_reject_invalid_arguments(schema: type, payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_invalid_arguments_go_back_to_the_agent_as_a_message() -> None:
    """El LLM recibe el error de validación y puede corregir la llamada: el grafo no se rompe."""
    message = variacion_porcentual.invoke(_call("variacion_porcentual", valor_inicial=0, valor_final=5))
    assert isinstance(message, ToolMessage)
    assert "argumentos inválidos" in str(message.content) and "valor_inicial" in str(message.content)
    empty = estadisticas.invoke(_call("estadisticas", valores=[]))
    assert "argumentos inválidos" in str(empty.content)


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


async def test_buscar_documentos_tags_every_fragment_and_returns_artifact() -> None:
    message = await buscar_documentos.ainvoke(_call("buscar_documentos", consulta="días de vacaciones por antigüedad", k=3))
    assert str(message.content).count("[Fuente:") == 3
    result = ResultadoBusqueda.model_validate(message.artifact)
    assert len(result.fragmentos) == 3 and result.fuentes == extract_sources(str(message.content))
    assert all(f.chunk_id.startswith(f.fuente) for f in result.fragmentos)


async def test_publicar_informe_writes_the_report(reports_dir) -> None:  # type: ignore[no-untyped-def]
    message = await publicar_informe.ainvoke(
        _call("publicar_informe", job_id="../fuera", consulta="c", contenido="respuesta", aprobado_por="RR. HH.")
    )
    path = reports_dir / "informe____fuera.md"  # el id se sanea: no puede salir del directorio
    assert message.artifact["ruta"] == str(path) and "RR. HH." in path.read_text(encoding="utf-8")


def test_extract_sources_supports_lists() -> None:
    assert extract_sources("Dato [Fuente: a.md] y otro [Fuente: b.md, a.md]") == ["a.md", "b.md"]
