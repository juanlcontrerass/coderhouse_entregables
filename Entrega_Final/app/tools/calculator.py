"""Herramientas de cómputo del Agente de Análisis.

`calcular` usa un evaluador sobre el AST de Python que solo admite números, operadores
aritméticos y unas pocas funciones: nunca `eval`, así que no puede ejecutar código.
"""

import ast
import math
import operator
import statistics
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from app.tools.retrieval import validation_message
from app.tools.schemas import (
    CalcularInput,
    ErrorHerramienta,
    Estadisticas,
    EstadisticasInput,
    ResultadoCalculo,
    ResultadoVariacion,
    VariacionInput,
)

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}
MAX_EXPONENT = 100
MAX_EXPRESSION_LENGTH = 300


def safe_eval(expression: str) -> float:
    """Evalúa una expresión aritmética; cualquier otra construcción lanza ValueError."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expresión demasiado larga")
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponente mayor a {MAX_EXPONENT}")
            return _BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
        raise ValueError(f"construcción no permitida: {type(node).__name__}")

    return _eval(tree)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


ToolOutput = tuple[str, dict[str, Any]]


def _error(message: str) -> ToolOutput:
    return f"Error: {message}", ErrorHerramienta(error=message).model_dump()


@tool(args_schema=CalcularInput, response_format="content_and_artifact")
def calcular(expresion: str) -> ToolOutput:
    """Evalúa una expresión aritmética. Admite + - * / // % **, paréntesis y las funciones
    abs, round, min, max y sqrt. Usa punto decimal (3.9, no 3,9).
    Ejemplo: '(28 - 14) / 14 * 100'."""
    try:
        value = float(safe_eval(expresion))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return _error(f"no se pudo calcular '{expresion}' ({exc})")
    result = ResultadoCalculo(expresion=expresion, resultado=value)
    return f"{expresion} = {_fmt(value)}", result.model_dump()


@tool(args_schema=EstadisticasInput, response_format="content_and_artifact")
def estadisticas(valores: list[float]) -> ToolOutput:
    """Calcula estadísticas descriptivas de una lista de números: cantidad, suma, promedio,
    mediana, mínimo, máximo y desvío estándar muestral."""
    result = Estadisticas(
        n=len(valores),
        suma=round(sum(valores), 4),
        promedio=round(statistics.fmean(valores), 4),
        mediana=round(statistics.median(valores), 4),
        minimo=min(valores),
        maximo=max(valores),
        desvio_estandar=round(statistics.stdev(valores), 4) if len(valores) > 1 else 0.0,
    )
    return result.model_dump_json(), result.model_dump()


@tool(args_schema=VariacionInput, response_format="content_and_artifact")
def variacion_porcentual(valor_inicial: float, valor_final: float) -> ToolOutput:
    """Variación porcentual de valor_inicial a valor_final: (final - inicial) / |inicial| * 100."""
    change = (valor_final - valor_inicial) / abs(valor_inicial) * 100
    result = ResultadoVariacion(valor_inicial=valor_inicial, valor_final=valor_final, variacion_porcentual=round(change, 4))
    return f"Variación de {_fmt(valor_inicial)} a {_fmt(valor_final)}: {change:+.2f}%", result.model_dump()


for _tool in (calcular, estadisticas, variacion_porcentual):
    _tool.handle_validation_error = validation_message  # type: ignore[assignment]
