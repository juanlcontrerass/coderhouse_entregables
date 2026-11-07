"""Herramientas acotadas por rol: cada especialista recibe solo las suyas."""

from tools.calculator import calcular, estadisticas, variacion_porcentual
from tools.retrieval import RESEARCH_TOOLS
from tools.sentiment import analizar_sentimiento

ANALYST_TOOLS = [calcular, estadisticas, variacion_porcentual, analizar_sentimiento]

__all__ = ["ANALYST_TOOLS", "RESEARCH_TOOLS"]
