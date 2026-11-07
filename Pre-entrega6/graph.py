"""Grafo jerárquico: Supervisor → {Investigador, Analista, Síntesis} → Validación → END."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import (
    make_analyst_node,
    make_research_node,
    make_supervisor_node,
    make_synthesizer_node,
    route_from_supervisor,
    route_from_validator,
    validator,
)
from state import OrchestratorState


def build_graph(
    llm: BaseChatModel,
    *,
    researcher_llm: BaseChatModel | None = None,
    analyst_llm: BaseChatModel | None = None,
    synthesizer_llm: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """`llm` lo usa el supervisor y, por defecto, el resto. Los tests pasan uno por rol."""
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", make_supervisor_node(llm))
    graph.add_node("researcher", make_research_node(researcher_llm or llm))
    graph.add_node("analyst", make_analyst_node(analyst_llm or llm))
    graph.add_node("synthesizer", make_synthesizer_node(synthesizer_llm or llm))
    graph.add_node("validator", validator)

    graph.add_edge(START, "supervisor")
    # Los destinos posibles se infieren del Literal que devuelve route_from_supervisor
    graph.add_conditional_edges("supervisor", route_from_supervisor)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("synthesizer", "validator")
    graph.add_conditional_edges("validator", route_from_validator, {"supervisor": "supervisor", END: END})
    return graph.compile()
