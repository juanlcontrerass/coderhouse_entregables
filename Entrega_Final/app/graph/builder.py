"""Construcción del grafo jerárquico.

    Supervisor → {Investigador, Analista, Síntesis} → Validación
        → (tarea crítica) human_approval [interrupt] → (aprobada y pide publicar) publisher → END
        → (no crítica) END

Se compila con el checkpointer que se le pase (Redis en producción, memoria en los tests): el
`thread_id` es el `conversation_id`, así la conversación sobrevive a reinicios.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    make_analyst_node,
    make_research_node,
    make_supervisor_node,
    make_synthesizer_node,
    route_from_supervisor,
    validator,
)
from app.graph.hitl import human_approval, publisher, route_after_approval, route_after_validation
from app.graph.state import OrchestratorState


def build_graph(
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
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
    graph.add_node("human_approval", human_approval)
    graph.add_node("publisher", publisher)

    graph.add_edge(START, "supervisor")
    # Los destinos posibles se infieren del Literal que devuelve route_from_supervisor
    graph.add_conditional_edges("supervisor", route_from_supervisor)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("synthesizer", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"supervisor": "supervisor", "human_approval": "human_approval", END: END},
    )
    graph.add_conditional_edges("human_approval", route_after_approval, {"publisher": "publisher", END: END})
    graph.add_edge("publisher", END)
    return graph.compile(checkpointer=checkpointer)
