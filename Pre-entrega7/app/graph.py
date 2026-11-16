"""Grafo jerárquico del Módulo 6 + Human-in-the-loop + checkpointer de Redis.

    Supervisor → {Investigador, Analista, Síntesis} → Validación
        → (tarea crítica) human_approval [interrupt] → (aprobada y pide publicar) publisher → END
        → (no crítica) END

El estado de cada ejecución se guarda en Redis con `AsyncRedisSaver`, usando el `job_id` como
`thread_id`: así una tarea en pausa sobrevive a reinicios del worker y puede reanudarse desde
cualquier proceso.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from app.config import REDIS_URL
from app.hitl import human_approval, publisher, route_after_approval, route_after_validation
from app.state import OrchestratorState


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


@asynccontextmanager
async def redis_checkpointer(redis_url: str = REDIS_URL) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Checkpointer de LangGraph en Redis. Requiere los módulos RedisJSON y RediSearch,
    incluidos en Redis 8 (imagen `redis:8`) y en Redis Stack."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()  # crea los índices si no existen
        yield saver
