"""API REST de The Intelligence System.

Arranque:
    uvicorn app.api.main:app --port 8000

Ningún endpoint ejecuta el grafo: los turnos se guardan en Redis, se encolan en Arq y la API
responde 202 con el `job_id` en milisegundos. Toda la E/S es asíncrona (`redis.asyncio`,
`AsyncRedisSaver`), así el event loop nunca se bloquea.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from redis.asyncio import Redis

from app import __version__
from app.api.deps import TaskQueue
from app.api.routes import approvals, conversations, system, tasks
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.storage.conversations import ConversationStore
from app.storage.jobs import JobStore

DESCRIPTION = """
Sistema de IA de TechCorp: un **Supervisor** coordina agentes especializados (investigación con
RAG híbrido, análisis y síntesis) sobre la base documental interna.

- `POST /conversations` + `POST /conversations/{id}/messages`: conversación multi-turno; el
  historial persiste en Redis (checkpointer de LangGraph) y sobrevive a reinicios.
- `POST /tasks`: atajo para una consulta de un solo turno.
- `GET /tasks/{id}`: polling del estado (`PENDING`, `RUNNING`, `WAITING_APPROVAL`, `DONE`,
  `REJECTED`, `FAILED`); incluye el `trace_id` para abrir la traza en Phoenix.
- `POST /tasks/{id}/approve`: Human-in-the-loop para las tareas críticas.
"""


def create_app(
    *,
    redis: Redis | None = None,
    queue: TaskQueue | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> FastAPI:
    """Crea la aplicación. Los tests inyectan Redis falso, cola simulada y checkpointer en memoria."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        settings = get_settings()
        async with AsyncExitStack() as stack:
            if redis is None:
                app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
                stack.push_async_callback(app.state.redis.aclose)
            else:
                app.state.redis = redis
            if queue is None:
                from arq import create_pool
                from arq.connections import RedisSettings

                pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
                stack.push_async_callback(pool.aclose)
                app.state.queue = pool
            else:
                app.state.queue = queue
            if checkpointer is None:
                from app.storage.checkpointer import redis_checkpointer

                app.state.checkpointer = await stack.enter_async_context(redis_checkpointer())
            else:
                app.state.checkpointer = checkpointer
            app.state.store = JobStore(app.state.redis)
            app.state.conversations = ConversationStore(app.state.redis, app.state.store)
            yield

    app = FastAPI(
        title="The Intelligence System — TechCorp",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(conversations.router)
    app.include_router(tasks.router)
    app.include_router(approvals.router)
    app.include_router(system.router)
    return app


app = create_app()
