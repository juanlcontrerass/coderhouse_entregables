"""API REST del orquestador multi-agente.

Arranque:
    uvicorn app.main:app --port 8000

Ningún endpoint ejecuta el grafo: `POST /tasks` guarda el job en Redis, lo encola en Arq y
responde 202 con el `job_id` en milisegundos. Todas las operaciones de E/S son asíncronas
(`redis.asyncio`), así el event loop nunca se bloquea.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from redis.asyncio import Redis

from app.config import REDIS_URL
from app.hitl import router as hitl_router
from app.jobs import JobStatus, JobStore, TaskQueue, get_queue, get_store
from app.schemas import TaskCreated, TaskRequest, TaskStatusResponse

logger = logging.getLogger("app.api")


def create_app(*, redis: Redis | None = None, queue: TaskQueue | None = None) -> FastAPI:
    """Crea la aplicación. Los tests inyectan un Redis falso y una cola simulada."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        own_redis = redis is None
        app.state.redis = redis or Redis.from_url(REDIS_URL, decode_responses=True)
        app.state.store = JobStore(app.state.redis)
        pool = None
        if queue is None:
            from arq import create_pool
            from arq.connections import RedisSettings

            pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        app.state.queue = queue or pool
        try:
            yield
        finally:
            if pool is not None:
                await pool.aclose()
            if own_redis:
                await app.state.redis.aclose()

    app = FastAPI(
        title="Orquestador multi-agente TechCorp",
        version="1.0.0",
        description="API asíncrona: encola consultas para el equipo de agentes, consulta su estado "
        "y aprueba las tareas críticas (Human-in-the-loop).",
        lifespan=lifespan,
    )
    app.include_router(hitl_router)

    @app.post(
        "/tasks",
        response_model=TaskCreated,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Encola una consulta y devuelve su job_id",
        tags=["tareas"],
    )
    async def create_task(task: TaskRequest, request: Request) -> TaskCreated:
        store = get_store(request)
        job_id = uuid.uuid4().hex
        await store.create(job_id, task.query, task.publish)
        try:
            await get_queue(request).enqueue_job("run_task", job_id, _job_id=job_id)
        except Exception as exc:
            await store.update(job_id, status=JobStatus.FAILED, error=f"No se pudo encolar la tarea: {exc}")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola de tareas no está disponible.") from exc
        return TaskCreated(job_id=job_id, status=JobStatus.PENDING, status_url=f"/tasks/{job_id}")

    @app.get(
        "/tasks/{job_id}",
        response_model=TaskStatusResponse,
        summary="Estado actual de una tarea (para polling)",
        tags=["tareas"],
    )
    async def get_task(job_id: str, request: Request) -> TaskStatusResponse:
        job = await get_store(request).get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la tarea {job_id}.")
        return TaskStatusResponse.from_job(job)

    @app.get("/health", summary="Verifica la conexión con Redis", tags=["sistema"])
    async def health(request: Request) -> dict[str, str]:
        try:
            await request.app.state.redis.ping()
        except Exception as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {exc}") from exc
        return {"status": "ok", "redis": "ok"}

    return app


app = create_app()
