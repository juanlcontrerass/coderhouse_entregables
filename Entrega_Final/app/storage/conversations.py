"""Metadatos de las conversaciones en Redis (hash `conversation:{id}` + lista de jobs).

El historial de mensajes NO vive aquí: vive en el checkpointer de LangGraph (thread_id =
conversation_id). Este registro solo sabe qué jobs tuvo la conversación y cuál está activo.

Una conversación procesa un turno a la vez: `acquire()` toma el turno de forma atómica
(WATCH/MULTI). El turno se libera cuando el job termina (DONE, REJECTED, FAILED); mientras
espera aprobación humana sigue tomado, porque el grafo está detenido en ese thread.
"""

from datetime import datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.core.config import get_settings
from app.storage.jobs import TERMINAL_STATUSES, JobStore, now_iso


class ConversationBusyError(Exception):
    def __init__(self, conversation_id: str, active_job_id: str) -> None:
        super().__init__(f"La conversación {conversation_id} tiene un turno en curso ({active_job_id}).")
        self.active_job_id = active_job_id


class ConversationStore:
    def __init__(self, redis: Redis, jobs: JobStore, ttl: int | None = None) -> None:
        self.redis = redis
        self.jobs = jobs
        self.ttl = ttl or get_settings().conversation_ttl_minutes * 60

    @staticmethod
    def key(conversation_id: str) -> str:
        return f"conversation:{conversation_id}"

    @staticmethod
    def jobs_key(conversation_id: str) -> str:
        return f"conversation:{conversation_id}:jobs"

    async def create(self, conversation_id: str) -> dict[str, Any]:
        timestamp = now_iso()
        fields = {"conversation_id": conversation_id, "created_at": timestamp, "updated_at": timestamp, "active_job_id": ""}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.key(conversation_id), mapping=fields)
            pipe.expire(self.key(conversation_id), self.ttl)
            await pipe.execute()
        return fields

    async def get(self, conversation_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hgetall(self.key(conversation_id))
        if not raw:
            return None
        return {**raw, "job_ids": await self.redis.lrange(self.jobs_key(conversation_id), 0, -1)}

    async def _is_stale(self, job_id: str) -> bool:
        """El job activo ya terminó, expiró o quedó colgado (un worker murió a mitad de camino)."""
        job = await self.jobs.get(job_id)
        if job is None or job["status"] in TERMINAL_STATUSES:
            return True
        settings = get_settings()
        limit = timedelta(seconds=settings.job_timeout_seconds + 120)
        return job["status"] == "RUNNING" and datetime.fromisoformat(job["updated_at"]) + limit < datetime.fromisoformat(now_iso())

    async def acquire(self, conversation_id: str, job_id: str) -> None:
        """Registra `job_id` como turno activo o lanza ConversationBusyError."""
        key = self.key(conversation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                active = await pipe.hget(key, "active_job_id")
                if active and not await self._is_stale(active):
                    await pipe.unwatch()
                    raise ConversationBusyError(conversation_id, active)
                pipe.multi()
                pipe.hset(key, mapping={"active_job_id": job_id, "updated_at": now_iso()})
                pipe.rpush(self.jobs_key(conversation_id), job_id)
                pipe.expire(key, self.ttl)
                pipe.expire(self.jobs_key(conversation_id), self.ttl)
                await pipe.execute()
            except WatchError as exc:  # otra petición tomó el turno en el medio
                raise ConversationBusyError(conversation_id, "en curso") from exc

    async def release(self, conversation_id: str, job_id: str) -> None:
        """Libera el turno solo si sigue siendo de `job_id`."""
        key = self.key(conversation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.hget(key, "active_job_id") != job_id:
                    await pipe.unwatch()
                    return
                pipe.multi()
                pipe.hset(key, mapping={"active_job_id": "", "updated_at": now_iso()})
                await pipe.execute()
            except WatchError:
                return
