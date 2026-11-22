"""Estado de los jobs en Redis (hash `job:{id}`), siempre con el cliente asíncrono.

Un job es la ejecución de un turno de una conversación. Ciclo de vida:
    PENDING → RUNNING → DONE
                      → WAITING_APPROVAL → (aprobación) PENDING → RUNNING → DONE | REJECTED
                      → FAILED (ante cualquier excepción del worker)
"""

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.core.config import get_settings


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.REJECTED, JobStatus.FAILED}
# Campos que se guardan serializados como JSON dentro del hash
JSON_FIELDS = {"result", "approval_request", "approval"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _encode(fields: dict[str, Any]) -> dict[str, str]:
    encoded = {}
    for key, value in fields.items():
        if key in JSON_FIELDS:
            encoded[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            encoded[key] = "1" if value else "0"
        elif value is None:
            encoded[key] = ""
        else:
            encoded[key] = str(value)
    return encoded


def _decode(raw: dict[str, str]) -> dict[str, Any]:
    job: dict[str, Any] = dict(raw)
    for key in JSON_FIELDS & raw.keys():
        job[key] = json.loads(raw[key]) if raw[key] else None
    job["publish"] = raw.get("publish") == "1"
    return job


class JobStore:
    """Lectura y escritura del estado de los jobs."""

    def __init__(self, redis: Redis, ttl: int | None = None) -> None:
        self.redis = redis
        self.ttl = ttl or get_settings().job_ttl_seconds

    @staticmethod
    def key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, job_id: str, conversation_id: str, query: str, publish: bool = False) -> dict[str, Any]:
        timestamp = now_iso()
        fields = {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "status": JobStatus.PENDING,
            "query": query,
            "publish": publish,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.key(job_id), mapping=_encode(fields))
            pipe.expire(self.key(job_id), self.ttl)
            await pipe.execute()
        return _decode(_encode(fields))

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hgetall(self.key(job_id))
        return _decode(raw) if raw else None

    async def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = now_iso()
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.key(job_id), mapping=_encode(fields))
            pipe.expire(self.key(job_id), self.ttl)
            await pipe.execute()

    async def transition(self, job_id: str, *, expected: JobStatus, new: JobStatus, **fields: Any) -> bool:
        """Cambia el estado solo si el actual es `expected` (WATCH/MULTI: atómico)."""
        key = self.key(job_id)
        fields.update(status=new, updated_at=now_iso())
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.hget(key, "status") != expected:
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.hset(key, mapping=_encode(fields))
                await pipe.execute()
                return True
            except WatchError:  # otro cliente lo modificó en el medio
                return False
