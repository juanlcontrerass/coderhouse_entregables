"""Dependencias de FastAPI y la lógica compartida para encolar un turno."""

import logging
import uuid
from typing import Any, Protocol

from fastapi import HTTPException, Request, status
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.api.schemas import MessageRequest, TaskCreated
from app.storage.conversations import ConversationBusyError, ConversationStore
from app.storage.jobs import JobStatus, JobStore

logger = logging.getLogger("app.api")


class TaskQueue(Protocol):
    """Lo que la API necesita de la cola (lo cumple `arq.ArqRedis`)."""

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> Any: ...


def get_store(request: Request) -> JobStore:
    store: JobStore = request.app.state.store
    return store


def get_conversations(request: Request) -> ConversationStore:
    conversations: ConversationStore = request.app.state.conversations
    return conversations


def get_queue(request: Request) -> TaskQueue:
    queue: TaskQueue = request.app.state.queue
    return queue


def get_checkpointer(request: Request) -> BaseCheckpointSaver[Any]:
    checkpointer: BaseCheckpointSaver[Any] = request.app.state.checkpointer
    return checkpointer


def new_id() -> str:
    return uuid.uuid4().hex


async def submit_turn(request: Request, conversation_id: str, message: MessageRequest) -> TaskCreated:
    """Toma el turno de la conversación, crea el job y lo encola. Nunca ejecuta el grafo."""
    store, conversations = get_store(request), get_conversations(request)
    job_id = new_id()
    try:
        await conversations.acquire(conversation_id, job_id)
    except ConversationBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await store.create(job_id, conversation_id, message.query, message.publish)
    try:
        await get_queue(request).enqueue_job("run_task", job_id, _job_id=job_id)
    except Exception as exc:
        logger.exception("No se pudo encolar el job %s", job_id)
        await store.update(job_id, status=JobStatus.FAILED, error=f"No se pudo encolar la tarea: {exc}")
        await conversations.release(conversation_id, job_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola de tareas no está disponible.") from exc
    return TaskCreated(
        job_id=job_id,
        conversation_id=conversation_id,
        status=JobStatus.PENDING,
        status_url=f"/tasks/{job_id}",
        conversation_url=f"/conversations/{conversation_id}",
    )
