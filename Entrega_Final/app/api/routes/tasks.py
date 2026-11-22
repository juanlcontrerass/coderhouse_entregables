"""Tareas (jobs): atajo de un solo turno y consulta de estado por polling."""

from fastapi import APIRouter, HTTPException, Request, status

from app.api.deps import get_conversations, get_store, new_id, submit_turn
from app.api.schemas import ErrorResponse, MessageRequest, TaskCreated, TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tareas"])


@router.post(
    "",
    response_model=TaskCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Encola una consulta en una conversación nueva y devuelve su job_id",
    responses={503: {"model": ErrorResponse}},
)
async def create_task(task: MessageRequest, request: Request) -> TaskCreated:
    conversation_id = new_id()
    await get_conversations(request).create(conversation_id)
    return await submit_turn(request, conversation_id, task)


@router.get(
    "/{job_id}",
    response_model=TaskStatusResponse,
    summary="Estado actual de una tarea (para polling)",
    responses={404: {"model": ErrorResponse}},
)
async def get_task(job_id: str, request: Request) -> TaskStatusResponse:
    job = await get_store(request).get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la tarea {job_id}.")
    return TaskStatusResponse.from_job(job)
