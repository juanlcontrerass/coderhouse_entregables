"""Human-in-the-loop: aprobación o rechazo de una tarea en pausa."""

import uuid

from fastapi import APIRouter, HTTPException, Request, status

from app.api.deps import get_conversations, get_queue, get_store
from app.api.schemas import ApprovalDecision, ErrorResponse, TaskStatusResponse
from app.storage.jobs import JobStatus

router = APIRouter(prefix="/tasks", tags=["human-in-the-loop"])


@router.post(
    "/{job_id}/approve",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aprueba o rechaza una tarea en pausa",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def approve_task(job_id: str, decision: ApprovalDecision, request: Request) -> TaskStatusResponse:
    store = get_store(request)
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la tarea {job_id}.")
    # Transición atómica: dos aprobaciones simultáneas no pueden reanudar dos veces el grafo
    resumed = await store.transition(
        job_id, expected=JobStatus.WAITING_APPROVAL, new=JobStatus.PENDING, approval=decision.model_dump()
    )
    if not resumed:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"La tarea {job_id} no espera aprobación (estado actual: {job['status']})."
        )
    try:
        await get_queue(request).enqueue_job(
            "resume_task", job_id, decision.model_dump(), _job_id=f"{job_id}:resume:{uuid.uuid4().hex[:8]}"
        )
    except Exception as exc:
        await store.update(job_id, status=JobStatus.FAILED, error=f"No se pudo encolar la reanudación: {exc}")
        await get_conversations(request).release(job["conversation_id"], job_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola de tareas no está disponible.") from exc
    return TaskStatusResponse.from_job(await store.get(job_id) or job)
