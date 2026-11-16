"""Human-in-the-loop: detección de tareas críticas, pausa del grafo y endpoint de aprobación.

Una tarea es crítica cuando:
1. pide una acción con efecto lateral (publicar o enviar el informe), o
2. consumió más tokens que `COST_APPROVAL_TOKEN_THRESHOLD` (costo alto).

En ese caso, después de validar la respuesta, el grafo entra en `human_approval`, que llama a
`interrupt()`: LangGraph guarda el checkpoint en Redis y la ejecución se detiene hasta que
alguien responde en `POST /tasks/{id}/approve`. El worker reanuda con `Command(resume=...)`.
"""

import re
import unicodedata
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

from app.agents.common import user_query
from app.agents.synthesizer import route_from_validator
from app.config import COST_APPROVAL_TOKEN_THRESHOLD
from app.jobs import JobStatus, get_queue, get_store
from app.schemas import ApprovalDecision, TaskStatusResponse
from app.state import OrchestratorState
from app.tools.publisher import publicar_informe

# Verbos que piden una acción con efecto lateral (se comparan sin tildes)
PUBLISH_WORDS = {
    "publica", "publicar", "publicalo", "publiquen", "publique",
    "envia", "enviar", "envialo", "envien", "envie",
    "difunde", "difundir", "difundelo",
}
PREVIEW_CHARS = 1500


def _words(text: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return set(re.findall(r"[a-z]+", plain))


def wants_publication(state: OrchestratorState) -> bool:
    """El usuario pidió publicar: por el flag del request o por la redacción de la consulta."""
    return bool(state.get("publish_requested")) or bool(_words(user_query(state)) & PUBLISH_WORDS)


def assess_risk(state: OrchestratorState) -> list[str]:
    """Motivos por los que la tarea es crítica; lista vacía si no lo es."""
    reasons = []
    if wants_publication(state):
        reasons.append("Efecto lateral: la tarea pide publicar el informe (herramienta publicar_informe).")
    tokens = state.get("tokens_used", 0)
    if tokens > COST_APPROVAL_TOKEN_THRESHOLD:
        reasons.append(f"Costo alto: la ejecución consumió {tokens} tokens (umbral {COST_APPROVAL_TOKEN_THRESHOLD}).")
    return reasons


def route_after_validation(state: OrchestratorState) -> Literal["supervisor", "human_approval", "__end__"]:
    """Tras el validador: reintento, pausa para aprobación humana o fin."""
    if route_from_validator(state) == "supervisor":
        return "supervisor"
    return "human_approval" if assess_risk(state) else END


def human_approval(state: OrchestratorState) -> dict[str, Any]:
    """Punto de interrupción. Al reanudar, LangGraph vuelve a ejecutar este nodo desde el
    principio y `interrupt()` devuelve la decisión humana, por eso no tiene efectos laterales."""
    reasons = assess_risk(state)
    decision = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "motivos": reasons,
            "accion_pendiente": "publicar_informe" if wants_publication(state) else "entregar_respuesta",
            "tokens_usados": state.get("tokens_used", 0),
            "vista_previa": state.get("final_answer", "")[:PREVIEW_CHARS],
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        comment = str(decision.get("comment", ""))
    else:
        approved, comment = bool(decision), ""
    verdict = "aprobada" if approved else "rechazada"
    return {
        "risk_reasons": reasons,
        "approval": {"approved": approved, "comment": comment},
        "messages": [AIMessage(f"Revisión humana: {verdict}. {comment}".strip(), name="human_approval")],
    }


def route_after_approval(state: OrchestratorState) -> Literal["publisher", "__end__"]:
    approved = bool((state.get("approval") or {}).get("approved"))
    return "publisher" if approved and wants_publication(state) else END


async def publisher(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Ejecuta la herramienta con efecto lateral, solo después de la aprobación."""
    job_id = str((config.get("configurable") or {}).get("thread_id", "sin-id"))
    comment = (state.get("approval") or {}).get("comment", "")
    path = await publicar_informe.ainvoke(
        {
            "job_id": job_id,
            "consulta": user_query(state),
            "contenido": state.get("final_answer", ""),
            "aprobado_por": comment,
        }
    )
    return {
        "published_path": str(path),
        "messages": [AIMessage(f"Informe publicado en {path}", name="publisher")],
    }


# --- Endpoint de aprobación ---

router = APIRouter(tags=["human-in-the-loop"])


@router.post(
    "/tasks/{job_id}/approve",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aprueba o rechaza una tarea en pausa",
)
async def approve_task(job_id: str, decision: ApprovalDecision, request: Request) -> TaskStatusResponse:
    store = get_store(request)
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la tarea {job_id}.")
    # Transición atómica: dos aprobaciones simultáneas no pueden reanudar dos veces el grafo
    resumed = await store.transition(
        job_id,
        expected=JobStatus.WAITING_APPROVAL,
        new=JobStatus.PENDING,
        approval=decision.model_dump(),
    )
    if not resumed:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"La tarea {job_id} no espera aprobación (estado actual: {job['status']}).",
        )
    try:
        await get_queue(request).enqueue_job(
            "resume_task", job_id, decision.model_dump(), _job_id=f"{job_id}:resume:{uuid.uuid4().hex[:8]}"
        )
    except Exception as exc:
        await store.update(job_id, status=JobStatus.FAILED, error=f"No se pudo encolar la reanudación: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "La cola de tareas no está disponible.") from exc
    return TaskStatusResponse.from_job(await store.get(job_id) or job)
