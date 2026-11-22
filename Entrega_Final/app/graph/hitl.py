"""Human-in-the-loop: detección de tareas críticas, pausa del grafo y publicación.

Una tarea es crítica cuando:
1. pide una acción con efecto lateral (publicar o enviar el informe), o
2. consumió en el turno más tokens que `COST_APPROVAL_TOKEN_THRESHOLD` (costo alto).

En ese caso, después de validar la respuesta, el grafo entra en `human_approval`, que llama a
`interrupt()`: LangGraph guarda el checkpoint en Redis y la ejecución se detiene hasta que
alguien responde en `POST /tasks/{id}/approve`. El worker reanuda con `Command(resume=...)`.
"""

import re
import unicodedata
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agents.common import user_query
from app.agents.synthesizer import route_from_validator
from app.core.config import get_settings
from app.graph.state import OrchestratorState
from app.tools.publisher import publicar_informe
from app.tools.schemas import InformePublicado

# Verbos que piden una acción con efecto lateral (se comparan sin tildes)
PUBLISH_WORDS = {
    "publica", "publicar", "publicalo", "publiquen", "publique",
    "envia", "enviar", "envialo", "envien", "envie",
    "difunde", "difundir", "difundelo",
}
PREVIEW_CHARS = 1500


class ApprovalRequest(BaseModel):
    """Lo que el grafo pide aprobar al detenerse (payload de `interrupt()`)."""

    tipo: Literal["aprobacion_requerida"] = "aprobacion_requerida"
    motivos: list[str] = Field(min_length=1)
    accion_pendiente: Literal["publicar_informe", "entregar_respuesta"]
    tokens_usados: int = Field(ge=0)
    vista_previa: str


class ApprovalDecision(BaseModel):
    approved: bool = Field(description="true aprueba la acción pendiente; false la rechaza")
    comment: str = Field(default="", max_length=500, description="Comentario del revisor")


def _words(text: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return set(re.findall(r"[a-z]+", plain))


def wants_publication(state: OrchestratorState) -> bool:
    """El usuario pidió publicar: por el flag del request o por la redacción de la consulta."""
    return bool(state.get("publish_requested")) or bool(_words(user_query(state)) & PUBLISH_WORDS)


def assess_risk(state: OrchestratorState) -> list[str]:
    """Motivos por los que la tarea es crítica; lista vacía si no lo es."""
    threshold = get_settings().cost_approval_token_threshold
    reasons = []
    if wants_publication(state):
        reasons.append("Efecto lateral: la tarea pide publicar el informe (herramienta publicar_informe).")
    tokens = state.get("tokens_used", 0)
    if tokens > threshold:
        reasons.append(f"Costo alto: el turno consumió {tokens} tokens (umbral {threshold}).")
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
    request = ApprovalRequest(
        motivos=reasons,
        accion_pendiente="publicar_informe" if wants_publication(state) else "entregar_respuesta",
        tokens_usados=state.get("tokens_used", 0),
        vista_previa=state.get("final_answer", "")[:PREVIEW_CHARS],
    )
    raw = interrupt(request.model_dump())
    decision = ApprovalDecision.model_validate(raw if isinstance(raw, dict) else {"approved": bool(raw)})
    verdict = "aprobada" if decision.approved else "rechazada"
    return {
        "risk_reasons": reasons,
        "approval": decision.model_dump(),
        "messages": [AIMessage(f"Revisión humana: {verdict}. {decision.comment}".strip(), name="human_approval")],
    }


def route_after_approval(state: OrchestratorState) -> Literal["publisher", "__end__"]:
    approved = bool((state.get("approval") or {}).get("approved"))
    return "publisher" if approved and wants_publication(state) else END


async def publisher(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Ejecuta la herramienta con efecto lateral, solo después de la aprobación."""
    configurable = config.get("configurable") or {}
    job_id = str(configurable.get("job_id") or configurable.get("thread_id") or "sin-id")
    comment = (state.get("approval") or {}).get("comment", "")
    message = await publicar_informe.ainvoke(
        {
            "type": "tool_call",
            "id": f"publish-{job_id}",
            "name": publicar_informe.name,
            "args": {
                "job_id": job_id,
                "consulta": user_query(state),
                "contenido": state.get("final_answer", "") or "(sin contenido)",
                "aprobado_por": comment,
            },
        }
    )
    report = InformePublicado.model_validate(message.artifact)
    return {
        "published_path": report.ruta,
        "messages": [AIMessage(f"Informe publicado en {report.ruta}", name="publisher")],
    }
