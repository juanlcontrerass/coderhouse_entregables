"""Modelos de entrada y salida de la API."""

from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, description="Consulta para el equipo de agentes")
    publish: bool = Field(
        default=False,
        description="Publicar el informe al terminar (acción crítica: requiere aprobación humana)",
    )


class TaskCreated(BaseModel):
    job_id: str
    status: str
    status_url: str


class ApprovalDecision(BaseModel):
    approved: bool = Field(description="true aprueba la acción pendiente; false la rechaza")
    comment: str = Field(default="", max_length=500, description="Comentario del revisor")


class TaskStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    publish: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    approval_request: dict[str, Any] | None = Field(
        default=None, description="Qué se pide aprobar (solo en WAITING_APPROVAL)"
    )
    approval: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def from_job(cls, job: dict[str, Any]) -> "TaskStatusResponse":
        return cls.model_validate({k: v for k, v in job.items() if v not in ("", None)})
