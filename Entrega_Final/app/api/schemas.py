"""Contratos Pydantic de la API: toda entrada y toda salida pasa por estos modelos."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.graph.hitl import ApprovalDecision, ApprovalRequest
from app.storage.jobs import JobStatus

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ConversationCreated",
    "ConversationOut",
    "DecisionOut",
    "ErrorResponse",
    "HealthResponse",
    "MessageRequest",
    "TaskCreated",
    "TaskResult",
    "TaskStatusResponse",
    "TurnOut",
]


class MessageRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000, description="Consulta para el equipo de agentes")
    publish: bool = Field(
        default=False,
        description="Publicar el informe al terminar (acción crítica: requiere aprobación humana)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "¿Cuántos días de teletrabajo permite la política y qué porcentaje de la semana representan?"},
                {"query": "Calcula la variación del eNPS 2025 respecto de 2024 y publica el informe", "publish": True},
            ]
        }
    )

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("la consulta debe tener al menos 3 caracteres útiles")
        return value


class TaskCreated(BaseModel):
    job_id: str
    conversation_id: str
    status: JobStatus
    status_url: str
    conversation_url: str


class DecisionOut(BaseModel):
    step: int
    next_agent: str
    instruction: str
    reasoning: str
    forced: bool


class TaskResult(BaseModel):
    """Resultado de un turno, tal como lo guarda el worker en Redis."""

    final_answer: str
    route: list[str] = Field(description="Secuencia de agentes que eligió el supervisor")
    decisions: list[DecisionOut] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="Archivos recuperados por el investigador")
    steps: int = Field(ge=0)
    tokens_used: int = Field(ge=0, description="Tokens del turno")
    total_tokens: int = Field(ge=0, description="Tokens acumulados de la conversación")
    task_completed: bool
    risk_reasons: list[str] = Field(default_factory=list)
    approval: ApprovalDecision | None = None
    published_path: str | None = None


class TaskStatusResponse(BaseModel):
    job_id: str
    conversation_id: str
    status: JobStatus
    query: str
    publish: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    trace_id: str | None = Field(default=None, description="Traza de la última ejecución en Phoenix")
    approval_request: ApprovalRequest | None = Field(default=None, description="Qué se pide aprobar (solo en WAITING_APPROVAL)")
    approval: ApprovalDecision | None = None
    result: TaskResult | None = None
    error: str | None = None

    @classmethod
    def from_job(cls, job: dict[str, Any]) -> "TaskStatusResponse":
        return cls.model_validate({k: v for k, v in job.items() if v not in ("", None)})


class ConversationCreated(BaseModel):
    conversation_id: str
    messages_url: str


class TurnOut(BaseModel):
    query: str
    answer: str | None = Field(description="None si el turno todavía no terminó")


class ConversationOut(BaseModel):
    conversation_id: str
    created_at: str
    updated_at: str
    active_job_id: str | None = None
    job_ids: list[str]
    turns: list[TurnOut] = Field(description="Historial leído del checkpointer de LangGraph")


class HealthResponse(BaseModel):
    status: str
    redis: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
