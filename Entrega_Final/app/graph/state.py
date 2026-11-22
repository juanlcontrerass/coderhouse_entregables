"""Esquema del estado compartido del grafo.

El estado vive en el checkpointer bajo el `thread_id` de la conversación, así que tiene dos
tipos de campos:

- **De la conversación** (se acumulan entre turnos): `messages` y `total_tokens`.
- **Del turno** (se reinician con cada pregunta): `contributions`, `decisions`,
  `tokens_used`, contadores y resultado. Los que usan reducers acumulativos se vacían
  enviando `None` al inicio del turno (`extend_or_reset`, `add_or_reset`).
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["researcher", "analyst", "synthesizer", "validator"]
SpecialistName = Literal["researcher", "analyst", "synthesizer"]


def extend_or_reset(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Reducer de listas: agrega; `None` reinicia la lista (nuevo turno)."""
    if right is None:
        return []
    return [*(left or []), *right]


def add_or_reset(left: int | None, right: int | None) -> int:
    """Reducer de contadores: suma; `None` reinicia a 0 (nuevo turno)."""
    if right is None:
        return 0
    return (left or 0) + right


class Contribution(TypedDict):
    agent: AgentName
    instruction: str  # lo que pidió el supervisor
    content: str  # el resultado del agente
    sources: list[str]  # archivos efectivamente recuperados (solo el investigador)
    iteration: int  # paso del supervisor en que se produjo


class Decision(TypedDict):
    step: int
    next_agent: SpecialistName
    instruction: str
    reasoning: str
    forced: bool  # True si la impuso una guarda y no el LLM


class OrchestratorState(MessagesState):
    """Hereda `messages` (reducer `add_messages`) y suma los campos del equipo."""

    next_agent: SpecialistName
    current_instruction: str  # lo único que recibe el especialista de turno
    contributions: Annotated[list[Contribution], extend_or_reset]
    decisions: Annotated[list[Decision], extend_or_reset]
    step_count: int
    revisions: dict[str, int]  # re-delegaciones por agente
    validation_feedback: str
    validation_attempts: int
    final_answer: str
    task_completed: bool
    tokens_used: Annotated[int, add_or_reset]  # tokens del turno
    total_tokens: Annotated[int, operator.add]  # tokens de toda la conversación
    publish_requested: bool  # el usuario pidió publicar el informe (efecto lateral)
    risk_reasons: list[str]  # por qué la tarea se consideró crítica
    approval: dict[str, Any]  # decisión humana recibida al reanudar
    published_path: str  # dónde quedó el informe publicado
