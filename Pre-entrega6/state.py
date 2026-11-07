"""Esquema del estado compartido del grafo.

Cada especialista agrega su aporte a `contributions` (reducer `operator.add`): nada se
sobrescribe y siempre queda registrado qué agente aportó qué, con qué instrucción y en
qué paso. El supervisor deja su razonamiento en `decisions`.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["researcher", "analyst", "synthesizer", "validator"]
SpecialistName = Literal["researcher", "analyst", "synthesizer"]


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
    """Hereda `messages` (con el reducer `add_messages`) y suma los campos del equipo."""

    next_agent: SpecialistName
    current_instruction: str  # lo único que recibe el especialista de turno
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    step_count: int
    revisions: dict[str, int]  # re-delegaciones por agente
    validation_feedback: str
    validation_attempts: int
    final_answer: str
    task_completed: bool
