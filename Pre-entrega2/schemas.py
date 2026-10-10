"""Contrato de salida del pipeline: qué debe devolver el LLM y con qué restricciones."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """Entidades técnicas extraídas de un texto (log de error, descripción de arquitectura, etc.)."""

    tecnologias: list[str] = Field(
        min_length=1,
        description="Tecnologías, frameworks, lenguajes, bases de datos o servicios mencionados en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Impacto del problema o riesgo descrito: baja, media o alta.",
    )
    resumen_tecnico: str = Field(
        min_length=10,
        max_length=400,
        description="Resumen técnico del texto en una o dos oraciones, en español.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valor: list[str]) -> list[str]:
        # dict.fromkeys deduplica conservando el orden en que el modelo las devolvió
        limpias = list(dict.fromkeys(t.strip() for t in valor if t.strip()))
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias


class RespuestaIncompletaError(Exception):
    """El LLM devolvió una respuesta truncada (finish_reason=length) o que no cumple el esquema."""
