from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator


class Provider(str, Enum):
    GEMINI = "gemini"
    ZAI = "zai"


class ChatMessage(BaseModel):
    role: str = Field(description="'user', 'assistant' o 'system'")
    content: str

    @field_validator("role")
    @classmethod
    def rol_valido(cls, v: str) -> str:
        roles_permitidos = {"user", "assistant", "system"}
        if v not in roles_permitidos:
            raise ValueError(f"role debe ser uno de {roles_permitidos}, recibido: '{v}'")
        return v


class LLMConfig(BaseModel):
    provider: Provider
    model: str
    api_key: SecretStr
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
