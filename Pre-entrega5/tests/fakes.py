"""LLM falso con respuestas programadas: permite probar el grafo real sin API key."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    """Respuesta del modelo que pide ejecutar una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Devuelve las respuestas en orden y repite la última al agotarlas.

    Guarda lo que recibió en cada llamada, que es lo que vería OpenAI.
    """

    responses: list[AIMessage]
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools = [tool.name for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(len(self.received), len(self.responses) - 1)
        self.received.append(list(messages))
        # Copia: como en OpenAI, cada respuesta recibe su propio id; si se repitiera,
        # add_messages la fusionaría con la anterior en lugar de agregarla al historial
        return ChatResult(generations=[ChatGeneration(message=self.responses[index].model_copy())])
