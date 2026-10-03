from typing import AsyncGenerator, List

from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider
from .zai_client import ZaiClient

CLIENTES: dict[Provider, type[BaseLLMClient]] = {
    Provider.GEMINI: GeminiClient,
    Provider.ZAI: ZaiClient,
}


class AsyncLLMManager:
    """Factory: cambiar de proveedor es cambiar `config.provider`, no reescribir código."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        clase = CLIENTES.get(self.config.provider)
        if clase is None:
            raise ValueError(f"Proveedor no soportado: {self.config.provider}")
        return clase(
            api_key=self.config.api_key.get_secret_value(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages):
            yield chunk
