from typing import AsyncGenerator, List

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from .base import BaseLLMClient
from .retry import con_reintentos
from .schemas import ChatMessage, ModelResponse, Provider

HF_ROUTER_URL = "https://router.huggingface.co/v1"


class ZaiClient(BaseLLMClient):
    """Modelos GLM de Z.ai, servidos por el router de HuggingFace (API compatible con OpenAI)."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, max_retries: int = 3):
        # max_retries=0: se desactiva el retry interno del SDK para que el nuestro sea el único
        self._client = AsyncOpenAI(base_url=HF_ROUTER_URL, api_key=api_key, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @staticmethod
    def _es_reintentable(e: Exception) -> bool:
        if isinstance(e, (RateLimitError, APIConnectionError)):
            return True
        return isinstance(e, APIStatusError) and e.status_code >= 500

    @staticmethod
    def _describir_error(e: Exception) -> str:
        if isinstance(e, AuthenticationError):
            return f"API key inválida: {e}"
        if isinstance(e, RateLimitError):
            return f"Límite de cuota excedido: {e}"
        if isinstance(e, APIConnectionError):
            return f"Error de conexión: {e}"
        return f"Error de la API de Z.ai: {e}"

    async def _crear(self, messages: List[ChatMessage], stream: bool):
        return await con_reintentos(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
            ),
            self._es_reintentable,
            self.max_retries,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._crear(messages, stream=False)
            return ModelResponse(
                provider=Provider.ZAI,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except Exception as e:
            return ModelResponse(provider=Provider.ZAI, model=self.model, content="",
                                 error=self._describir_error(e))

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            # Solo se reintenta la apertura del stream: una vez emitido texto, reintentar lo duplicaría
            stream = await self._crear(messages, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {self._describir_error(e)}]"
