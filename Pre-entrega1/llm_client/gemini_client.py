from typing import AsyncGenerator, List

from google import genai
from google.genai import errors, types

from .base import BaseLLMClient
from .retry import con_reintentos
from .schemas import ChatMessage, ModelResponse, Provider


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, max_retries: int = 3):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @staticmethod
    def _es_reintentable(e: Exception) -> bool:
        return isinstance(e, errors.APIError) and (e.code == 429 or e.code >= 500)

    @staticmethod
    def _describir_error(e: Exception) -> str:
        if isinstance(e, errors.APIError) and e.code == 429:
            return f"Límite de cuota excedido: {e}"
        return f"Error de la API de Gemini: {e}"

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    def _argumentos(self, messages: List[ChatMessage]) -> dict:
        contents, system_instruction = self._convertir_mensajes(messages)
        return dict(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=system_instruction,
            ),
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            argumentos = self._argumentos(messages)
            response = await con_reintentos(
                lambda: self._client.aio.models.generate_content(**argumentos),
                self._es_reintentable,
                self.max_retries,
            )
            return ModelResponse(provider=Provider.GEMINI, model=self.model, content=response.text or "")
        except Exception as e:
            return ModelResponse(provider=Provider.GEMINI, model=self.model, content="",
                                 error=self._describir_error(e))

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            argumentos = self._argumentos(messages)
            # Solo se reintenta la apertura del stream: una vez emitido texto, reintentar lo duplicaría
            stream = await con_reintentos(
                lambda: self._client.aio.models.generate_content_stream(**argumentos),
                self._es_reintentable,
                self.max_retries,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {self._describir_error(e)}]"
