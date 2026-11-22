"""Cliente del LLM.

Hereda las ideas del cliente unificado del Módulo 1: llamadas asíncronas, timeout explícito y
reintentos con backoff exponencial ante errores transitorios (429, 5xx, cortes de red). Aquí
los resuelve el SDK oficial de OpenAI, que usa `httpx.AsyncClient` por debajo: ninguna llamada
bloquea el event loop.
"""

from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.require_openai_key(),
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        stream_usage=True,  # tokens en usage_metadata, también en streaming
    )
