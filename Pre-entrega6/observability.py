"""Instrumentación OpenInference: exporta las trazas del orquestador a Phoenix (OTLP/HTTP).

- `OpenAIInstrumentor`: cada llamada del SDK de OpenAI (prompt, respuesta, tokens, tool calls).
- `LangChainInstrumentor`: la estructura del grafo (nodos de LangGraph, agentes ReAct, herramientas).
- Chroma no tiene instrumentor de OpenInference: `tools/retrieval.py` abre sus propios spans
  (RETRIEVER en la búsqueda y EMBEDDING en la indexación) con el tracer de este módulo.

Sin `setup_tracing()` (por ejemplo en los tests) el tracer de OpenTelemetry es un no-op.
"""

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider

from config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME, TRACING_ENABLED

_provider: TracerProvider | None = None


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def langchain_parent_context() -> Context | None:
    """Contexto con el span de LangChain en curso (p. ej. el TOOL que llamó a Chroma).

    El instrumentor de LangChain asocia los spans a los run_id de sus callbacks, pero no los fija
    en el contexto de OpenTelemetry: sin esto, un span manual dentro de una herramienta quedaría
    como una traza aparte en vez de colgar del agente que lo originó.
    """
    from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span

    if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
        return None
    span = get_current_span()
    return trace.set_span_in_context(span) if span is not None else None


def setup_tracing() -> TracerProvider | None:
    """Registra el exportador hacia Phoenix e instrumenta OpenAI y LangChain (idempotente)."""
    global _provider
    if _provider is not None or not TRACING_ENABLED:
        return _provider

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    _provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_COLLECTOR_ENDPOINT,
        protocol="http/protobuf",
        batch=True,
        set_global_tracer_provider=True,
        verbose=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=_provider)
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    print(f"Trazas OpenInference → {PHOENIX_COLLECTOR_ENDPOINT} (proyecto '{PHOENIX_PROJECT_NAME}')")
    return _provider


def shutdown_tracing() -> None:
    """Envía los spans pendientes del BatchSpanProcessor antes de salir."""
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
