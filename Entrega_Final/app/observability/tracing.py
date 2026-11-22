"""Instrumentación con Arize Phoenix (OpenTelemetry + OpenInference).

- `LangChainInstrumentor` genera un span por cada nodo del grafo, agente ReAct, herramienta y
  llamada al LLM, con el modelo y los tokens de entrada y salida; con eso Phoenix calcula el
  costo de cada ejecución.
- `tracer` es un `OITracer`: su decorador `@tracer.chain` crea el span raíz de cada job.
- `using_session(conversation_id)` agrupa en la pestaña *Sessions* todos los turnos de una
  conversación (y la reanudación tras una aprobación humana).
- El exportador es `BatchSpanProcessor`: envía los spans desde un hilo propio, sin frenar al
  event loop ni a los agentes.
"""

import logging

from openinference.instrumentation import OITracer, TraceConfig, using_session
from opentelemetry import trace as trace_api

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Usa el proveedor global: hasta que se llame a init_observability() es un no-op
tracer = OITracer(trace_api.get_tracer("app.intelligence"), config=TraceConfig())

_provider = None

__all__ = ["current_trace_id", "init_observability", "shutdown_observability", "tracer", "using_session"]


def init_observability(service: str = "worker") -> None:
    """Registra Phoenix como destino de las trazas e instrumenta LangChain/LangGraph."""
    global _provider
    settings = get_settings()
    if _provider is not None:
        return
    if not settings.phoenix_enabled:
        logger.info("Observabilidad desactivada (PHOENIX_ENABLED=false).")
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        batch=True,
        set_global_tracer_provider=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    logger.info(
        "Trazas de %s enviadas a Phoenix (%s, proyecto %s).",
        service,
        settings.phoenix_collector_endpoint,
        settings.phoenix_project_name,
    )


def current_trace_id() -> str | None:
    """Id (hex) de la traza activa; None si no hay una que se esté registrando."""
    context = trace_api.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def shutdown_observability() -> None:
    """Vacía el buffer de spans antes de terminar el proceso."""
    global _provider
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
        _provider = None
