"""Instrumentación con Arize Phoenix (OpenTelemetry + OpenInference).

- `LangChainInstrumentor` genera un span por cada nodo del grafo, agente ReAct, herramienta y
  llamada al LLM, con el modelo y los tokens de entrada y salida; con eso Phoenix calcula el
  costo de cada ejecución.
- `tracer` es un `OITracer`: sus decoradores (`@tracer.chain`, `@tracer.tool`) agregan spans
  propios, por ejemplo el que envuelve cada job del worker.
- `using_session(job_id)` agrupa en Phoenix todas las trazas de un mismo job (la ejecución
  inicial y la reanudación después de la aprobación humana).
"""

import logging

from openinference.instrumentation import OITracer, TraceConfig, using_session
from opentelemetry import trace as trace_api

from app.config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_ENABLED, PHOENIX_PROJECT_NAME

logger = logging.getLogger(__name__)

# Usa el proveedor global: hasta que se llame a init_observability() es un no-op
tracer = OITracer(trace_api.get_tracer("app.orquestador"), config=TraceConfig())

_provider = None

__all__ = ["init_observability", "shutdown_observability", "tracer", "using_session"]


def init_observability(service: str = "worker") -> None:
    """Registra Phoenix como destino de las trazas e instrumenta LangChain/LangGraph."""
    global _provider
    if _provider is not None:
        return
    if not PHOENIX_ENABLED:
        logger.info("Observabilidad desactivada (PHOENIX_ENABLED=false).")
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_COLLECTOR_ENDPOINT,
        batch=True,  # exporta en segundo plano, sin frenar al agente
        set_global_tracer_provider=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    logger.info(
        "Trazas de %s enviadas a Phoenix (%s, proyecto %s).", service, PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME
    )


def shutdown_observability() -> None:
    """Vacía el buffer de spans antes de terminar el proceso."""
    global _provider
    if _provider is not None:
        _provider.force_flush()
        _provider.shutdown()
        _provider = None
