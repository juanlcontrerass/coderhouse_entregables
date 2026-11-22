"""Worker de Arq: ejecuta el grafo en segundo plano y mantiene el estado del job en Redis.

Arranque:
    arq app.worker.worker.WorkerSettings

La API solo encola (`run_task` o `resume_task`) y responde al instante; este proceso toma los
jobs de la cola, corre el orquestador sobre el thread de la conversación y escribe cada
transición de estado. Cualquier excepción (incluido el timeout) deja el job en FAILED con el
error, así el cliente que hace polling nunca queda esperando para siempre.
"""

import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal

from arq.connections import RedisSettings
from langgraph.types import Command
from redis.asyncio import Redis

from app.api.schemas import ApprovalRequest, DecisionOut, TaskResult
from app.core.config import get_settings
from app.core.logging import log_context, setup_logging, update_log_context
from app.graph.turn import graph_config, turn_input
from app.observability.tracing import current_trace_id, init_observability, shutdown_observability, tracer, using_session
from app.storage.conversations import ConversationStore
from app.storage.jobs import TERMINAL_STATUSES, JobStatus, JobStore, now_iso

logger = logging.getLogger("app.worker")

if not hasattr(signal, "SIGUSR1"):
    # Windows no define SIGUSR1 y arq lo usa al cerrar el worker cuando handle_signals=False;
    # sin este alias el cierre falla antes de ejecutar on_shutdown (y se pierden spans).
    signal.SIGUSR1 = signal.SIGTERM  # type: ignore[attr-defined,misc]


@dataclass
class Runtime:
    graph: Any
    store: JobStore
    conversations: ConversationStore


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    """Fija el grafo y los stores que usan los jobs (los tests inyectan los suyos)."""
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("El worker no está inicializado: falta ejecutar startup().")
    return _runtime


def summarize(values: dict[str, Any]) -> TaskResult:
    """Resultado validado que se guarda en Redis y devuelve GET /tasks/{id}."""
    decisions = [DecisionOut.model_validate(d) for d in values.get("decisions") or []]
    sources = [s for c in values.get("contributions") or [] if c["agent"] == "researcher" for s in c["sources"]]
    return TaskResult(
        final_answer=values.get("final_answer", ""),
        route=[d.next_agent for d in decisions],
        decisions=decisions,
        sources=list(dict.fromkeys(sources)),
        steps=values.get("step_count", 0),
        tokens_used=values.get("tokens_used", 0),
        total_tokens=values.get("total_tokens", 0),
        task_completed=values.get("task_completed", False),
        risk_reasons=values.get("risk_reasons") or [],
        approval=values.get("approval") or None,
        published_path=values.get("published_path") or None,
    )


@tracer.chain(name="job_turno")
async def process_job(job_id: str, mode: Literal["run", "resume"], decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta (o reanuda) el turno de un job y deja el estado final en Redis.

    El decorador crea el span raíz en Phoenix; los spans de cada nodo y llamada al LLM cuelgan de él.
    """
    runtime = get_runtime()
    store, graph = runtime.store, runtime.graph
    job = await store.get(job_id)
    if job is None:
        logger.warning("Job %s inexistente o expirado; se descarta.", job_id)
        return {"status": "DESCARTADO"}

    conversation_id = job["conversation_id"]
    trace_id = current_trace_id()
    update_log_context(trace_id=trace_id)
    config = graph_config(conversation_id, job_id, mode=mode)
    config["tags"] = ["api", mode]
    graph_input: Any = turn_input(job["query"], job.get("publish", False)) if mode == "run" else Command(resume=decision or {})
    await store.update(job_id, status=JobStatus.RUNNING, started_at=now_iso(), approval_request=None, trace_id=trace_id)

    try:
        async with asyncio.timeout(get_settings().job_timeout_seconds):
            await graph.ainvoke(graph_input, config)
        snapshot = await graph.aget_state(config)
    except TimeoutError:
        error = f"TimeoutError: la ejecución superó {get_settings().job_timeout_seconds} s."
        logger.error("Job %s: %s", job_id, error)
        return await _finish(runtime, job_id, conversation_id, JobStatus.FAILED, error=error)
    except Exception as exc:  # cualquier fallo del agente termina en FAILED, nunca en silencio
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Job %s falló", job_id)
        return await _finish(runtime, job_id, conversation_id, JobStatus.FAILED, error=error)

    if snapshot.interrupts:
        request = ApprovalRequest.model_validate(snapshot.interrupts[0].value)
        await store.update(job_id, status=JobStatus.WAITING_APPROVAL, approval_request=request.model_dump())
        logger.info("Job %s en pausa: espera aprobación humana.", job_id)
        return {"status": JobStatus.WAITING_APPROVAL, "approval_request": request.model_dump()}

    result = summarize(snapshot.values)
    final_status = JobStatus.REJECTED if result.approval and not result.approval.approved else JobStatus.DONE
    logger.info("Job %s terminó: %s (%s tokens).", job_id, final_status, result.tokens_used)
    return await _finish(runtime, job_id, conversation_id, final_status, result=result)


async def _finish(
    runtime: Runtime,
    job_id: str,
    conversation_id: str,
    status: JobStatus,
    *,
    result: TaskResult | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Estado terminal del job y liberación del turno de la conversación."""
    assert status in TERMINAL_STATUSES
    fields: dict[str, Any] = {"status": status, "finished_at": now_iso()}
    if result is not None:
        fields["result"] = result.model_dump(mode="json")
    if error is not None:
        fields["error"] = error
    try:
        await runtime.store.update(job_id, **fields)
        await runtime.conversations.release(conversation_id, job_id)
    except Exception:  # si Redis tampoco responde, al menos queda en el log
        logger.exception("No se pudo registrar el estado final del job %s", job_id)
    outcome: dict[str, Any] = {"status": status}
    if result is not None:
        outcome["tokens_used"] = result.tokens_used
    if error is not None:
        outcome["error"] = error
    return outcome


# --- Funciones que encola la API ---


async def _session_for(job_id: str) -> str:
    job = await get_runtime().store.get(job_id)
    return job["conversation_id"] if job else job_id


async def run_task(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    conversation_id = await _session_for(job_id)
    with log_context(job_id=job_id, conversation_id=conversation_id), using_session(conversation_id):
        return await process_job(job_id, "run")


async def resume_task(ctx: dict[str, Any], job_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    conversation_id = await _session_for(job_id)
    with log_context(job_id=job_id, conversation_id=conversation_id), using_session(conversation_id):
        return await process_job(job_id, "resume", decision)


# --- Ciclo de vida del worker ---


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging()
    init_observability("worker")
    settings = get_settings()

    from app.graph.builder import build_graph
    from app.llm.client import get_llm
    from app.rag.retriever import get_retriever
    from app.storage.checkpointer import redis_checkpointer

    stack = AsyncExitStack()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    stack.push_async_callback(redis.aclose)
    checkpointer = await stack.enter_async_context(redis_checkpointer())
    store = JobStore(redis)
    set_runtime(
        Runtime(
            graph=build_graph(get_llm(), checkpointer=checkpointer),
            store=store,
            conversations=ConversationStore(redis, store),
        )
    )
    ctx["exit_stack"] = stack
    # Carga el modelo de embeddings y el índice una sola vez (en un hilo, fuera del event loop)
    await get_retriever()
    logger.info("Worker listo (max_jobs=%s, modelo=%s).", settings.worker_max_jobs, settings.openai_model)


async def shutdown(ctx: dict[str, Any]) -> None:
    stack: AsyncExitStack | None = ctx.get("exit_stack")
    if stack is not None:
        await stack.aclose()
    set_runtime(None)
    shutdown_observability()


class WorkerSettings:
    functions = [run_task, resume_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = get_settings().worker_max_jobs
    max_tries = 1  # un job fallido queda en FAILED; no se reintenta a ciegas
    job_timeout = int(get_settings().job_timeout_seconds) + 30  # red de seguridad; el timeout propio va antes
    keep_result = 3600
    # En Windows el event loop no admite add_signal_handler
    handle_signals = sys.platform != "win32"
