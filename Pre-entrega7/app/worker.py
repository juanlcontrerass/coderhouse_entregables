"""Worker de Arq: ejecuta el grafo en segundo plano y mantiene el estado del job en Redis.

Arranque:
    arq app.worker.WorkerSettings

La API solo encola (`run_task` o `resume_task`) y responde al instante; este proceso toma los
jobs de la cola de Redis, corre el orquestador y escribe cada transición de estado. Cualquier
excepción (incluido el timeout) deja el job en FAILED con el error, así el cliente que hace
polling nunca queda esperando para siempre.
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

from app.config import JOB_TIMEOUT_SECONDS, RECURSION_LIMIT, REDIS_URL, WORKER_MAX_JOBS
from app.jobs import JobStatus, JobStore, now_iso
from app.observability import init_observability, shutdown_observability, tracer, using_session
from app.runner import initial_state

logger = logging.getLogger("app.worker")

if not hasattr(signal, "SIGUSR1"):
    # Windows no define SIGUSR1 y arq lo usa al cerrar el worker cuando handle_signals=False;
    # sin este alias el cierre falla antes de ejecutar on_shutdown (y se pierden spans).
    signal.SIGUSR1 = signal.SIGTERM  # type: ignore[attr-defined,misc]


@dataclass
class Runtime:
    graph: Any
    store: JobStore


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    """Fija el grafo y el store que usan los jobs (los tests inyectan los suyos)."""
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("El worker no está inicializado: falta ejecutar startup().")
    return _runtime


def summarize(values: dict[str, Any]) -> dict[str, Any]:
    """Resultado que se guarda en Redis y devuelve GET /tasks/{id}."""
    return {
        "final_answer": values.get("final_answer", ""),
        "route": [d["next_agent"] for d in values.get("decisions", [])],
        "steps": values.get("step_count", 0),
        "tokens_used": values.get("tokens_used", 0),
        "task_completed": values.get("task_completed", False),
        "risk_reasons": values.get("risk_reasons", []),
        "approval": values.get("approval") or None,
        "published_path": values.get("published_path") or None,
    }


@tracer.chain(name="job_orquestador")
async def process_job(job_id: str, mode: Literal["run", "resume"], decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ejecuta (o reanuda) el grafo de un job y deja el estado final en Redis.

    El decorador crea el span raíz del job en Phoenix; los spans de cada nodo y llamada al
    LLM cuelgan de él.
    """
    runtime = get_runtime()
    store, graph = runtime.store, runtime.graph
    job = await store.get(job_id)
    if job is None:
        logger.warning("Job %s inexistente o expirado; se descarta.", job_id)
        return {"status": "DESCARTADO"}

    config: dict[str, Any] = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": RECURSION_LIMIT,
        "metadata": {"job_id": job_id, "mode": mode},
        "tags": ["api", mode],
    }
    graph_input: Any = (
        initial_state(job["query"], job.get("publish", False)) if mode == "run" else Command(resume=decision or {})
    )
    await store.update(job_id, status=JobStatus.RUNNING, started_at=now_iso(), approval_request=None)
    try:
        async with asyncio.timeout(JOB_TIMEOUT_SECONDS):
            await graph.ainvoke(graph_input, config)
        snapshot = await graph.aget_state(config)
    except TimeoutError:
        error = f"TimeoutError: la ejecución superó {JOB_TIMEOUT_SECONDS} s."
        logger.error("Job %s: %s", job_id, error)
        await _mark_failed(store, job_id, error)
        return {"status": JobStatus.FAILED, "error": error}
    except Exception as exc:  # cualquier fallo del agente termina en FAILED, nunca en silencio
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Job %s falló", job_id)
        await _mark_failed(store, job_id, error)
        return {"status": JobStatus.FAILED, "error": error}

    if snapshot.interrupts:
        request = snapshot.interrupts[0].value
        await store.update(job_id, status=JobStatus.WAITING_APPROVAL, approval_request=request)
        logger.info("Job %s en pausa: espera aprobación humana.", job_id)
        return {"status": JobStatus.WAITING_APPROVAL, "approval_request": request}

    result = summarize(snapshot.values)
    approval = result["approval"]
    final_status = JobStatus.REJECTED if approval and not approval.get("approved") else JobStatus.DONE
    await store.update(job_id, status=final_status, result=result, finished_at=now_iso())
    logger.info("Job %s terminó: %s (%s tokens).", job_id, final_status, result["tokens_used"])
    return {"status": final_status, "tokens_used": result["tokens_used"]}


async def _mark_failed(store: JobStore, job_id: str, error: str) -> None:
    try:
        await store.update(job_id, status=JobStatus.FAILED, error=error, finished_at=now_iso())
    except Exception:  # si Redis tampoco responde, al menos queda en el log
        logger.exception("No se pudo marcar el job %s como FAILED", job_id)


# --- Funciones que encola la API ---


async def run_task(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    with using_session(job_id):
        return await process_job(job_id, "run")


async def resume_task(ctx: dict[str, Any], job_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    with using_session(job_id):
        return await process_job(job_id, "resume", decision)


# --- Ciclo de vida del worker ---


async def startup(ctx: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability("worker")

    from app.graph import build_graph, redis_checkpointer
    from app.llm import get_llm
    from app.tools.retrieval import get_vectorstore

    stack = AsyncExitStack()
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    stack.push_async_callback(redis.aclose)
    checkpointer = await stack.enter_async_context(redis_checkpointer(REDIS_URL))
    set_runtime(Runtime(graph=build_graph(get_llm(), checkpointer=checkpointer), store=JobStore(redis)))
    ctx["exit_stack"] = stack
    # Carga Chroma y el modelo de embeddings una sola vez, fuera del event loop
    await asyncio.to_thread(get_vectorstore)
    logger.info("Worker listo (max_jobs=%s).", WORKER_MAX_JOBS)


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
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = WORKER_MAX_JOBS  # las 5 peticiones de la prueba de carga corren en paralelo
    max_tries = 1  # un job fallido queda en FAILED; no se reintenta a ciegas
    job_timeout = JOB_TIMEOUT_SECONDS + 30  # red de seguridad; el timeout propio va antes
    keep_result = 3600
    # En Windows el event loop no admite add_signal_handler
    handle_signals = sys.platform != "win32"
