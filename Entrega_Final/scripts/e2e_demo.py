"""Las 5 pruebas de punta a punta para la evidencia de observabilidad.

Uso (con el sistema levantado por `./run.sh` o `docker compose up -d --build`):
    python scripts/e2e_demo.py
    python scripts/e2e_demo.py --no-restart     # si la API y el worker no corren en Docker
    python scripts/e2e_demo.py --base-url http://localhost:8000 --timeout 600

Pruebas:
    1. RAG simple: requisitos de contraseñas (investigador).
    2. RAG + cálculo: días de teletrabajo y porcentaje de la semana (investigador + analista).
    3. RAG + sentimiento: comentarios de la encuesta de clima (investigador + analista).
    4. Conversación de 2 turnos con `docker compose restart api worker` entre turnos: la
       repregunta se responde con el historial que guardó el checkpointer en Redis.
    5. Tarea crítica: pide publicar el informe → WAITING_APPROVAL → aprobación → DONE.

Para cada prueba imprime job_id, conversation_id, trace_id, estado, latencia y tokens. Con el
trace_id se ubica la traza en Phoenix (http://localhost:6006).
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"DONE", "REJECTED", "FAILED"}


@dataclass
class Outcome:
    test: str
    job_id: str
    conversation_id: str
    status: str
    seconds: float
    tokens: int | None
    trace_id: str | None
    answer: str = ""
    error: str = ""
    extra: dict[str, str] = field(default_factory=dict)


async def wait_for(client: httpx.AsyncClient, job_id: str, timeout: float, approve: bool = False) -> dict:
    """Polling con backoff hasta un estado terminal; aprueba si se pide y el job está en pausa."""
    start, delay, approved = time.perf_counter(), 0.5, False
    while time.perf_counter() - start < timeout:
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 5.0)
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] == "WAITING_APPROVAL":
            if not approve:
                return job
            if not approved:
                print(f"    {job_id} espera aprobación: {job['approval_request']['motivos']}")
                response = await client.post(
                    f"/tasks/{job_id}/approve", json={"approved": True, "comment": "Aprobado en la demo e2e"}
                )
                response.raise_for_status()
                approved, delay = True, 0.5
        elif job["status"] in TERMINAL:
            return job
    raise TimeoutError(f"El job {job_id} no terminó en {timeout} s")


def outcome(test: str, job: dict, seconds: float, **extra: str) -> Outcome:
    result = job.get("result") or {}
    return Outcome(
        test=test,
        job_id=job["job_id"],
        conversation_id=job["conversation_id"],
        status=job["status"],
        seconds=seconds,
        tokens=result.get("tokens_used"),
        trace_id=job.get("trace_id"),
        answer=result.get("final_answer", ""),
        error=job.get("error") or "",
        extra=extra,
    )


async def single_turn(client: httpx.AsyncClient, test: str, query: str, timeout: float, publish: bool = False) -> Outcome:
    start = time.perf_counter()
    response = await client.post("/tasks", json={"query": query, "publish": publish})
    response.raise_for_status()
    job = await wait_for(client, response.json()["job_id"], timeout, approve=publish)
    return outcome(test, job, time.perf_counter() - start)


async def restart_services() -> None:
    print("    reiniciando api y worker (docker compose restart api worker) ...")
    process = await asyncio.create_subprocess_exec("docker", "compose", "restart", "api", "worker", cwd=ROOT)
    if await process.wait() != 0:
        raise RuntimeError("docker compose restart falló")


async def wait_healthy(base_url: str, timeout: float = 120) -> None:
    start = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
        while time.perf_counter() - start < timeout:
            try:
                if (await client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError("La API no volvió a responder después del reinicio")


async def conversation_with_restart(base_url: str, timeout: float, restart: bool) -> list[Outcome]:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        conversation = (await client.post("/conversations")).json()
        url = conversation["messages_url"]
        start = time.perf_counter()
        first = await client.post(url, json={"query": "¿Cuántos días de vacaciones corresponden con 3 años de antigüedad?"})
        first.raise_for_status()
        job1 = await wait_for(client, first.json()["job_id"], timeout)
        turn1 = outcome("4a. conversación, turno 1", job1, time.perf_counter() - start)

    if restart:
        await restart_services()
        await wait_healthy(base_url)

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        start = time.perf_counter()
        second = await client.post(url, json={"query": "¿Y con 12 años? ¿En qué porcentaje supera al caso anterior?"})
        second.raise_for_status()
        job2 = await wait_for(client, second.json()["job_id"], timeout)
        turn2 = outcome("4b. conversación, turno 2 (tras reinicio)" if restart else "4b. conversación, turno 2", job2,
                        time.perf_counter() - start)
        history = (await client.get(f"/conversations/{conversation['conversation_id']}")).json()
        turn2.extra["turnos_en_historial"] = str(len(history["turns"]))
    return [turn1, turn2]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600, help="segundos máximos por job")
    parser.add_argument("--no-restart", action="store_true", help="no reinicia los contenedores en la prueba 4")
    args = parser.parse_args()

    await wait_healthy(args.base_url, timeout=30)
    started = time.perf_counter()
    print(f"Pruebas 1, 2, 3 y 5 en paralelo contra {args.base_url} ...")
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        parallel = await asyncio.gather(
            single_turn(client, "1. RAG simple",
                        "¿Qué requisitos de contraseñas exige la política de seguridad informática y cada cuánto deben cambiarse?",
                        args.timeout),
            single_turn(client, "2. RAG + cálculo",
                        "¿Cuántos días de teletrabajo permite la política y qué porcentaje de la semana laboral representan?",
                        args.timeout),
            single_turn(client, "3. RAG + sentimiento",
                        "Busca los comentarios abiertos de la encuesta de clima 2025 y analiza su sentimiento.", args.timeout),
            single_turn(client, "5. HITL: publicar informe",
                        "Calcula la variación porcentual del eNPS 2025 respecto de 2024 y publica el informe para RR. HH.",
                        args.timeout, publish=True),
        )
    print("Prueba 4: conversación de dos turnos ...")
    conversation = await conversation_with_restart(args.base_url, args.timeout, restart=not args.no_restart)
    outcomes = sorted([*parallel, *conversation], key=lambda o: o.test)

    print(f"\n{'prueba':44} {'estado':9} {'latencia':>9} {'tokens':>7}  trace_id")
    for o in outcomes:
        tokens = "-" if o.tokens is None else str(o.tokens)
        print(f"{o.test:44} {o.status:9} {o.seconds:8.1f}s {tokens:>7}  {o.trace_id or '-'}")
        print(f"    job_id={o.job_id} conversation_id={o.conversation_id} {' '.join(f'{k}={v}' for k, v in o.extra.items())}")
        if o.error:
            print(f"    error: {o.error}")
        elif o.answer:
            print(f"    respuesta: {o.answer[:160].replace(chr(10), ' ')}...")
    print(f"\nTiempo total: {time.perf_counter() - started:.1f} s")
    print("Trazas: http://localhost:6006 (proyecto intelligence-system). La prueba 4 se ve agrupada en Sessions.")
    if any(o.status != "DONE" for o in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos en la consola de Windows
    asyncio.run(main())
