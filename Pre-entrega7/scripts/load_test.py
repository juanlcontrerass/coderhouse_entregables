"""Prueba de carga: 5 peticiones concurrentes contra la API.

Uso:
    python scripts/load_test.py                    # 5 consultas; la crítica queda en pausa
    python scripts/load_test.py --auto-approve     # aprueba la tarea crítica y espera su fin
    python scripts/load_test.py --base-url http://localhost:8000 --timeout 600

Mide la latencia extremo a extremo de cada job (desde el POST hasta un estado terminal) y
muestra p50/p95 del lado del cliente. El costo por ejecución y la latencia p95 de referencia
se leen en el dashboard de Phoenix (http://localhost:6006).
"""

import argparse
import asyncio
import math
import sys
import time
from dataclasses import dataclass

import httpx

QUERIES: list[tuple[str, bool]] = [
    ("¿Cuántos días de vacaciones corresponden con 3, 7 y 12 años de antigüedad? "
     "Calcula el promedio y en qué porcentaje supera el de 12 años al de 3.", False),
    ("¿Cuántos días de teletrabajo permite la política y qué porcentaje de la semana laboral representan?", False),
    ("Busca los comentarios abiertos de la encuesta de clima 2025 y analiza su sentimiento.", False),
    ("¿Qué requisitos de contraseñas exige la política de seguridad informática y cada cuánto deben cambiarse?", False),
    # Tarea crítica: pide una acción con efecto lateral → pausa para aprobación humana
    ("Calcula la variación porcentual del eNPS 2025 respecto de 2024 y publica el informe para RR. HH.", True),
]
TERMINAL = {"DONE", "REJECTED", "FAILED"}


@dataclass
class Outcome:
    job_id: str
    status: str
    seconds: float
    tokens: int | None
    critical: bool
    error: str = ""


async def run_one(
    client: httpx.AsyncClient, query: str, publish: bool, auto_approve: bool, timeout: float
) -> Outcome:
    start = time.perf_counter()
    response = await client.post("/tasks", json={"query": query, "publish": publish})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"  encolado {job_id} ({time.perf_counter() - start:.3f} s){' [crítica]' if publish else ''}")

    delay, approved = 0.5, False
    while time.perf_counter() - start < timeout:
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 5.0)  # backoff para no saturar la API
        job = (await client.get(f"/tasks/{job_id}")).json()
        state = job["status"]
        if state == "WAITING_APPROVAL":
            if not auto_approve:
                return Outcome(job_id, state, time.perf_counter() - start, None, publish)
            if not approved:
                print(f"  {job_id} espera aprobación → se aprueba automáticamente")
                await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "prueba de carga"})
                approved, delay = True, 0.5
        elif state in TERMINAL:
            result = job.get("result") or {}
            return Outcome(job_id, state, time.perf_counter() - start, result.get("tokens_used"), publish, job.get("error") or "")
    return Outcome(job_id, "TIMEOUT_CLIENTE", time.perf_counter() - start, None, publish)


def percentile(values: list[float], p: float) -> float:
    """Percentil por el método del rango más cercano."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--auto-approve", action="store_true", help="aprueba automáticamente la tarea crítica")
    parser.add_argument("--timeout", type=float, default=600, help="segundos máximos por job")
    args = parser.parse_args()

    print(f"Enviando {len(QUERIES)} peticiones concurrentes a {args.base_url} ...")
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        outcomes = await asyncio.gather(
            *(run_one(client, q, publish, args.auto_approve, args.timeout) for q, publish in QUERIES)
        )
    total = time.perf_counter() - started

    print(f"\n{'job_id':34} {'estado':18} {'latencia':>9} {'tokens':>8}")
    for o in outcomes:
        tokens = "-" if o.tokens is None else str(o.tokens)
        print(f"{o.job_id:34} {o.status:18} {o.seconds:8.1f}s {tokens:>8}{'  [crítica]' if o.critical else ''}")
        if o.error:
            print(f"    error: {o.error}")

    finished = [o.seconds for o in outcomes if o.status in TERMINAL]
    print(f"\nTiempo total de la corrida: {total:.1f} s")
    if finished:
        print(f"Latencia extremo a extremo (cliente, {len(finished)} jobs terminados): "
              f"p50 = {percentile(finished, 50):.1f} s · p95 = {percentile(finished, 95):.1f} s")
    print("Costo por ejecución y latencia p95 por traza: http://localhost:6006 (proyecto orquestador-multiagente).")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos en la consola de Windows
    asyncio.run(main())
