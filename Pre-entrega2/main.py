"""Mini-script de prueba asíncrono del pipeline de extracción de entidades técnicas."""

import asyncio
import logging
import os
import sys

from chain import process_text

CASOS = {
    "Arquitectura": (
        "El backend es una API en FastAPI que cachea las consultas frecuentes en Redis y persiste en PostgreSQL. "
        "Bajo carga, el pool de conexiones se agota con más de 500 usuarios concurrentes y las peticiones "
        "empiezan a devolver 503 en producción."
    ),
    "Log de error": (
        "2026-10-09 03:14:07 ERROR [worker-3] celery.app.trace: Task orders.sync_inventory raised "
        "ConnectionResetError: [Errno 104] Connection reset by peer. Retrying in 30s (attempt 2/5). "
        "Broker: amqp://rabbitmq:5672//"
    ),
    # Prueba de estrés: no nombra ninguna tecnología, así que la lista podría venir vacía
    "Texto ambiguo": "Ayer el sistema anduvo raro un rato, pero después se arregló solo. Nadie sabe bien qué pasó.",
}


async def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        sys.exit("⚠️  Falta HF_TOKEN en el .env")

    for nombre, texto in CASOS.items():
        print(f"\n=== {nombre} ===")
        try:
            resultado = await process_text(texto)
        except Exception as e:
            print(f"❌ El pipeline no obtuvo un objeto válido: {type(e).__name__}: {e}")
        else:
            print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no usa UTF-8 por defecto
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main())
