"""Configuración compartida por todo el paquete."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CHECKPOINT_DB = BASE_DIR / "checkpoints.sqlite"
TRACES_DIR = BASE_DIR / "traces"

DEFAULT_MODEL = "gpt-5.4-mini"
# Techo de pasos del grafo por turno (cada ida y vuelta modelo → herramienta consume 2)
RECURSION_LIMIT = 10
# Tope aproximado del historial que se envía al modelo en cada llamada
MAX_CONTEXT_TOKENS = 4000


def require_env(name: str) -> str:
    """Devuelve una variable de entorno obligatoria o falla con un mensaje claro."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("tu_"):
        raise RuntimeError(f"Falta {name} en el archivo .env (ver .env.example)")
    return value
