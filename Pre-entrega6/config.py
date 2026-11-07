"""Configuración compartida: variables de entorno, rutas y límites del orquestador."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
TRACES_DIR = BASE_DIR / "traces"

DEFAULT_MODEL = "gpt-5.4-mini"

# Vector DB local (Chroma) con embeddings multilingües que corren en la CPU
COLLECTION_NAME = "techcorp_docs"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = 1000  # caracteres
CHUNK_OVERLAP = 100
TOP_K = 4

# --- Observabilidad: exportador OpenInference hacia el servidor local de Phoenix ---
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-techcorp")
TRACING_ENABLED = os.getenv("PHOENIX_TRACING", "true").strip().lower() not in ("0", "false", "no")

# --- Condiciones de parada (evitan el "supervisor infinito") ---
# Decisiones máximas del supervisor por consulta; al alcanzarlas se fuerza la síntesis
MAX_STEPS = 8
# Veces que un especialista puede ser re-delegado para refinar su resultado
MAX_REVISIONS = 2
# Veces que el validador puede devolver la respuesta al supervisor
MAX_VALIDATION_RETRIES = 1
# Techo de pasos de LangGraph: del grafo principal y de cada agente ReAct interno
RECURSION_LIMIT = 50
AGENT_RECURSION_LIMIT = 12


def require_env(name: str) -> str:
    """Devuelve una variable de entorno obligatoria o falla con un mensaje claro."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("tu_"):
        raise RuntimeError(f"Falta {name} en el archivo .env (ver .env.example)")
    return value
