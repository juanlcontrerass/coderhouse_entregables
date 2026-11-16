"""Configuración compartida: variables de entorno, rutas y límites del orquestador y la API."""

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = APP_DIR / "data"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))

DEFAULT_MODEL = "gpt-5.4-mini"

# Vector DB local (Chroma) con embeddings multilingües que corren en la CPU
COLLECTION_NAME = "techcorp_docs"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = 1000  # caracteres
CHUNK_OVERLAP = 100
TOP_K = 4

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

# --- Infraestructura de producción ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Segundos que se conserva el estado de un job en Redis (por defecto, 7 días)
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(7 * 24 * 3600)))
# Tiempo máximo de una ejecución del grafo antes de marcar el job como FAILED
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
# Jobs que un worker ejecuta en paralelo
WORKER_MAX_JOBS = int(os.getenv("WORKER_MAX_JOBS", "10"))

# --- Human-in-the-loop ---
# Si una ejecución consume más tokens que este umbral, se pide aprobación humana
COST_APPROVAL_TOKEN_THRESHOLD = int(os.getenv("COST_APPROVAL_TOKEN_THRESHOLD", "60000"))

# --- Observabilidad (Arize Phoenix) ---
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").strip().lower() in ("1", "true", "yes", "si")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-multiagente")


def require_env(name: str) -> str:
    """Devuelve una variable de entorno obligatoria o falla con un mensaje claro."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("tu_"):
        raise RuntimeError(f"Falta {name} en el archivo .env (ver .env.example)")
    return value
