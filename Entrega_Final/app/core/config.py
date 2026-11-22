"""Configuración centralizada del sistema.

Todo valor que cambia entre entornos (modelo, embeddings, colección, URLs, límites) se lee de
variables de entorno o del archivo `.env`; ningún módulo los escribe en el código. Los valores
obligatorios no tienen default: si faltan, el proceso falla al arrancar con un mensaje claro.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenAI, cliente asíncrono) ---
    openai_api_key: SecretStr | None = Field(default=None, description="Obligatoria solo en el worker")
    openai_model: str = Field(min_length=1, description="Modelo de chat que usan todos los agentes")
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- RAG: corpus, Chroma y búsqueda híbrida ---
    data_dir: Path = APP_DIR / "data"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    chroma_collection: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    chunk_size: int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=4, ge=1, le=8)
    retrieval_candidates: int = Field(default=8, ge=1, le=50, description="Candidatos por retriever antes de fusionar")
    hybrid_bm25_weight: float = Field(default=0.5, ge=0)
    hybrid_dense_weight: float = Field(default=0.5, ge=0)
    rrf_k: int = Field(default=60, ge=1, description="Constante de Reciprocal Rank Fusion")

    # --- Condiciones de parada del orquestador ---
    max_steps: int = Field(default=8, ge=1)
    max_revisions: int = Field(default=2, ge=0)
    max_validation_retries: int = Field(default=1, ge=0)
    recursion_limit: int = Field(default=50, ge=10)
    agent_recursion_limit: int = Field(default=12, ge=4)
    history_turns: int = Field(default=3, ge=0, description="Turnos previos que ven el supervisor y el sintetizador")

    # --- Infraestructura ---
    redis_url: str = "redis://localhost:6379/0"
    job_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    conversation_ttl_minutes: int = Field(default=7 * 24 * 60, gt=0, description="TTL de los checkpoints")
    job_timeout_seconds: float = Field(default=300, gt=0)
    worker_max_jobs: int = Field(default=10, ge=1)
    reports_dir: Path = BASE_DIR / "reports"

    # --- Human-in-the-loop ---
    cost_approval_token_threshold: int = Field(default=60000, gt=0)

    # --- Observabilidad (Arize Phoenix) ---
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "intelligence-system"

    @model_validator(mode="after")
    def _check_weights(self) -> "Settings":
        if self.hybrid_bm25_weight + self.hybrid_dense_weight <= 0:
            raise ValueError("HYBRID_BM25_WEIGHT y HYBRID_DENSE_WEIGHT no pueden ser ambos 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        return self

    def require_openai_key(self) -> SecretStr:
        key = self.openai_api_key
        if key is None or not key.get_secret_value().strip() or key.get_secret_value().startswith("tu_"):
            raise RuntimeError("Falta OPENAI_API_KEY en el archivo .env (ver .env.example).")
        return key


@lru_cache
def get_settings() -> Settings:
    """Instancia única; los tests modifican sus atributos con monkeypatch."""
    return Settings()  # type: ignore[call-arg]  # los obligatorios vienen del entorno
