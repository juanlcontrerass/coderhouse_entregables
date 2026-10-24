"""Configuración compartida por todo el módulo.

Todos los scripts importan get_embeddings() y EMBEDDING_DIM de acá: así el índice
se crea con la misma dimensión que generan los embeddings (evita el "mismatch de
dimensiones") y la ingesta y la consulta usan siempre el mismo modelo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
GOLDEN_SET_PATH = BASE_DIR / "evaluation" / "golden_set.json"

INDEX_NAME = os.getenv("INDEX_NAME", "fastapi-rag-hibrido")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "fastapi-docs")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
METRIC = "cosine"

# text-embedding-3-small genera vectores de 1536 dimensiones: el índice DEBE coincidir
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Clave de metadata donde se guarda el texto original del chunk dentro de Pinecone
TEXT_KEY = "text"

CHUNK_SIZE = 600  # tokens (tiktoken), dentro del rango 500-800 sugerido
CHUNK_OVERLAP = 100
TOP_K = 5
# Pesos del EnsembleRetriever: [BM25, vectorial]
ENSEMBLE_WEIGHTS = (0.5, 0.5)


def require_env(name: str) -> str:
    """Devuelve una variable de entorno obligatoria o falla con un mensaje claro."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("tu_"):
        raise RuntimeError(f"Falta {name} en el archivo .env (ver .env.example)")
    return value


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
        api_key=require_env("OPENAI_API_KEY"),
    )
