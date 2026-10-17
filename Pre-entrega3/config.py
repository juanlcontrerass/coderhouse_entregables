"""Configuración compartida entre la ingesta y la cadena RAG.

Ambos scripts importan get_embeddings() de acá: así es imposible indexar con un
modelo de embeddings y consultar con otro (el error #1 de la consigna).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "techcorp_policies"

# Modelo multilingüe: los documentos y las preguntas están en español
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

CHUNK_SIZE = 500  # tokens (tiktoken), no caracteres
CHUNK_OVERLAP = 50
TOP_K = 4  # entre 3 y 5 para evitar el "contexto infinito"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
