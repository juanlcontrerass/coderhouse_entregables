"""Inicialización de infraestructura: verifica si el índice existe y lo crea (Serverless) si hace falta.

Uso:
    python setup_index.py
"""

from rag_pinecone.config import INDEX_NAME
from rag_pinecone.index import ensure_index

if __name__ == "__main__":
    index = ensure_index(index_name=INDEX_NAME)
    print(f"✅ Índice '{INDEX_NAME}' listo")
    print(index.describe_index_stats())
