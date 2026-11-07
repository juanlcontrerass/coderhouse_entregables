"""Herramientas del Agente de Investigación: búsqueda sobre la Vector DB local (Chroma).

El vector store se crea una sola vez y se cachea. Los tests inyectan uno en memoria con
`set_vectorstore()`, así la búsqueda se prueba sin descargar el modelo de embeddings.

Cada consulta a la Vector DB abre un span OpenInference de tipo RETRIEVER (consulta, top-k,
fragmentos con id, fuente y score) y la indexación uno de tipo EMBEDDING; Phoenix los muestra
dentro de la llamada a la herramienta que los originó.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import DocumentAttributes, OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR
from observability import get_tracer, langchain_parent_context

SOURCE_PATTERN = re.compile(r"\[Fuente:\s*([^\]]+)\]")

tracer = get_tracer(__name__)

_store: VectorStore | None = None


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee los Markdown del corpus; la metadata `source` es el nombre del archivo."""
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(data_dir.glob("*.md"))
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """Corta por secciones Markdown y luego por tamaño, conservando el título de la sección."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo"), ("##", "seccion")], strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata["source"] = doc.metadata["source"]
            chunks.extend(size_splitter.split_documents([section]))
    return chunks


def build_chroma(rebuild: bool = False) -> VectorStore:
    """Abre (o crea) la colección persistente de Chroma e indexa el corpus si está vacía."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        persist_directory=str(VECTORSTORE_DIR),
    )
    if rebuild:
        store.reset_collection()
    if not store.get(limit=1)["ids"]:
        chunks = split_documents(load_documents())
        ids = [f"{c.metadata['source']}::c{i}" for i, c in enumerate(chunks)]
        with tracer.start_as_current_span("chroma.add_documents") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
            span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, EMBEDDING_MODEL)
            span.set_attribute("db.system", "chroma")
            span.set_attribute("db.collection.name", COLLECTION_NAME)
            span.set_attribute("db.chroma.chunk_count", len(chunks))
            store.add_documents(chunks, ids=ids)
    return store


def get_vectorstore() -> VectorStore:
    global _store
    if _store is None:
        _store = build_chroma()
    return _store


def set_vectorstore(store: VectorStore | None) -> None:
    global _store
    _store = store


def extract_sources(text: str) -> list[str]:
    """Nombres de archivo citados como [Fuente: archivo] (admite varios separados por coma)."""
    found: list[str] = []
    for group in SOURCE_PATTERN.findall(text):
        for name in group.split(","):
            name = name.strip()
            if name and name not in found:
                found.append(name)
    return found


def search_vectorstore(query: str, k: int) -> list[Document]:
    """Búsqueda por similitud trazada como span RETRIEVER de OpenInference."""
    store = get_vectorstore()
    with tracer.start_as_current_span("chroma.similarity_search", context=langchain_parent_context()) as span:
        span.set_attributes(dict(get_attributes_from_context()))  # session_id, tags y metadata de run_query
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, query)
        span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, EMBEDDING_MODEL)
        span.set_attribute("db.system", type(store).__name__)
        span.set_attribute("db.collection.name", COLLECTION_NAME)
        span.set_attribute("retrieval.top_k", k)
        try:
            # En Chroma el score es una distancia: cuanto menor, más parecido
            results = store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
        for i, (doc, score) in enumerate(results):
            span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}", str(doc.id or i))
            span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", doc.page_content)
            span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}", float(score))
            span.set_attribute(
                f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_METADATA}", json.dumps(doc.metadata, ensure_ascii=False)
            )
        span.set_attribute("retrieval.sources", sorted({d.metadata.get("source", "?") for d, _ in results}))
        span.set_status(Status(StatusCode.OK))
        return [doc for doc, _ in results]


@tool
def buscar_documentos(consulta: str, k: int = TOP_K) -> str:
    """Busca fragmentos relevantes en la base documental interna de TechCorp (políticas de
    vacaciones, teletrabajo, seguridad informática, onboarding y encuesta de clima 2025).
    Devuelve cada fragmento precedido por [Fuente: archivo]."""
    docs = search_vectorstore(consulta, k=max(1, min(k, 8)))
    if not docs:
        return "Sin resultados para esa búsqueda."
    return "\n\n---\n\n".join(f"[Fuente: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


@tool
def listar_documentos() -> str:
    """Lista los documentos disponibles en la base documental."""
    return "\n".join(f"- {d.metadata['source']}" for d in load_documents())


RESEARCH_TOOLS = [buscar_documentos, listar_documentos]
