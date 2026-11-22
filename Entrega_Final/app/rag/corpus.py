"""Carga y fragmentación del corpus de TechCorp (Markdown).

Se corta primero por secciones Markdown y luego por tamaño, conservando el título y la sección
en la metadata. Cada fragmento recibe un `chunk_id` determinista (`archivo::cN`): es la clave
que comparten el índice denso (Chroma) y el léxico (BM25) para fusionar resultados.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import get_settings


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """Lee los Markdown del corpus; la metadata `source` es el nombre del archivo."""
    directory = data_dir or get_settings().data_dir
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(directory.glob("*.md"))
    ]


def split_documents(docs: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
    settings = get_settings()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo"), ("##", "seccion")], strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        source = doc.metadata["source"]
        pieces: list[Document] = []
        for section in header_splitter.split_text(doc.page_content):
            section.metadata["source"] = source
            pieces.extend(size_splitter.split_documents([section]))
        for i, piece in enumerate(pieces):
            piece.metadata["chunk_id"] = f"{source}::c{i}"
        chunks.extend(pieces)
    return chunks
