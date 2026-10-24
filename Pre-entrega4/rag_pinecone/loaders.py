"""Carga de documentos técnicos en tres formatos: Markdown, PDF y JSON.

Cada loader devuelve Documents de LangChain con metadata homogénea:
    source / doc_id  nombre del archivo (es el "documento_id_esperado" del Golden Set)
    section          dónde está el fragmento dentro del archivo (full, página o id de entrada)
    page             número de página (1..n en PDFs, 0 si el formato no tiene páginas)
    category, tags   etiquetas para filtrar en Pinecone
    title            título legible del documento o entrada
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = (".md", ".pdf", ".json")


def _base_metadata(path: Path) -> dict:
    return {"source": path.name, "doc_id": path.name, "file_type": path.suffix.lstrip(".")}


def _parse_tags(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else raw.split(",")
    return [t.strip().lower() for t in items if t.strip()]


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Separa un bloque front-matter simple (clave: valor entre líneas '---') del cuerpo."""
    if not text.startswith("---"):
        return {}, text
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip()
    return meta, body.lstrip()


def load_markdown(path: Path) -> list[Document]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    metadata = _base_metadata(path) | {
        "section": "full",
        "page": 0,
        "title": meta.get("title", path.stem),
        "category": meta.get("category", "general"),
        "tags": _parse_tags(meta.get("tags")),
    }
    return [Document(page_content=body, metadata=metadata)]


def load_pdf(path: Path, category: str = "despliegue") -> list[Document]:
    """Un Document por página, para conservar el número de página en la metadata."""
    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata=_base_metadata(path)
                | {
                    "section": f"p{number}",
                    "page": number,
                    "title": title,
                    "category": category,
                    "tags": ["deployment", "docker", "uvicorn"],
                },
            )
        )
    return docs


def load_json(path: Path) -> list[Document]:
    """Un Document por entrada de la lista 'entries' (cada error es una unidad semántica)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    category = data.get("category", "general")
    return [
        Document(
            page_content=f"{entry['title']}\n\n{entry['content']}",
            metadata=_base_metadata(path)
            | {
                "section": entry["id"],
                "page": 0,
                "title": entry["title"],
                "category": category,
                "tags": _parse_tags(entry.get("tags")),
            },
        )
        for entry in data["entries"]
    ]


LOADERS = {".md": load_markdown, ".pdf": load_pdf, ".json": load_json}


def load_documents(data_dir: Path) -> list[Document]:
    """Carga todos los archivos soportados de data_dir, ordenados por nombre."""
    paths = sorted(p for p in data_dir.iterdir() if p.suffix in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No hay archivos {SUPPORTED_SUFFIXES} en {data_dir}")
    return [doc for p in paths for doc in LOADERS[p.suffix](p)]
