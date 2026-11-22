"""Herramientas del Agente de Investigación: el RAG híbrido como herramienta asíncrona."""

import re

from langchain_core.tools import tool
from pydantic import ValidationError

from app.rag.corpus import load_documents
from app.rag.retriever import get_retriever
from app.tools.schemas import (
    BuscarDocumentosInput,
    Fragmento,
    ListaDocumentos,
    ResultadoBusqueda,
)

SOURCE_PATTERN = re.compile(r"\[Fuente:\s*([^\]]+)\]")


def validation_message(error: ValidationError) -> str:
    """Error de validación legible para el LLM, que así puede corregir la llamada."""
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return f"Error: argumentos inválidos ({details}). Corrige la llamada y vuelve a intentarlo."


def extract_sources(text: str) -> list[str]:
    """Nombres de archivo citados como [Fuente: archivo] (admite varios separados por coma)."""
    found: list[str] = []
    for group in SOURCE_PATTERN.findall(text):
        for name in group.split(","):
            name = name.strip()
            if name and name not in found:
                found.append(name)
    return found


@tool(args_schema=BuscarDocumentosInput, response_format="content_and_artifact")
async def buscar_documentos(consulta: str, k: int = 4) -> tuple[str, dict[str, object]]:
    """Busca fragmentos relevantes en la base documental interna de TechCorp (políticas de
    vacaciones, teletrabajo, seguridad informática, onboarding y encuesta de clima 2025).
    Combina búsqueda semántica y por palabras clave. Devuelve cada fragmento precedido por
    [Fuente: archivo]."""
    retriever = await get_retriever()
    hits = await retriever.asearch(consulta, k)
    result = ResultadoBusqueda(
        consulta=consulta,
        fragmentos=[
            Fragmento(
                fuente=str(h.document.metadata.get("source", "?")),
                seccion=str(h.document.metadata.get("seccion") or h.document.metadata.get("titulo") or ""),
                chunk_id=h.chunk_id,
                puntaje_rrf=h.score,
                contenido=h.document.page_content,
            )
            for h in hits
        ],
    )
    return result.to_text(), result.model_dump()


@tool(response_format="content_and_artifact")
def listar_documentos() -> tuple[str, dict[str, object]]:
    """Lista los documentos disponibles en la base documental."""
    result = ListaDocumentos(documentos=[d.metadata["source"] for d in load_documents()])
    return "\n".join(f"- {name}" for name in result.documentos), result.model_dump()


for _tool in (buscar_documentos, listar_documentos):
    _tool.handle_validation_error = validation_message  # type: ignore[assignment]

RESEARCH_TOOLS = [buscar_documentos, listar_documentos]
