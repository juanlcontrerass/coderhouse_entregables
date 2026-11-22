"""Herramienta con efecto lateral: publica el informe final.

Es la acción crítica del sistema: deja un archivo fuera del grafo, por eso solo la ejecuta el
nodo `publisher`, al que se llega únicamente después de la aprobación humana.
"""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.core.config import get_settings
from app.tools.schemas import InformePublicado, PublicarInformeInput

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def report_path(job_id: str, reports_dir: Path | None = None) -> Path:
    """Ruta del informe; el id se sanea para que no pueda salir del directorio."""
    return (reports_dir or get_settings().reports_dir) / f"informe_{_SAFE_ID.sub('_', job_id)}.md"


def _write_report(path: Path, content: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.stat().st_size


@tool(args_schema=PublicarInformeInput, response_format="content_and_artifact")
async def publicar_informe(job_id: str, consulta: str, contenido: str, aprobado_por: str = "") -> tuple[str, dict[str, Any]]:
    """Publica el informe final en el repositorio de informes de TechCorp. Tiene efectos
    laterales: solo debe ejecutarse con aprobación humana."""
    path = report_path(job_id)
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = (
        f"# Informe {job_id}\n\n"
        f"- Publicado: {published_at}\n"
        f"- Aprobación: {aprobado_por or 'sin comentario'}\n\n"
        f"## Consulta\n\n{consulta}\n\n## Respuesta\n\n{contenido}\n"
    )
    size = await asyncio.to_thread(_write_report, path, body)  # E/S de disco fuera del event loop
    result = InformePublicado(ruta=str(path), publicado_en=published_at, bytes=size)
    return str(path), result.model_dump()
