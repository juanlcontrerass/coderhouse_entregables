"""Herramienta con efecto lateral: publica el informe final.

Es la acción "crítica" del sistema: deja un archivo fuera del grafo, por eso solo se ejecuta
después de la aprobación humana (ver `app/hitl.py`).
"""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from app.config import REPORTS_DIR

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def report_path(job_id: str, reports_dir: Path | None = None) -> Path:
    """Ruta del informe; el id se sanea para que no pueda salir del directorio."""
    return (reports_dir or REPORTS_DIR) / f"informe_{_SAFE_ID.sub('_', job_id)}.md"


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@tool
async def publicar_informe(job_id: str, consulta: str, contenido: str, aprobado_por: str = "") -> str:
    """Publica el informe final en el repositorio de informes de TechCorp. Tiene efectos
    laterales: solo debe ejecutarse con aprobación humana."""
    path = report_path(job_id, REPORTS_DIR)
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = (
        f"# Informe {job_id}\n\n"
        f"- Publicado: {published_at}\n"
        f"- Aprobación: {aprobado_por or 'sin comentario'}\n\n"
        f"## Consulta\n\n{consulta}\n\n## Respuesta\n\n{contenido}\n"
    )
    # Escritura en disco fuera del event loop
    await asyncio.to_thread(_write_report, path, body)
    return str(path)
