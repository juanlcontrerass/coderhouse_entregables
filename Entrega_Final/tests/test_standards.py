"""Estándares del repositorio: nada síncrono escondido, nada hard-codeado, nada de voseo."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for p in (ROOT / "app").rglob("*.py")] + [p for p in (ROOT / "scripts").rglob("*.py")]
TEXTS = SOURCES + [ROOT / "README.md", ROOT / ".env.example", *(ROOT / "docs").rglob("*.md")]

VOSEO = re.compile(
    r"\b(vos|podés|tenés|querés|sabés|hacés|sos|usá|revisá|completá|cloná|probá|mirá|fijate|"
    r"arrancás|entregás|ensamblás|tenés|copiá|ejecutá|abrí|elegí|escribí|guardá|poné|andá)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_synchronous_http_clients(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import requests|from requests)", text, re.MULTILINE)


@pytest.mark.parametrize("path", list((ROOT / "app").rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_hardcoded_models_or_indexes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"[\"'](gpt-[\w.-]+|text-embedding-[\w-]+|sentence-transformers/[\w-]+)[\"']", text)


@pytest.mark.parametrize("path", [p for p in TEXTS if p.exists()], ids=lambda p: str(p.relative_to(ROOT)))
def test_neutral_spanish_without_voseo(path: Path) -> None:
    found = VOSEO.findall(path.read_text(encoding="utf-8"))
    assert not found, f"voseo en {path.name}: {found}"
