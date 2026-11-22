"""Logs con contexto de correlación.

`log_context(job_id=..., conversation_id=..., trace_id=...)` fija esos valores en una
`ContextVar`; cada línea de log los incluye, así un error del worker se cruza directamente con
su traza en Phoenix.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_context: ContextVar[dict[str, str]] = ContextVar("log_context", default={})

FORMAT = "%(asctime)s %(levelname)s %(name)s [%(ctx)s] %(message)s"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _context.get()
        record.ctx = " ".join(f"{k}={v}" for k, v in ctx.items() if v) or "-"
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(ContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


@contextmanager
def log_context(**values: str | None) -> Iterator[None]:
    token = _context.set({**_context.get(), **{k: v for k, v in values.items() if v}})
    try:
        yield
    finally:
        _context.reset(token)


def update_log_context(**values: str | None) -> None:
    """Agrega valores al contexto actual (por ejemplo, el trace_id una vez conocido)."""
    _context.set({**_context.get(), **{k: v for k, v in values.items() if v}})
