"""Checkpointer de LangGraph en Redis y lectura del historial de una conversación."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import get_settings


@asynccontextmanager
async def redis_checkpointer(redis_url: str | None = None) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """`AsyncRedisSaver`: requiere RedisJSON y RediSearch (incluidos en la imagen `redis:8`).

    Los checkpoints expiran a los `CONVERSATION_TTL_MINUTES` sin uso; cada lectura renueva el TTL.
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    settings = get_settings()
    ttl = {"default_ttl": settings.conversation_ttl_minutes, "refresh_on_read": True}
    async with AsyncRedisSaver.from_conn_string(redis_url or settings.redis_url, ttl=ttl) as saver:
        await saver.asetup()  # crea los índices si no existen
        yield saver


async def load_messages(checkpointer: BaseCheckpointSaver[Any], conversation_id: str) -> list[BaseMessage]:
    """Mensajes del último checkpoint de la conversación, sin ejecutar el grafo."""
    snapshot = await checkpointer.aget_tuple({"configurable": {"thread_id": conversation_id, "checkpoint_ns": ""}})
    if snapshot is None:
        return []
    messages: list[BaseMessage] = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
    return messages
