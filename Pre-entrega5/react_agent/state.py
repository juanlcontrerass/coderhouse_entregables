"""Esquema de estado del grafo."""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Hereda `messages` y su reducer `add_messages`, que acumula el historial."""
