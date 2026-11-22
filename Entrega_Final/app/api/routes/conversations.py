"""Conversaciones multi-turno: el historial vive en el checkpointer de LangGraph."""

from fastapi import APIRouter, HTTPException, Request, status

from app.agents.common import conversation_turns
from app.api.deps import get_checkpointer, get_conversations, new_id, submit_turn
from app.api.schemas import ConversationCreated, ConversationOut, ErrorResponse, MessageRequest, TaskCreated, TurnOut
from app.storage.checkpointer import load_messages

router = APIRouter(prefix="/conversations", tags=["conversaciones"])


@router.post(
    "",
    response_model=ConversationCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Crea una conversación vacía",
)
async def create_conversation(request: Request) -> ConversationCreated:
    conversation_id = new_id()
    await get_conversations(request).create(conversation_id)
    return ConversationCreated(conversation_id=conversation_id, messages_url=f"/conversations/{conversation_id}/messages")


@router.post(
    "/{conversation_id}/messages",
    response_model=TaskCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Envía un mensaje: encola el turno y devuelve su job_id",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def send_message(conversation_id: str, message: MessageRequest, request: Request) -> TaskCreated:
    if await get_conversations(request).get(conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversation_id}.")
    return await submit_turn(request, conversation_id, message)


@router.get(
    "/{conversation_id}",
    response_model=ConversationOut,
    summary="Historial de la conversación (leído del checkpoint, sin ejecutar el grafo)",
    responses={404: {"model": ErrorResponse}},
)
async def get_conversation(conversation_id: str, request: Request) -> ConversationOut:
    record = await get_conversations(request).get(conversation_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {conversation_id}.")
    messages = await load_messages(get_checkpointer(request), conversation_id)
    return ConversationOut(
        conversation_id=conversation_id,
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        active_job_id=record.get("active_job_id") or None,
        job_ids=record["job_ids"],
        turns=[TurnOut(query=q, answer=a) for q, a in conversation_turns(messages)],
    )
