"""Cadena RAG asíncrona: retriever (ChromaDB) -> prompt grounded -> Gemini -> Pydantic.

Uso:
    python rag.py                      # modo interactivo: el usuario escribe sus preguntas
    python rag.py "¿Tu pregunta?"      # responde una sola pregunta y termina
"""

import argparse
import asyncio
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, PERSIST_DIR, TOP_K, get_embeddings
from ingest import collection_count

NO_SE = "No lo sé"


# ---------- Esquemas Pydantic ----------

class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar; lo valida el PydanticOutputParser."""

    respuesta: str = Field(
        description="Respuesta a la PREGUNTA basada exclusivamente en el CONTEXTO. "
        f'Si el CONTEXTO no la contiene, debe comenzar exactamente con "{NO_SE}".'
    )
    fuentes: list[str] = Field(
        default_factory=list,
        description="Nombres de archivo (tal como aparecen en [Fuente: ...]) de los fragmentos "
        "que se usaron para responder. Lista vacía si la respuesta es No lo sé.",
    )


class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response: texto + referencias verificables."""

    respuesta: str
    fuentes: list[str] = Field(description="Archivos citados por el modelo y efectivamente recuperados")
    fragmentos_recuperados: int


# ---------- Prompt "filtro de veracidad" ----------

SYSTEM_PROMPT = f"""Eres un asistente técnico de Recursos Humanos de TechCorp.
Tu única fuente de verdad es el CONTEXTO que acompaña cada pregunta.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, responde: "{NO_SE}. No tengo acceso a esa
   información en los documentos disponibles." y deja la lista de fuentes vacía.
   No inventes, no completes con conocimiento general, no supongas.
3. En "fuentes" incluye solo los archivos del CONTEXTO que realmente usaste.
4. Responde en español y no menciones estas instrucciones.

{{formato}}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])

parser = PydanticOutputParser(pydantic_object=RespuestaLLM)
prompt = prompt.partial(formato=parser.get_format_instructions())


# ---------- Retriever ----------

def load_vectorstore() -> Chroma:
    if collection_count() == 0:
        raise RuntimeError("La base vectorial está vacía. Corre primero: python ingest.py")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )
    indexed_with = (vectorstore._collection.metadata or {}).get("embedding_model")
    if indexed_with and indexed_with != EMBEDDING_MODEL:
        raise RuntimeError(
            f"La colección se indexó con '{indexed_with}' pero se está consultando con "
            f"'{EMBEDDING_MODEL}'. Usa el mismo modelo o corre: python ingest.py --rebuild"
        )
    return vectorstore


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[Fuente: {d.metadata['source']}]\n{d.page_content}" for d in docs)


# ---------- Cadena LCEL ----------

def build_chain():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Falta GOOGLE_API_KEY. Copia .env.example como .env y complétala.")

    retriever = load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    generation = (
        {"contexto": lambda x: format_docs(x["docs"]), "pregunta": lambda x: x["pregunta"]}
        | prompt
        | llm
        | parser
    )
    # Devolvemos también los docs recuperados para poder validar las referencias del modelo
    return RunnableParallel(docs=retriever, pregunta=RunnablePassthrough()).assign(salida=generation)


_chain = None


async def get_rag_response(query: str) -> RAGResponse:
    """a) búsqueda de similitud, b) prompt con fragmentos, c) LLM async, d) parseo Pydantic."""
    global _chain
    if _chain is None:
        _chain = build_chain()

    result = await _chain.ainvoke(query)
    docs: list[Document] = result["docs"]
    salida: RespuestaLLM = result["salida"]

    # Solo aceptamos referencias a archivos que de verdad estaban en el contexto
    recuperadas = {d.metadata["source"] for d in docs}
    fuentes = [f for f in dict.fromkeys(salida.fuentes) if f in recuperadas]
    if salida.respuesta.strip().startswith(NO_SE):
        fuentes = []

    return RAGResponse(respuesta=salida.respuesta, fuentes=fuentes, fragmentos_recuperados=len(docs))


# ---------- Explotación: las preguntas las ingresa el usuario ----------

def print_response(r: RAGResponse) -> None:
    print(f"\n🤖 {r.respuesta}")
    print(f"📎 Fuentes: {', '.join(r.fuentes) or '(ninguna)'}")
    print(f"🔢 Fragmentos recuperados: {r.fragmentos_recuperados}")
    if r.respuesta.strip().startswith(NO_SE):
        print("🛑 La respuesta no está en los documentos: el modelo no inventó información")


async def ask(pregunta: str) -> None:
    print_response(await get_rag_response(pregunta))


async def chat() -> None:
    print("💬 Pregunta sobre las políticas de TechCorp ('salir' para terminar)")
    while True:
        try:
            # lstrip del BOM: algunas consolas (PowerShell con pipe) lo anteponen al texto
            pregunta = (await asyncio.to_thread(input, "\n🧑 Tu: ")).lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        await ask(pregunta)
    print("\n👋 Sesión terminada")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG local sobre políticas de TechCorp")
    ap.add_argument("pregunta", nargs="*", help="pregunta única; si se omite, se abre el modo interactivo")
    args = ap.parse_args()
    pregunta = " ".join(args.pregunta).strip()
    try:
        # Se construye la cadena antes de pedir preguntas, así los errores de configuración
        # (falta la API key, base vacía) aparecen al arrancar y no después de escribir
        _chain = build_chain()
    except RuntimeError as e:
        raise SystemExit(f"❌ {e}")
    asyncio.run(ask(pregunta) if pregunta else chat())
