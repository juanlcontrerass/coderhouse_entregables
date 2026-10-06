"""Cadena LCEL asíncrona: ChatPromptTemplate | ChatOpenAI | StrOutputParser."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Mismo proveedor del Módulo 1: GLM de Z.ai vía el router de HuggingFace (API compatible con OpenAI)
HF_ROUTER_URL = "https://router.huggingface.co/v1"

# Antes: AsyncOpenAI(...) + client.chat.completions.create(...) a mano. Ahora: un modelo de LangChain
modelo = ChatOpenAI(
    model=os.environ.get("ZAI_MODEL", "zai-org/GLM-5.3-Flash:novita"),
    temperature=0.3,
    max_tokens=512,
    base_url=HF_ROUTER_URL,
    api_key=os.environ.get("HF_TOKEN"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente experto y conciso. Responde siempre en español, en {max_lineas} líneas como máximo."),
        ("human", "{pregunta}"),
    ]
)

# Pipeline declarativo: dict -> mensajes -> AIMessage -> str
chain = prompt | modelo | StrOutputParser()


async def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        sys.exit("⚠️  Falta HF_TOKEN en el .env")

    # Las llaves del diccionario deben coincidir con las {variables} del prompt
    respuesta = await chain.ainvoke({"pregunta": "¿Qué es la entropía?", "max_lineas": 2})

    print(f"¿La salida es texto plano (str)?: {isinstance(respuesta, str)}")
    print(respuesta)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no usa UTF-8 por defecto
    asyncio.run(main())
