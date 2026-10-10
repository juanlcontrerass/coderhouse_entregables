"""Cadena LCEL de extracción: ChatPromptTemplate | ChatOpenAI.with_structured_output | validación, con reintentos."""

import itertools
import logging
import os
from contextvars import ContextVar
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import EntidadesTecnicas, RespuestaIncompletaError

load_dotenv()

logger = logging.getLogger("pipeline")

# Mismo proveedor del Módulo 1: GLM de Z.ai vía el router de HuggingFace (API compatible con OpenAI)
HF_ROUTER_URL = "https://router.huggingface.co/v1"
MAX_INTENTOS = 3

modelo = ChatOpenAI(
    model=os.environ.get("ZAI_MODEL", "zai-org/GLM-5.3-Flash:novita"),
    temperature=0,
    max_tokens=int(os.environ.get("MAX_TOKENS", "512")),
    base_url=HF_ROUTER_URL,
    api_key=os.environ.get("HF_TOKEN"),
)

CRITERIOS_CRITICIDAD = (
    "- alta: caída de servicio, pérdida de datos, fallo de seguridad o bloqueo en producción.\n"
    "- media: degradación de rendimiento, errores intermitentes o riesgo que aún no afecta a usuarios.\n"
    "- baja: texto descriptivo, advertencias menores o sin problema evidente."
)

# Sin f-strings: LangChain gestiona las variables {texto} y {criterios_criticidad}
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un ingeniero de software senior. Extrae las entidades técnicas del texto que envía el usuario.\n"
            "Lista solo tecnologías que el texto mencione o que se deduzcan sin ambigüedad; no inventes.\n"
            "Asigna el nivel de criticidad con estos criterios:\n{criterios_criticidad}\n"
            "Escribe el resumen técnico en español, en una o dos oraciones.",
        ),
        ("human", "{texto}"),
    ]
).partial(criterios_criticidad=CRITERIOS_CRITICIDAD)

# Contador de intentos por ejecución de process_text (seguro con llamadas concurrentes)
_intentos: ContextVar[Iterator[int]] = ContextVar("intentos")


def _validar(salida: dict) -> EntidadesTecnicas:
    """Revisa la respuesta cruda antes de entregar el objeto; lanza un error reintentable si no sirve."""
    contador = _intentos.get(None)
    intento = next(contador) if contador else 1
    finish_reason = salida["raw"].response_metadata.get("finish_reason")

    # Primero el finish_reason: si el modelo se quedó sin tokens, el JSON está cortado
    if finish_reason == "length":
        logger.warning("Intento %d/%d: respuesta truncada (finish_reason=length)", intento, MAX_INTENTOS)
        raise RespuestaIncompletaError("respuesta truncada por límite de tokens")

    if salida["parsing_error"] is not None:
        logger.warning("Intento %d/%d: salida inválida: %s", intento, MAX_INTENTOS, salida["parsing_error"])
        raise RespuestaIncompletaError("la salida no cumple el esquema") from salida["parsing_error"]

    if salida["parsed"] is None:
        logger.warning("Intento %d/%d: el modelo no devolvió salida estructurada", intento, MAX_INTENTOS)
        raise RespuestaIncompletaError("el modelo no devolvió salida estructurada")

    logger.info("Intento %d/%d: validación OK (finish_reason=%s)", intento, MAX_INTENTOS, finish_reason)
    return salida["parsed"]


# include_raw=True devuelve {"raw", "parsed", "parsing_error"}: permite ver el finish_reason del AIMessage.
# method="function_calling": el esquema viaja como una tool, que es lo que soportan los proveedores del router
# de HF; además, un JSON cortado o inválido llega a _validar en vez de explotar dentro del SDK de OpenAI.
extraccion = modelo.with_structured_output(
    EntidadesTecnicas, method="function_calling", include_raw=True
) | RunnableLambda(_validar)

# El reintento envuelve modelo + validación: cada intento fallido vuelve a llamar al LLM
chain = prompt | extraccion.with_retry(
    retry_if_exception_type=(RespuestaIncompletaError, ValidationError, OutputParserException),
    stop_after_attempt=MAX_INTENTOS,
    wait_exponential_jitter=True,
)


async def process_text(text: str) -> EntidadesTecnicas:
    """Extrae las entidades técnicas de `text` y devuelve un objeto validado."""
    if not text.strip():
        raise ValueError("el texto de entrada está vacío")

    _intentos.set(itertools.count(1))
    logger.info("Procesando texto de %d caracteres", len(text))
    try:
        resultado = await chain.ainvoke({"texto": text})
    except (RespuestaIncompletaError, ValidationError, OutputParserException) as e:
        logger.error("Sin respuesta válida tras %d intentos: %s", MAX_INTENTOS, e)
        raise
    logger.info(
        "Extracción completa: %d tecnologías, criticidad %s",
        len(resultado.tecnologias),
        resultado.nivel_de_criticidad.value,
    )
    return resultado
