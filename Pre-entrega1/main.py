"""Script de validación: pregunta corta en modo normal, streaming, en paralelo y con una key inválida."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from pydantic import SecretStr

from llm_client import AsyncLLMManager, ChatMessage, LLMConfig, ModelResponse, Provider

load_dotenv()

# Variable de entorno de la API key y modelo por defecto de cada proveedor
PROVEEDORES = {
    Provider.GEMINI: ("GOOGLE_API_KEY", os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")),
    Provider.ZAI: ("HF_TOKEN", os.environ.get("ZAI_MODEL", "zai-org/GLM-5.3-Flash:novita")),
}

PREGUNTA = [ChatMessage(role="user", content="¿Qué es la entropía? Responde en 2 líneas.")]


def crear_managers() -> dict[Provider, AsyncLLMManager]:
    managers = {}
    for provider, (variable, modelo) in PROVEEDORES.items():
        api_key = os.environ.get(variable)
        if not api_key:
            print(f"⚠️  Falta {variable} en el .env: se omite {provider.value}")
            continue
        config = LLMConfig(provider=provider, model=modelo, api_key=SecretStr(api_key))
        managers[provider] = AsyncLLMManager(config)
    return managers


def mostrar(resultado: ModelResponse) -> None:
    texto = resultado.content if not resultado.error else f"❌ {resultado.error}"
    print(f"[{resultado.provider.value} · {resultado.model}] {texto}\n")


async def main() -> None:
    managers = crear_managers()

    print("\n=== Modo normal ===")
    for manager in managers.values():
        mostrar(await manager.generate(PREGUNTA))

    print("=== Modo streaming ===")
    for provider, manager in managers.items():
        print(f"[{provider.value}] ", end="")
        async for chunk in manager.generate_stream(PREGUNTA):
            print(chunk, end="", flush=True)
        print("\n")

    print("=== En paralelo (asyncio.gather) ===")
    resultados = await asyncio.gather(*(m.generate(PREGUNTA) for m in managers.values()))
    for resultado in resultados:
        mostrar(resultado)

    print("=== Resiliencia (API key inválida a propósito) ===")
    for provider, (_, modelo) in PROVEEDORES.items():
        config = LLMConfig(provider=provider, model=modelo, api_key=SecretStr("key-invalida-a-proposito"))
        resultado = await AsyncLLMManager(config).generate(PREGUNTA)
        print(f"[{provider.value}] Error capturado (sin crash): {resultado.error}\n")
    print("¿El programa siguió vivo?: ✅ Sí")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no usa UTF-8 por defecto
    asyncio.run(main())
