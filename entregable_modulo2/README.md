# Refactorización a LCEL Asíncrono

Cadena declarativa con LCEL (LangChain Expression Language) que reemplaza la llamada manual al SDK del Módulo 1. Recibe una pregunta como diccionario y devuelve la respuesta en texto plano:

```
ChatPromptTemplate | ChatOpenAI | StrOutputParser
```

El modelo es GLM de Z.ai, servido por el router de HuggingFace (API compatible con OpenAI), por eso se usa `ChatOpenAI` con `base_url` personalizada.

## Requisitos

- Python 3.10 o superior
- Un token de HuggingFace (`HF_TOKEN`)

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Copia `.env.example` como `.env` y completa tu clave:

```
HF_TOKEN=tu_token
```

Opcionalmente puedes cambiar el modelo con `ZAI_MODEL` (por defecto `zai-org/GLM-5.3-Flash:novita`).

## Ejecución

```bash
python main.py
```

Salida esperada:

```
¿La salida es texto plano (str)?: True
La entropía es una medida del desorden o aleatoriedad de un sistema...
```

## Cómo funciona

| Paso | Componente | Entrada → Salida |
|---|---|---|
| 1 | `ChatPromptTemplate.from_messages` (roles `system` y `human`) | `dict` → mensajes |
| 2 | `ChatOpenAI` (`temperature=0.3`, `max_tokens=512`) | mensajes → `AIMessage` |
| 3 | `StrOutputParser` | `AIMessage` → `str` |

La cadena se ejecuta de forma asíncrona con `await chain.ainvoke({...})`. Las llaves del diccionario deben coincidir exactamente con las variables del prompt: `pregunta` y `max_lineas`.

Para hacer otra pregunta, cambia el diccionario que recibe `ainvoke` en `main.py`.
