# RAG local: Políticas de TechCorp

Sistema de recuperación semántica local (RAG) de punta a punta. Recibe una pregunta, busca los fragmentos más relevantes en una base vectorial **ChromaDB** persistida en disco y genera una respuesta con **Gemini** usando *solo* esa información. Si la respuesta no está en los documentos, el modelo responde **"No lo sé"**.

```
pregunta ─► retriever (ChromaDB, top_k=4) ─► prompt grounded ─► Gemini (async) ─► PydanticOutputParser ─► RAGResponse
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `data/*.md` | Dataset de ejemplo: 4 políticas internas de una empresa ficticia (vacaciones, teletrabajo, seguridad, onboarding) |
| `config.py` | Rutas, parámetros de chunking y **la única función de embeddings** que usan la ingesta y la consulta |
| `ingest.py` | Módulo de ingesta: carga `/data`, fragmenta y persiste en `./vectorstore` |
| `rag.py` | Cadena LCEL, función asíncrona `get_rag_response(query)` y la interfaz de consultas del usuario |

## Requisitos

- Python 3.12 o superior
- Una API key gratuita de Google AI Studio (`GOOGLE_API_KEY`): https://aistudio.google.com/apikey

## Instalación (en un ambiente que tenga Python 3.12 o superior)

```bash
pip install -r requirements.txt
```

## Configuración

Copia `.env.example` como `.env` y completa tu clave:

```
GOOGLE_API_KEY=tu_api_key
```

El `.env` está en `.gitignore`, así que la clave nunca se sube al repositorio.

## Ejecución

**1. Ingesta** (la primera vez descarga el modelo de embeddings, unos 470 MB):

```bash
python ingest.py
```

```
📄 Documentos cargados: 4
✂️  Fragmentos generados: 8 (chunk_size=500, overlap=50 tokens)
📦 8 fragmentos persistidos en .../vectorstore
```

Si la vuelves a correr, detecta el índice y no reindexa:

```
♻️  Índice existente con 8 fragmentos — se omite la reindexación
```

Si modificas los archivos de `/data`, fuerza la reindexación con `python ingest.py --rebuild`.

**2. Consultas.** El código no trae preguntas fijas: las escribe el usuario.

Modo interactivo (escribe `salir` para terminar):

```bash
python rag.py
```

```
💬 Pregunta sobre las políticas de TechCorp ('salir' para terminar)

🧑 Tu: ¿Cuántos días de vacaciones tengo con 5 años de antigüedad?

🤖 A un empleado con 5 años de antigüedad le corresponden 21 días corridos de vacaciones.
📎 Fuentes: politica_vacaciones.md
🔢 Fragmentos recuperados: 4
```

O una sola pregunta pasada como argumento:

```bash
python rag.py "¿Puedo trabajar desde el exterior?"
```

## Cómo funciona

### Ingesta (`ingest.py`)

- Lee todos los `.txt` y `.md` de `/data`.
- Fragmenta con `RecursiveCharacterTextSplitter.from_tiktoken_encoder` usando **500 tokens con 50 de overlap**. Se mide en tokens y no en caracteres porque el límite de contexto del LLM se mide en tokens.
- Guarda cada fragmento con su metadata (`source` y número de `chunk`) en una colección de ChromaDB persistida en `./vectorstore`.
- **Persistencia:** antes de indexar, consulta la cantidad de fragmentos de la colección; si ya hay datos, no recalcula los embeddings.

### Embeddings

Se usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, que corre en local, no tiene costo y está entrenado para español. Para evitar el error de **embeddings no coincidentes**:

1. `ingest.py` y `rag.py` obtienen los embeddings de la misma función, `config.get_embeddings()`.
2. Al indexar se guarda el nombre del modelo en la metadata de la colección, y `rag.py` corta con un error si se intenta consultar con otro modelo.

### Recuperación y generación (`rag.py`)

La cadena LCEL une el retriever, el formateo de documentos, el prompt, el LLM y el parser:

```python
generation = {"contexto": ..., "pregunta": ...} | prompt | llm | parser
chain = RunnableParallel(docs=retriever, pregunta=RunnablePassthrough()).assign(salida=generation)
```

- **Retriever:** búsqueda por similitud con `k=4`, dentro del rango de 3 a 5 que pide la consigna para no caer en el "contexto infinito" (*Lost in the Middle*).
- **Prompt "filtro de veracidad":** el modelo solo puede usar el CONTEXTO; si la información no está, responde `"No lo sé. No tengo acceso a esa información en los documentos disponibles."`.
- **Salida estructurada:** el `PydanticOutputParser` valida que el LLM devuelva un `RespuestaLLM` con `respuesta` y `fuentes`.

`get_rag_response(query: str)` es asíncrona (`await chain.ainvoke(...)`) y devuelve un `RAGResponse`:

```python
class RAGResponse(BaseModel):
    respuesta: str
    fuentes: list[str]            # archivos citados por el modelo
    fragmentos_recuperados: int
```

Las **referencias** las propone el LLM, pero el código descarta cualquier archivo que no haya estado en los fragmentos recuperados, así que no puede devolver una fuente inventada. Cuando la respuesta es "No lo sé", la lista de fuentes queda vacía.

## Pruebas

Para verificar que el sistema no alucina, ingresa una pregunta cuya respuesta esté en los documentos y otra cuya respuesta no esté. Por ejemplo:

| Caso | Pregunta de ejemplo | Resultado esperado |
|---|---|---|
| Respuesta en los documentos | ¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad? | 21 días corridos, con fuente `politica_vacaciones.md` |
| Pregunta trampa | ¿Cuál es la política de bonos por rendimiento anual en TechCorp? | "No lo sé...", sin fuentes |

Cuando la respuesta no está en los documentos, `rag.py` lo indica con `🛑 La respuesta no está en los documentos: el modelo no inventó información`.
