# Módulo de Recuperación Escalable con Pinecone (RAG híbrido)

Pre-entrega 4 — Coderhouse AI Engineering. Sistema RAG en la nube que indexa documentación técnica de **FastAPI** (Markdown, PDF y JSON) en un índice **Pinecone Serverless**, la recupera con un **recuperador híbrido** (BM25 léxico + similitud vectorial, fusionados con `EnsembleRetriever`) y mide su calidad con **Precision@5** y **Recall@5** sobre un Golden Set.

```
data/ (md, pdf, json) ─► loaders ─► RecursiveCharacterTextSplitter (600 tok) ─► OpenAI embeddings (1536)
                                                                                     │
                                       Pinecone Serverless  ◄── upsert (id + vector + metadata{text, source, page, category, tags})
                                       namespace "fastapi-docs"
                                          │                 │
                         list + fetch (corpus)      similarity search (k=5)
                                          ▼                 ▼
                                  BM25Retriever ──┐   ┌── Vector retriever
                                                  ▼   ▼
                                        EnsembleRetriever (RRF 0.5/0.5) ─► RAGSystem.search() ─► top-5
```

## Estructura

| Ruta | Qué hace |
|---|---|
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless; valida que la dimensión sea 1536 |
| `ingest.py` | Pipeline de ingesta: carga, fragmenta, genera embeddings y sube a Pinecone (`--rebuild` para reindexar) |
| `query.py` | Consultas interactivas o de una línea (`--mode hybrid\|vector\|bm25`, `--category`) |
| `evaluate.py` | Precision@5, Recall@5 y MRR sobre `evaluation/golden_set.json`, comparando los tres modos |
| `rag_pinecone/config.py` | Variables de entorno, constantes y **la única** función de embeddings |
| `rag_pinecone/loaders.py` | Loaders de Markdown (front-matter), PDF (una unidad por página) y JSON (una por entrada) |
| `rag_pinecone/chunking.py` | Chunking por tokens e IDs deterministas `doc::sección::cN` |
| `rag_pinecone/index.py` | Cliente Pinecone, `ensure_index()` y conteo por namespace |
| `rag_pinecone/ingest.py` | Upsert con `PineconeVectorStore` en el namespace |
| `rag_pinecone/corpus.py` | Reconstruye los chunks desde la metadata de Pinecone (para BM25) |
| `rag_pinecone/rag_system.py` | Clase `RAGSystem` que encapsula el `EnsembleRetriever` |
| `rag_pinecone/metrics.py` | `precision_at_k`, `recall_at_k`, `reciprocal_rank` |
| `data/` | 5 `.md` + 1 `.pdf` (3 páginas) + 1 `.json` (8 entradas) de documentación de FastAPI |
| `tests/` | Tests offline (sin red) de métricas, loaders, chunking y metadata |

## Requisitos

- Python 3.12
- Cuenta en [Pinecone](https://app.pinecone.io) y una API key de [OpenAI]. Indexar este dataset cuesta fracciones de centavo.

## Pasos para replicar el índice

**1. Entorno con Python 3.12**

```bash
pip install -r requirements.txt
```

(En macOS/Linux: `python3.12 -m venv .venv && source .venv/bin/activate`.)

**2. Variables** — copia `.env.example` como `.env` y completa:

```
PINECONE_API_KEY=...
OPENAI_API_KEY=...
INDEX_NAME=fastapi-rag-hibrido
```

Opcionales: `PINECONE_NAMESPACE` (por defecto `fastapi-docs`), `PINECONE_CLOUD` (`aws`) y `PINECONE_REGION` (`us-east-1`, la región del plan gratuito). El `.env` está en `.gitignore`.

**3. Crear el índice** (dense, `dimension=1536`, `metric=cosine`, `ServerlessSpec(aws, us-east-1)`):

```bash
python setup_index.py
```

Si el índice ya existe, no se recrea; si existe con otra dimensión, el script se detiene con un error explicativo en lugar de fallar al subir los vectores.

**4. Ingesta**

```bash
python ingest.py
```

Carga los 7 archivos (16 unidades: 5 markdown, 3 páginas de PDF y 8 entradas JSON), genera **26 chunks** y los sube con IDs deterministas. Volver a correrlo no duplica nada; con `--rebuild` vacía el namespace y reindexa.

**5. Consultar y evaluar**

```bash
python query.py "¿Qué significa un error HTTP 422?"
```

```bash
python evaluate.py --verbose
```

**Tests offline** (no usan Pinecone ni OpenAI):

```bash
pytest -q
```

## Decisiones de diseño

- **Dimensión única**: `EMBEDDING_DIM = 1536` vive en `config.py` y lo usan tanto la creación del índice como `OpenAIEmbeddings(dimensions=1536)`. `ensure_index()` compara la dimensión de un índice existente antes de usarlo.
- **Texto en la metadata**: cada vector guarda el chunk original en `metadata["text"]` junto con `source`/`doc_id`, `page`, `section`, `category`, `tags` (lista), `title` y `chunk_id`. Una sola consulta a Pinecone devuelve todo; no hace falta una base relacional.
- **Namespace**: todos los vectores van a `fastapi-docs`. Otro dataset o inquilino iría a otro namespace del mismo índice, sin mezclar resultados.
- **Chunking**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con 600 tokens y 100 de solapamiento (rango sugerido 500–800). Las páginas del PDF y las entradas JSON se fragmentan por separado, así cada chunk conserva su página o entrada.
- **IDs deterministas** (`routing.md::full::c1`, `deployment_guide.pdf::p2::c0`, `error_reference.json::http-422::c0`): re-ingestar sobrescribe en vez de duplicar, y el `EnsembleRetriever` deduplica por `id_key="chunk_id"` en lugar de por texto.
- **BM25 desde Pinecone**: `corpus.py` lista y trae los vectores del namespace para construir el `BM25Retriever`, así el retriever léxico y el vectorial trabajan siempre sobre los mismos chunks. El tokenizer de BM25 pasa a minúsculas y quita tildes y puntuación.
- **Filtros por metadata**: `RAGSystem.search(query, category="seguridad")` aplica `{"category": {"$eq": ...}}` en Pinecone y el mismo filtro al corpus de BM25.

## Evaluación

El Golden Set (`evaluation/golden_set.json`) tiene 6 preguntas con su `documento_id_esperado` (un archivo o una lista): 3 semánticas (parafraseadas), 2 con términos exactos (`OAuth2PasswordBearer`, `HTTP 422`) y 1 que cruza dos documentos.

- **Recall@5** = documentos esperados que aparecen en el top-5 / documentos esperados. Con un solo documento esperado vale 0 o 1; con dos puede valer 0.5.
- **Precision@5** = chunks del top-5 que pertenecen a un documento esperado / 5. Cada `.md` y el PDF tienen 3 chunks, así que con un solo documento esperado el máximo posible es 60%, no 100% (el JSON tiene 8 chunks y sí puede llegar a 100%).
- **MRR** (extra) = promedio de 1 / posición del primer resultado correcto.

### Resultados

_Pendiente: se completa con la salida real de `python evaluate.py` una vez creado el índice._
