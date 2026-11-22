# Decisiones de arquitectura

Registro breve de las decisiones técnicas (formato ADR: contexto, decisión, consecuencias).

## ADR-01. Ensamblar las pre-entregas en lugar de reescribir

- **Contexto:** el orquestador del Módulo 6 y la API del Módulo 7 ya estaban probados (tests offline, Phoenix, HITL).
- **Decisión:** partir de la API del Módulo 7 y reorganizarla por capas (`core`, `llm`, `rag`, `tools`, `agents`, `graph`, `storage`, `api`, `worker`, `observability`). La lógica de los agentes no cambia: cambian sus fronteras (configuración, contratos y persistencia).
- **Consecuencias:** se conserva la cobertura de tests existente y se reduce el riesgo; cada capa se puede leer sin conocer el historial del curso.

## ADR-02. La API nunca ejecuta el grafo (cola Arq + worker)

- **Contexto:** un turno tarda entre 10 y 60 s (varias llamadas al LLM). Ejecutarlo dentro del request ata la conexión y degrada la API con pocas peticiones concurrentes.
- **Decisión:** `POST` responde `202` con un `job_id` en milisegundos; un worker de Arq (proceso separado, `WORKER_MAX_JOBS` en paralelo) ejecuta el grafo. El cliente hace polling de `GET /tasks/{id}`.
- **Consecuencias:** la API escala de forma independiente del cómputo; el costo es un componente más (worker) y la necesidad de un estado de job explícito (`PENDING` → `RUNNING` → `DONE`/`FAILED`...).

## ADR-03. Redis como única pieza de estado

- **Contexto:** se necesitan tres cosas persistentes: estado de jobs, cola de tareas y checkpoints de LangGraph.
- **Decisión:** Redis 8 (incluye RedisJSON y RediSearch, que exige `langgraph-checkpoint-redis`) con AOF activado. `AsyncRedisSaver` como checkpointer en lugar de `AsyncSqliteSaver` (Módulo 5), porque API y worker son procesos distintos y el worker puede escalar a varias réplicas.
- **Consecuencias:** un solo servicio de infraestructura; las conversaciones y las tareas en pausa sobreviven a reinicios de la API, del worker y del propio contenedor de Redis (volumen + AOF). Los checkpoints expiran por TTL (`CONVERSATION_TTL_MINUTES`).

## ADR-04. Conversación = thread del checkpointer; un turno a la vez

- **Contexto:** la consigna pide que las conversaciones no se pierdan al reiniciar el servidor.
- **Decisión:** `conversation_id` es el `thread_id` de LangGraph. `messages` y `total_tokens` se acumulan entre turnos; los campos del turno (`contributions`, `decisions`, `tokens_used`, contadores) se reinician con reducers que aceptan `None` como "vaciar". Una conversación solo procesa un turno por vez (bloqueo atómico con `WATCH/MULTI`; un segundo mensaje recibe `409`).
- **Consecuencias:** el supervisor y el sintetizador reciben las últimas `HISTORY_TURNS` preguntas y respuestas para resolver repreguntas; los especialistas siguen recibiendo solo su instrucción (contexto mínimo, menos tokens). El bloqueo evita dos ejecuciones concurrentes sobre el mismo thread, que corromperían el checkpoint.

## ADR-05. Chroma local + búsqueda híbrida (BM25 + densa, RRF) en lugar de Pinecone

- **Contexto:** la entrega se evalúa con despliegue local y debe arrancar en una máquina limpia. Pinecone exige otra cuenta, otra clave y red.
- **Decisión:** Chroma persistente con embeddings multilingües locales (`EMBEDDING_MODEL`) y BM25 sobre los mismos fragmentos, fusionados con Reciprocal Rank Fusion ponderada, la misma estrategia del Módulo 4 (`EnsembleRetriever`, pesos 0.5/0.5, deduplicación por `chunk_id`). Se reimplementó en ~20 líneas (`app/rag/retriever.py`) para controlar la asincronía y evitar dos dependencias pesadas.
- **Consecuencias:** cero claves extra; BM25 recupera términos exactos ("eNPS", "2FA") que los embeddings diluyen. Colección, modelo y pesos se configuran por entorno: cambiar de vector DB solo requiere otra implementación de `VectorStore` en `_open_chroma`.

## ADR-06. Pydantic en cada frontera, también en las herramientas

- **Contexto:** el LLM es una fuente de datos no confiable: puede proponer argumentos inválidos.
- **Decisión:** cada herramienta declara `args_schema` (restricciones de longitud, rangos, valores prohibidos como `valor_inicial = 0`) y devuelve su salida como *artifact* validado (`response_format="content_and_artifact"`). Un error de validación vuelve al agente como mensaje para que corrija la llamada. La API usa modelos tipados para toda entrada y salida (`TaskResult`, `ApprovalRequest`...), y el payload de `interrupt()` también es un modelo.
- **Consecuencias:** las fuentes que valida el nodo de Validación salen del artifact estructurado, no del texto del LLM. Un cambio de contrato rompe un test, no producción.

## ADR-07. Human-in-the-loop determinista

- **Contexto:** publicar un informe es un efecto lateral; un turno muy caro también merece revisión.
- **Decisión:** `assess_risk` (sin LLM) marca la tarea como crítica si pide publicar o si supera `COST_APPROVAL_TOKEN_THRESHOLD`. El nodo `human_approval` llama a `interrupt()` antes del efecto lateral; la aprobación usa una transición atómica para que dos aprobaciones simultáneas no reanuden dos veces.
- **Consecuencias:** el criterio es auditable y testeable; el turno queda tomado mientras espera aprobación.

## ADR-08. Observabilidad con Arize Phoenix (OpenInference)

- **Contexto:** "un sistema que funciona pero es una caja negra no puntúa".
- **Decisión:** `LangChainInstrumentor` genera un span por nodo, agente, herramienta y llamada al LLM (con tokens, por lo que Phoenix calcula costo). Cada job tiene un span raíz `job_turno`; `using_session(conversation_id)` agrupa los turnos de una conversación. El `trace_id` se guarda en el job y lo devuelve la API; los logs incluyen `job_id`, `conversation_id` y `trace_id`.
- **Consecuencias:** desde una respuesta de la API se llega a su traza en un paso. El exportador es por lotes y corre en un hilo propio, sin bloquear el event loop.

## ADR-09. Todo async, y lo que no lo es, fuera del event loop

- **Decisión:** LLM con el cliente async del SDK de OpenAI (httpx), Redis con `redis.asyncio`, checkpointer `AsyncRedisSaver`, búsqueda densa con `asimilarity_search`. Lo inevitablemente bloqueante (cargar el modelo de embeddings, indexar, escribir el informe en disco) corre en `asyncio.to_thread`. Un test verifica que ningún módulo importe `requests`.

## ADR-10. Configuración sin hard-coding

- **Decisión:** `pydantic-settings` (`app/core/config.py`) valida todas las variables al arrancar. Modelo, embeddings y colección son obligatorios y no tienen valor por defecto en el código; un test verifica que no haya nombres de modelos escritos en `app/`.
