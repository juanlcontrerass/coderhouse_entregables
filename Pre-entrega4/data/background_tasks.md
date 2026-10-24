---
title: Tareas en segundo plano, lifespan y middlewares
category: concurrencia
tags: backgroundtasks, lifespan, middleware, async, celery
---

# Tareas en segundo plano, lifespan y middlewares

Este documento cubre los mecanismos de FastAPI para ejecutar trabajo fuera del ciclo petición-respuesta, preparar recursos al iniciar la aplicación y envolver todas las peticiones con lógica transversal.

## BackgroundTasks

**`BackgroundTasks`** permite programar funciones que se ejecutan *después* de enviar la respuesta al cliente. Es ideal para trabajos cortos que el cliente no necesita esperar: enviar un email de confirmación, escribir un registro de auditoría o invalidar una caché. Se declara como parámetro de la operación y se agregan tareas con `add_task`:

```python
@app.post("/registro")
def registrar(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(enviar_email_bienvenida, email, asunto="Bienvenido")
    return {"mensaje": "Registro recibido"}
```

`add_task` recibe la función y sus argumentos posicionales y con nombre. La función puede ser `def` o `async def`; FastAPI la ejecuta adecuadamente en cada caso. `BackgroundTasks` también puede declararse dentro de dependencias: todas las tareas agregadas en la operación y en sus dependencias se combinan y se ejecutan en orden al finalizar la respuesta.

## Limitaciones de BackgroundTasks

Las tareas en segundo plano se ejecutan **en el mismo proceso** que atiende las peticiones. Eso implica que:

- Si el proceso se reinicia o se cae, las tareas pendientes se pierden; no hay reintentos ni persistencia.
- Una tarea pesada en CPU compite con el procesamiento de las peticiones del mismo worker.
- No hay forma nativa de consultar el estado de una tarea ni de programarla para más tarde.

Para trabajos largos, críticos o que requieren reintentos (procesamiento de video, generación de reportes masivos, integración con sistemas externos lentos) se recomienda una **cola de tareas** dedicada como **Celery**, **RQ**, **Dramatiq** o **ARQ**, con un broker como Redis o RabbitMQ y workers en procesos separados. La API solo encola el trabajo y devuelve un identificador (típicamente con código **202 Accepted**) que el cliente puede consultar luego.

## Eventos de ciclo de vida: lifespan

Muchas aplicaciones necesitan preparar recursos al iniciar (cargar un modelo de machine learning, abrir un pool de conexiones, calentar una caché) y liberarlos al apagar. La forma recomendada es el parámetro **`lifespan`** del constructor, que recibe un *async context manager*:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = cargar_modelo()
    yield
    app.state.modelo = None

app = FastAPI(lifespan=lifespan)
```

El código anterior al `yield` corre una vez antes de aceptar peticiones; el posterior corre una vez al apagar el servidor. Los antiguos decoradores `@app.on_event("startup")` y `@app.on_event("shutdown")` están **deprecados** en favor de `lifespan`; además, si se define `lifespan`, los manejadores `on_event` dejan de ejecutarse. Los recursos creados en el lifespan pueden guardarse en `app.state` o devolverse en el `yield` como diccionario de estado, accesible luego desde `request.state`.

## Middlewares

Un **middleware** es una función que recibe cada petición antes que cualquier operación de path y cada respuesta antes de que se envíe. Se declara con el decorador `@app.middleware("http")`:

```python
@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    inicio = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.perf_counter() - inicio)
    return response
```

`call_next` pasa la petición al resto de la aplicación y devuelve la respuesta. Los middlewares se ejecutan en orden inverso al de registro para la petición (el último agregado es el más externo). FastAPI y Starlette incluyen middlewares listos para usar: `CORSMiddleware`, `GZipMiddleware` (comprime respuestas grandes, con el parámetro `minimum_size`), `TrustedHostMiddleware` (rechaza peticiones con headers `Host` no permitidos) y `HTTPSRedirectMiddleware` (redirige HTTP a HTTPS).

Las dependencias con `yield` se ejecutan después de los middlewares en el orden de entrada; por eso un middleware no puede acceder a los valores de las dependencias. Cuando la lógica solo aplica a un subconjunto de rutas, una dependencia suele ser una mejor opción que un middleware global.

## Concurrencia: async y el event loop

FastAPI se ejecuta sobre un event loop asíncrono. Las operaciones `async def` se ejecutan directamente en el loop, por lo que **nunca deben bloquearlo**: llamar a `time.sleep`, a `requests.get` o a un driver síncrono dentro de una función `async def` detiene todas las demás peticiones del worker. Las alternativas son usar la versión asíncrona de la librería (`asyncio.sleep`, `httpx.AsyncClient`) o declarar la función con `def` para que FastAPI la ejecute en el pool de hilos. Para trabajo intensivo en CPU dentro de un endpoint asíncrono, se puede delegar con `run_in_threadpool` de Starlette o a un `ProcessPoolExecutor`.

## WebSockets y respuestas en streaming

Además de HTTP clásico, FastAPI soporta **WebSockets** con el decorador `@app.websocket("/ws")`, donde la función recibe un objeto `WebSocket` y usa `await websocket.accept()`, `receive_text()` y `send_text()`. Para enviar datos de forma progresiva por HTTP se usa `StreamingResponse` con un generador, útil para descargar archivos grandes o transmitir tokens de un modelo de lenguaje a medida que se generan. Los *Server-Sent Events* se implementan con `StreamingResponse` y el media type `text/event-stream`.
