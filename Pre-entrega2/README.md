# Pipeline de Extracción de Entidades Técnicas

Pipeline LCEL que recibe un párrafo de texto técnico sin procesar (una descripción de arquitectura, un log de error) y devuelve un objeto Pydantic validado:

```
ChatPromptTemplate | ( ChatOpenAI.with_structured_output(EntidadesTecnicas) | validación ).with_retry()
```

El modelo es GLM de Z.ai, servido por el router de HuggingFace (API compatible con OpenAI), por eso se usa `ChatOpenAI` con `base_url` personalizada, igual que en el Módulo 1.

## Archivos

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelo Pydantic `EntidadesTecnicas`, enum `NivelCriticidad` y la excepción `RespuestaIncompletaError` |
| `chain.py` | Cliente, `ChatPromptTemplate`, cadena LCEL con reintentos y la función asíncrona `process_text()` |
| `main.py` | Mini-script de prueba asíncrono con tres textos de ejemplo |

## Requisitos

- Python 3.12 o superior
- Un token de HuggingFace (`HF_TOKEN`) con créditos de Inference Providers

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Copia `.env.example` como `.env` y completa tu clave:

```
HF_TOKEN=tu_token
```

Variables opcionales: `ZAI_MODEL` (por defecto `zai-org/GLM-5.3-Flash:novita`) y `MAX_TOKENS` (por defecto `512`).

## Ejecución

```bash
python main.py
```

El script procesa tres textos: una descripción de arquitectura, un log de error y un texto ambiguo que no nombra ninguna tecnología (prueba de estrés del validador).

## Ejemplo de salida

Entrada:

> El backend es una API en FastAPI que cachea las consultas frecuentes en Redis y persiste en PostgreSQL. Bajo carga, el pool de conexiones se agota con más de 500 usuarios concurrentes y las peticiones empiezan a devolver 503 en producción.

Salida (`resultado.model_dump_json(indent=2)`):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Uso desde código

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("Traceback ... psycopg2.OperationalError: too many connections"))
print(resultado.tecnologias, resultado.nivel_de_criticidad.value)
```

## Cómo funciona

| Paso | Componente | Entrada → Salida |
|---|---|---|
| 1 | `ChatPromptTemplate` (roles `system` y `human`, variables `{texto}` y `{criterios_criticidad}`) | `dict` → mensajes |
| 2 | `ChatOpenAI.with_structured_output(EntidadesTecnicas, method="function_calling", include_raw=True)` | mensajes → `{"raw", "parsed", "parsing_error"}` |
| 3 | `RunnableLambda(_validar)` | ese `dict` → `EntidadesTecnicas` o excepción |
| 4 | `.with_retry(stop_after_attempt=3)` sobre los pasos 2 y 3 | reintenta la llamada al LLM si el paso 3 falla |

### El contrato (`schemas.py`)

- `tecnologias`: lista de strings con al menos un elemento; se limpian espacios y duplicados.
- `nivel_de_criticidad`: enum `baja` / `media` / `alta`.
- `resumen_tecnico`: string de entre 10 y 400 caracteres.

### Resiliencia

`_validar` revisa la respuesta cruda antes de entregar el objeto, y lanza `RespuestaIncompletaError` en tres casos:

1. **`finish_reason == "length"`**: el modelo se quedó sin tokens y el JSON está cortado. Se detecta antes de intentar usar el objeto.
2. **`parsing_error`**: el JSON está mal formado o no cumple el esquema (campo faltante, lista vacía, criticidad fuera del enum).
3. **Sin salida estructurada**: el modelo respondió sin llamar a la tool.

Esa excepción es la que dispara `.with_retry()`, que vuelve a llamar al LLM con espera exponencial hasta 3 intentos. Si se agotan, `process_text()` registra el error y relanza la excepción.

Se usa `method="function_calling"` porque es el modo que soportan los proveedores del router de HF, y porque con él un JSON truncado llega a `_validar` en lugar de fallar dentro del SDK de OpenAI.

### Logs

Cada intento deja una línea en el logger `pipeline`. Ejemplo de una ejecución que se recupera al tercer intento:

```
INFO    pipeline: Procesando texto de 239 caracteres
WARNING pipeline: Intento 1/3: respuesta truncada (finish_reason=length)
WARNING pipeline: Intento 2/3: salida inválida: 2 validation errors for EntidadesTecnicas ...
INFO    pipeline: Intento 3/3: validación OK (finish_reason=tool_calls)
INFO    pipeline: Extracción completa: 3 tecnologías, criticidad alta
```

Para ver los reintentos en vivo, fuerza una respuesta truncada bajando el límite de tokens en el `.env`:

```
MAX_TOKENS=20
```
