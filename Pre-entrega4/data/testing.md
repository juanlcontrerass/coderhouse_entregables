---
title: Testing de aplicaciones FastAPI
category: calidad
tags: testclient, pytest, httpx, dependency-overrides, asyncclient
---

# Testing de aplicaciones FastAPI

Probar una API construida con FastAPI es sencillo porque el framework permite invocar la aplicación directamente desde los tests, sin levantar un servidor real. Las herramientas centrales son `TestClient`, `pytest` y el mecanismo de sobrescritura de dependencias.

## TestClient

**`TestClient`** (en `fastapi.testclient`) está construido sobre **HTTPX** y expone la misma interfaz que un cliente HTTP: `client.get`, `client.post`, `client.put`, `client.delete`. Internamente llama a la aplicación ASGI en memoria, por lo que los tests son rápidos y no requieren puertos abiertos. Para usarlo es necesario instalar `httpx`.

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_leer_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "nombre": "Martillo"}
```

Las funciones de test se escriben con `def` normal, no con `async def`, aunque la aplicación sea asíncrona: `TestClient` se encarga de ejecutar el event loop. Los parámetros se envían igual que con cualquier cliente HTTP: `json=` para cuerpos JSON, `data=` para formularios, `params=` para query, `headers=` para headers y `files=` para archivos.

## Organización con pytest

Se recomienda ubicar los tests en un directorio `tests/` con archivos `test_*.py` que `pytest` descubre automáticamente. Las **fixtures** de pytest son la forma idiomática de compartir configuración: una fixture `client` que construye el `TestClient`, otra que crea una base de datos temporal, otra que genera un usuario autenticado. Se definen en `tests/conftest.py` para que estén disponibles en todos los archivos de test.

Para que el `lifespan` de la aplicación se ejecute durante los tests (por ejemplo, para cargar recursos de inicio), el cliente debe usarse como context manager: `with TestClient(app) as client:`. Sin el bloque `with`, los eventos de inicio y apagado no se disparan.

## Sobrescritura de dependencias

El mecanismo **`app.dependency_overrides`** es la herramienta clave para aislar los tests de recursos externos. Supongamos que la aplicación depende de `get_db` para obtener una sesión de base de datos. En los tests se define una dependencia alternativa que entrega una sesión a una base SQLite en memoria y se registra así:

```python
def get_db_test():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = get_db_test
```

Con la misma técnica se puede simular un usuario autenticado sobrescribiendo `get_current_user` con una función que devuelve un usuario fijo, evitando generar tokens reales en cada test. Es importante limpiar los overrides al terminar (`app.dependency_overrides.clear()`), idealmente en el teardown de una fixture, para que un test no afecte a otro.

## Tests asíncronos con AsyncClient

Cuando el propio test necesita usar `await` (por ejemplo, para consultar una base de datos asíncrona dentro del test), se usa **`httpx.AsyncClient`** con un `ASGITransport` y un plugin como `pytest-anyio` o `pytest-asyncio`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.anyio
async def test_raiz():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
```

`AsyncClient` no ejecuta los eventos de lifespan por sí solo; si se necesitan, se puede usar una herramienta como `asgi-lifespan` con su `LifespanManager`.

## Probar validaciones y errores

Un buen conjunto de tests no solo cubre el camino feliz. Conviene verificar que:

- Un cuerpo con tipos incorrectos devuelve **422** y que `response.json()["detail"]` indica el campo con problemas.
- Un recurso inexistente devuelve **404**.
- Una petición sin token devuelve **401** y una con token pero sin el scope adecuado devuelve **403**.
- Los modelos de respuesta no filtran campos sensibles como contraseñas o hashes.

## Probar WebSockets

`TestClient` también soporta WebSockets mediante `with client.websocket_connect("/ws") as websocket:`, dentro del cual se usan `websocket.send_text()` y `websocket.receive_text()` o sus variantes JSON. Esto permite probar el protocolo completo sin un navegador.

## Cobertura e integración continua

Con `pytest-cov` se mide la cobertura: `pytest --cov=app --cov-report=term-missing` muestra qué líneas no fueron ejecutadas. En integración continua (GitHub Actions, GitLab CI) se ejecutan los tests en cada push, idealmente contra una base de datos de servicio real (por ejemplo un contenedor de PostgreSQL) para detectar diferencias de comportamiento que SQLite en memoria podría ocultar. Separar tests unitarios rápidos de tests de integración más lentos mediante *markers* de pytest (`@pytest.mark.integration`) permite ejecutarlos selectivamente.
