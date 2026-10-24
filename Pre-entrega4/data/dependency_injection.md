---
title: Inyección de dependencias en FastAPI
category: arquitectura
tags: depends, dependency-injection, yield, overrides
---

# Inyección de dependencias en FastAPI

FastAPI incluye un sistema de **inyección de dependencias** simple pero muy potente. Una dependencia es cualquier "invocable" (una función, una clase o un método) que FastAPI ejecuta antes de la operación de path y cuyo resultado entrega como argumento. Se declara con `Depends`: `def leer_items(commons: Annotated[dict, Depends(parametros_comunes)])`. FastAPI llama a `parametros_comunes`, resuelve a su vez los parámetros que ésta declare (query, path, headers, cookies u otras dependencias) y pasa el resultado a la función.

## Para qué sirve

Las dependencias permiten reutilizar lógica compartida sin repetir código: paginación común, obtención de una sesión de base de datos, verificación de autenticación, lectura de configuración o control de permisos. Como las dependencias también declaran parámetros, éstos se validan y se documentan en OpenAPI igual que los de la operación. Una dependencia que declara `token: str = Header()` hace que el header aparezca documentado en todas las operaciones que la usen.

## Alias con Annotated

Una práctica recomendada es crear alias de tipo para dependencias que se repiten: `SessionDep = Annotated[Session, Depends(get_session)]`. Luego cada operación simplemente declara `session: SessionDep`. Esto reduce el ruido visual, mantiene la información de tipo para el editor y hace trivial cambiar la implementación en un solo lugar.

## Clases como dependencias

Si se pasa una clase a `Depends`, FastAPI la instancia usando los parámetros de su `__init__`. Por ejemplo, una clase `Paginador` con `__init__(self, skip: int = 0, limit: int = 100)` se convierte en una dependencia que lee esos dos parámetros de query. La sintaxis abreviada `Depends()` sin argumentos, combinada con la anotación de tipo de la clase, evita repetir el nombre. Usar clases resulta útil cuando la dependencia necesita métodos auxiliares o estado configurado.

## Sub-dependencias y caché por petición

Las dependencias pueden depender de otras dependencias, formando un árbol de profundidad arbitraria. FastAPI resuelve el árbol completo en el orden correcto. Si la misma dependencia aparece varias veces dentro del árbol de una misma petición, por defecto se ejecuta **una sola vez** y su valor se reutiliza (caché por petición). Cuando se necesita que se ejecute cada vez, se declara `Depends(mi_dependencia, use_cache=False)`.

## Dependencias con yield

Para recursos que requieren limpieza, como conexiones o sesiones de base de datos, se usan **dependencias con `yield`**. El código anterior al `yield` se ejecuta antes de la operación; el valor entregado por `yield` se inyecta; y el código posterior se ejecuta después de enviada la respuesta. El patrón clásico es:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

El bloque `finally` garantiza que la sesión se cierre aunque la operación lance una excepción. Si se captura una excepción dentro de una dependencia con yield, debe relanzarse con `raise` para no ocultar el error; de lo contrario FastAPI no podrá generar la respuesta de error adecuada. Las dependencias con yield también pueden lanzar `HTTPException` antes del yield para cortar la petición.

## Dependencias en el decorador y globales

Cuando una dependencia no devuelve un valor útil pero debe ejecutarse (por ejemplo, validar un header `X-API-Key`), se coloca en el decorador: `@app.get("/items", dependencies=[Depends(verificar_api_key)])`. El mismo parámetro existe en `APIRouter` y en el constructor `FastAPI(dependencies=[...])`, que la aplica a **todas** las operaciones de la aplicación. Es la forma idiomática de proteger un grupo completo de rutas.

## Sobrescritura de dependencias en tests

El diccionario **`app.dependency_overrides`** permite reemplazar una dependencia por otra durante las pruebas. Se asigna `app.dependency_overrides[get_db] = get_db_de_prueba` y todas las operaciones que usaban `get_db` pasan a recibir la base de prueba, sin modificar el código de producción. Al finalizar el test se limpia con `app.dependency_overrides.clear()` para no contaminar otros tests. Este mecanismo es la razón principal por la que conviene encapsular el acceso a recursos externos en dependencias: vuelve a la aplicación fácilmente testeable.

## Buenas prácticas

- Mantener las dependencias pequeñas y con una sola responsabilidad; componerlas mediante sub-dependencias.
- Preferir dependencias con `yield` frente a middlewares cuando la limpieza solo aplica a ciertas rutas.
- Declarar como `async def` las dependencias que hacen I/O asíncrono, y como `def` las que llaman código bloqueante, igual que con las operaciones de path.
- Evitar efectos secundarios globales: cada petición debe obtener sus propios recursos.
- Centralizar en `app/dependencies.py` las dependencias compartidas por varios routers.

## Relación con la seguridad

Las utilidades de seguridad de FastAPI, como `OAuth2PasswordBearer` o `HTTPBearer`, son en realidad dependencias. Una función `get_current_user` que depende del esquema de seguridad, decodifica el token y busca al usuario en la base es el ejemplo más común de sub-dependencia. A su vez, una dependencia `get_current_active_user` puede depender de `get_current_user` y rechazar usuarios deshabilitados, construyendo así una cadena de verificaciones reutilizable en cualquier operación.
