---
title: Ruteo y operaciones de path en FastAPI
category: fundamentos
tags: routing, path-parameters, query-parameters, apirouter
---

# Ruteo y operaciones de path en FastAPI

En FastAPI, una **operación de path** es la combinación de una ruta (por ejemplo `/items/{item_id}`) y un método HTTP (GET, POST, PUT, PATCH, DELETE). Cada operación se declara con un decorador sobre la instancia de la aplicación, como `@app.get("/items")` o `@app.post("/items")`, y la función decorada se denomina *path operation function*. FastAPI inspecciona la firma de esa función para decidir de dónde sale cada argumento, cómo validarlo y cómo documentarlo en el esquema OpenAPI que se publica automáticamente en `/docs` (Swagger UI) y en `/redoc`.

## Creación de la aplicación

El punto de partida es siempre un objeto `FastAPI()`. Al crearlo se pueden pasar parámetros descriptivos como `title`, `version` y `description`, que aparecen en la documentación interactiva. También se puede desactivar la documentación en producción pasando `docs_url=None` y `redoc_url=None`. La aplicación es un objeto ASGI, por lo que se ejecuta con un servidor como Uvicorn: `uvicorn main:app --reload` levanta el servicio en modo desarrollo con recarga automática ante cambios de código.

## Parámetros de path

Los parámetros de path se declaran entre llaves dentro de la ruta y con el mismo nombre en la firma de la función. Si la función declara `item_id: int`, FastAPI convierte el segmento de la URL a entero y, si la conversión falla, responde automáticamente con un error de validación. Esta conversión basada en anotaciones de tipo es una de las características centrales del framework: el mismo tipo sirve para validar, convertir y documentar.

El orden de declaración de las rutas importa. Si existe una ruta fija `/users/me` y otra variable `/users/{user_id}`, la ruta fija debe declararse primero; de lo contrario, la petición a `/users/me` sería capturada por la ruta variable e intentaría interpretar "me" como identificador. Para restringir los valores posibles de un parámetro de path se puede usar una clase `Enum` que herede de `str`; la documentación mostrará entonces un desplegable con los valores permitidos.

Cuando un parámetro de path debe contener barras, como una ruta de archivo, se usa el convertidor de Starlette `{file_path:path}`. Así la URL `/files/home/usuario/notas.txt` entrega el valor completo `home/usuario/notas.txt`.

## Parámetros de query

Todo argumento de la función que no forma parte de la ruta y que es de un tipo simple (int, float, str, bool) se interpreta como **parámetro de query**. Por ejemplo, `def listar(skip: int = 0, limit: int = 10)` acepta la URL `/items?skip=20&limit=5`. Si el argumento tiene valor por defecto, el parámetro es opcional; si no lo tiene, es obligatorio y su ausencia produce un error de validación. Los booleanos son flexibles: los valores `1`, `true`, `on` y `yes` se interpretan como verdadero.

Para agregar validaciones adicionales se utiliza `Query` junto con `Annotated`: `q: Annotated[str | None, Query(min_length=3, max_length=50)] = None`. `Query` admite también `pattern` para expresiones regulares, `alias` para aceptar un nombre de parámetro distinto del nombre de la variable y `deprecated=True` para marcarlo como obsoleto en la documentación. Si se declara un parámetro de tipo `list[str]` con `Query()`, el cliente puede repetirlo en la URL (`?tag=a&tag=b`) y se recibe como lista.

Para los parámetros de path existe el equivalente `Path`, que permite validaciones numéricas como `gt`, `ge`, `lt` y `le`: `item_id: Annotated[int, Path(ge=1, le=1000)]`.

## Cuerpo de la petición

Cuando un argumento está anotado con un modelo de Pydantic (una clase que hereda de `BaseModel`), FastAPI lo lee desde el **cuerpo JSON** de la petición. El modelo define los campos, sus tipos y valores por defecto; FastAPI valida el JSON entrante contra el modelo y entrega a la función una instancia ya validada. Se pueden combinar en la misma función parámetros de path, de query y un cuerpo: FastAPI los distingue por el lugar donde se declaran y por su tipo. Para forzar que un valor simple venga en el cuerpo se usa `Body()`, y con `Body(embed=True)` se espera que el modelo venga anidado bajo una clave con su nombre.

## Modelos de respuesta y códigos de estado

El parámetro `response_model` del decorador indica el esquema de salida. FastAPI filtra la respuesta según ese modelo, lo que permite, por ejemplo, devolver un objeto de usuario sin exponer el campo de contraseña. Desde versiones recientes también se puede declarar el tipo de retorno de la función (`-> UserOut`) con el mismo efecto. Opciones como `response_model_exclude_unset=True` omiten los campos que no se asignaron explícitamente.

El código de estado por defecto es 200, pero puede cambiarse con `status_code=201` (o usando las constantes de `fastapi.status`, como `status.HTTP_201_CREATED`). Para devolver errores controlados se lanza `HTTPException(status_code=404, detail="Item no encontrado")`; FastAPI la convierte en una respuesta JSON con la clave `detail`.

## Organización con APIRouter

En aplicaciones medianas y grandes no conviene declarar todas las rutas en un solo archivo. **`APIRouter`** funciona como una "mini aplicación": se crea `router = APIRouter(prefix="/users", tags=["users"])`, se declaran las operaciones con `@router.get(...)` y luego se registra en la aplicación principal con `app.include_router(router)`. El `prefix` se antepone a todas las rutas del router y los `tags` agrupan las operaciones en la documentación.

`include_router` acepta además `dependencies`, que se aplican a todas las operaciones del router (por ejemplo, una verificación de token), y `responses`, para documentar respuestas adicionales comunes. Una estructura típica de proyecto ubica cada router en `app/routers/<recurso>.py`, los modelos en `app/schemas.py` y las dependencias compartidas en `app/dependencies.py`.

## Funciones síncronas y asíncronas

Una operación de path puede declararse con `def` o con `async def`. Si la función usa librerías asíncronas (por ejemplo, un cliente HTTP con `await`), debe declararse con `async def`. Si usa librerías bloqueantes, como un driver de base de datos síncrono, conviene declararla con `def` normal: FastAPI la ejecuta en un pool de hilos externo para no bloquear el event loop. Mezclar ambos estilos en una misma aplicación es perfectamente válido; lo que debe evitarse es llamar código bloqueante dentro de una función `async def`, porque congela el event loop y degrada la latencia de todas las peticiones concurrentes.
