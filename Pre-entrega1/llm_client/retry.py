import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def con_reintentos(
    operacion: Callable[[], Awaitable[T]],
    es_reintentable: Callable[[Exception], bool],
    max_retries: int,
    base_delay: float = 1.0,
) -> T:
    """Ejecuta `operacion` y la reintenta con backoff exponencial ante errores transitorios.

    Si el error no es reintentable o se agotan los intentos, relanza la excepción
    para que el cliente la convierta en un error estructurado.
    """
    for intento in range(max_retries + 1):
        try:
            return await operacion()
        except Exception as e:
            if intento == max_retries or not es_reintentable(e):
                raise
            # asyncio.sleep cede el control al event loop: la espera no bloquea otras tareas
            await asyncio.sleep(base_delay * 2**intento)
    raise AssertionError("inalcanzable")
