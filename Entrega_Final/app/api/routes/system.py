"""Endpoints de sistema."""

from fastapi import APIRouter, HTTPException, Request, status

from app import __version__
from app.api.schemas import ErrorResponse, HealthResponse

router = APIRouter(tags=["sistema"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verifica la conexión con Redis",
    responses={503: {"model": ErrorResponse}},
)
async def health(request: Request) -> HealthResponse:
    try:
        await request.app.state.redis.ping()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {exc}") from exc
    return HealthResponse(status="ok", redis="ok", version=__version__)
