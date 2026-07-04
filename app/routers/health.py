
from fastapi import APIRouter
from app.schemas.common import HealthResponse
from app.config.settings import get_settings

settings = get_settings()
router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse)
def read_root():
    return HealthResponse(
        message=f"Welcome to {settings.APP_NAME}",
        version=settings.APP_VERSION,
        author=settings.APP_AUTHOR,
    )

