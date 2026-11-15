from fastapi import APIRouter

from app.api.schema import HealthResponse
from app.dependencies import LangfuseDependency

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(langfuse: LangfuseDependency):
    return {"status": "ok", "langfuse": langfuse.auth_check()}
