from fastapi import APIRouter

from app.api.schema import ChatRequest, ChatResponse
from app.dependencies import ChatServiceDependency

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, service: ChatServiceDependency) -> ChatResponse:
    return service.chat(req)
