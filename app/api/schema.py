from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    trace_id: str | None


class HealthResponse(BaseModel):
    status: str
    langfuse: bool
