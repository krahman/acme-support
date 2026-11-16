from pydantic import UUID4, BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: UUID4 | None = None


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    session_id: str
    trace_id: str | None


class HealthResponse(BaseModel):
    status: str
    langfuse: bool


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
