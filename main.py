"""FastAPI service with Langfuse tracing + basic observability."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from agent import run

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}')
log = logging.getLogger("genai")
langfuse = get_client()  # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    langfuse.flush()  # don't lose buffered traces on shutdown


app = FastAPI(title="Acme Support GenAI", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    resp.headers["x-request-id"] = rid
    log.info(f'"{request.method} {request.url.path} {resp.status_code} {ms:.0f}ms rid={rid}"')
    return resp


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    trace_id: str | None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):  # sync def -> runs in threadpool, fine for a blocking graph.invoke
    handler = CallbackHandler()  # every LLM call, tool call and graph node becomes a span
    result = run(req.message, callbacks=[handler], session_id=req.session_id)
    return ChatResponse(**result, trace_id=handler.last_trace_id)


@app.get("/health")
def health():
    return {"status": "ok", "langfuse": langfuse.auth_check()}
