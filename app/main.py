"""FastAPI service with Langfuse tracing + basic observability."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse

from app.api.routes import chat, health
from app.bootstrap import build_agent
from app.config import get_settings
from app.logging import setup_logging
from app.middleware import AccessLogMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.langfuse = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    app.state.graph = build_agent(settings)
    yield
    app.state.langfuse.flush()  # don't lose buffered traces on shutdown


app = FastAPI(title="Acme Support GenAI", lifespan=lifespan)

app.add_middleware(AccessLogMiddleware)
app.include_router(chat.router)
app.include_router(health.router)
