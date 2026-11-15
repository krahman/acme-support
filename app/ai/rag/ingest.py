"""Build the vector index ahead of time (e.g. in a deploy step): python -m app.ai.rag.ingest"""

from app.ai.rag.store import get_vector_store
from app.config import get_settings
from app.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
get_vector_store(settings)
