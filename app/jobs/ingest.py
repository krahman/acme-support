"""Build the vector index ahead of time (e.g. in a deploy step): python -m app.jobs.ingest"""

from app.config import get_settings
from app.db.pgvector import open_vector_store
from app.logging import setup_logging

if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    with open_vector_store(settings):
        pass
