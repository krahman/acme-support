"""Delete conversation idle longer than the retention period. run it on a schedule: python -m app.jobs.purge"""

import logging
from datetime import timedelta

from app.config import Settings, get_settings
from app.db.checkpointer import open_checkpointer
from app.db.pgvector import open_kb_pool, purge_retired_indexes
from app.db.sessions import make_session_store
from app.logging import setup_logging

log = logging.getLogger("genai.purge")


def purge(settings: Settings) -> int:
    with open_checkpointer(settings) as checkpointer:
        return make_session_store(checkpointer).purge_expired(timedelta(days=settings.session_retention_days))


def purge_indexes(settings: Settings) -> int:
    if settings.database_url is None:
        return 0
    with open_kb_pool(settings) as pool:
        return purge_retired_indexes(pool, settings.kb_indexes_to_keep)


if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info(
        "purged %d sessions idle for more than %d days",
        purge(settings),
        settings.session_retention_days,
    )
    log.info(
        "purged %d retired knowledge-base indexes (kept the newest %d)",
        purge_indexes(settings),
        settings.kb_indexes_to_keep,
    )
