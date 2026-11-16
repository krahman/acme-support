"""Delete conversation idle longer than the retention period. run it on a schedule: python -m app.ai.purge"""

import logging
from datetime import timedelta

from app.ai.checkpointer import open_checkpointer
from app.ai.sessions import make_session_store
from app.config import Settings, get_settings
from app.logging import setup_logging

log = logging.getLogger("genai.purge")


def purge(settings: Settings) -> int:
    with open_checkpointer(settings) as checkpointer:
        return make_session_store(checkpointer).purge_expired(timedelta(days=settings.session_retention_days))


if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("purged %d sessions idel for more than %d days", purge(settings), settings.session_retention_days)
