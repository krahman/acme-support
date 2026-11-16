from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.db.advisory import schema_lock


@contextmanager
def open_checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver]:
    """Postgres when DATABASE_URL is set (sessions survice restarts), else in-memory (dev, tests, evals)."""
    if settings.database_url is None:
        yield InMemorySaver()
        return

    pool = ConnectionPool(
        settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_timeout_seconds,  # a request gives up (503) instead of hanging while the DB is down
        check=ConnectionPool.check_connection,  # drop connections that died (DB restart) instead of handing them out
        # required by PostgresSaver: it manages its own transactions and reads rows by column name
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )

    pool.open(wait=True, timeout=settings.db_timeout_seconds)  # fail fast at startup if the DB is down
    try:
        saver = PostgresSaver(pool)
        with schema_lock(pool):  # replicas starting together must not run the migration at the same time
            saver.setup()
        yield saver
    finally:
        pool.close()
