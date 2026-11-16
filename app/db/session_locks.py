import logging
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

import psycopg
from psycopg_pool import ConnectionPool

from app.config import Settings
from app.errors import SessionBusyError

log = logging.getLogger("genai.locks")


class SessionLocks(Protocol):
    def hold(self, session_id: str) -> AbstractContextManager:
        """Hold the session for one turn. Raises SessionBusyError at once if another turn already holds it."""


class InProcessLocks:
    """For in-memory mode: serializes turns inside this one process."""

    def __init__(self) -> None:
        self._busy: set[str] = set()
        self._guard = threading.Lock()

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        with self._guard:
            if session_id in self._busy:
                raise SessionBusyError()
            self._busy.add(session_id)
        try:
            yield
        finally:
            with self._guard:
                self._busy.discard(session_id)


class PostgresAdvisoryLocks:
    """A Postgres advisory lock per session: works across processes and replicas, and a crashed holder frees it."""

    _LOCK = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))"
    _UNLOCK = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        key = f"chat-session:{session_id}"  # the prefix keeps us clear of any other advisory locks in this database
        with self._pool.connection() as conn:
            if not conn.execute(self._LOCK, (key,)).fetchone()[0]:
                raise SessionBusyError()
            try:
                yield
            finally:
                try:
                    conn.execute(self._UNLOCK, (key,))
                except psycopg.Error:
                    # a dead connection has already dropped its lock; close it so the pool never reuses a doubtful one
                    log.warning("could not release the lock for a session", exc_info=True)
                    conn.close()


@contextmanager
def open_session_locks(settings: Settings) -> Iterator[SessionLocks]:
    """Postgres advisory locks when DATABASE_URL is set, else in-process locks."""

    if settings.database_url is None:
        yield InProcessLocks()
        return

    pool = ConnectionPool(
        settings.database_url.get_secret_value(),
        min_size=1,
        max_size=settings.db_lock_pool_max_size,
        timeout=settings.db_timeout_seconds,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True},
        open=False,
    )
    pool.open(wait=True, timeout=settings.db_timeout_seconds)
    try:
        yield PostgresAdvisoryLocks(pool)
    finally:
        pool.close()
