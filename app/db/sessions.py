from datetime import timedelta
from typing import Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver

from app.db.advisory import schema_lock


class SessionStore(Protocol):
    def touch(self, session_id: str) -> None:
        """Record that the session was just used (starts or resets its retention clock)."""

    def purge_expired(self, older_than: timedelta) -> int:
        """Delete sessions idle for longer than 'older_than', with their conversations. Returns how many."""


class NullSessionStore:
    """In-memory sessions die with the process, so there is nothing to track or purge."""

    def touch(self, session_id: str) -> None:
        pass

    def purge_expired(self, older_than: timedelta) -> int:
        return 0


class PostgresSessionStore:
    """One row per conversation: when it was last used. LangGraph owns the messages; this owns their lifetime."""

    def __init__(self, saver: PostgresSaver) -> None:
        self._saver = saver
        self._pool = saver.conn

    def setup(self) -> None:
        with schema_lock(self._pool), self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id text PRIMARY KEY,
                    created_at timestamptz NOT NULL DEFAULT NOW(),
                    last_active_at timestamptz NOT NULL DEFAULT NOW()
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_sessions_last_active_idx ON chat_sessions (last_active_at)
                """
            )
            # adopt conversations that predate this table: their retention clock starts now
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id)
                SELECT DISTINCT thread_id FROM checkpoints ON CONFLICT (session_id) DO NOTHING
                """
            )

    def touch(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id)
                values (%s) ON CONFLICT (session_id) do
                update set last_active_at = NOW()
                WHERE chat_sessions.last_active_at < NOW() - interval '1 minute'
                """,  # the WHERE skips the write when it was touched moments ago: one row update per minute at most
                (session_id,),
            )

    def purge_expired(self, older_than: timedelta) -> int:
        with self._pool.connection() as conn:
            expired = [
                r["session_id"]
                for r in conn.execute(
                    """
                    SELECT session_id
                    FROM chat_sessions
                    WHERE last_active_at < NOW() - %s
                    """,
                    (older_than,),
                )
            ]

            purged = 0
            for session_id in expired:
                with self._pool.connection() as conn:
                    # claim first, re-checking the age:
                    # a session used since the list above is skipped, never destroyed
                    claimed = conn.execute(
                        """
                        DELETE FROM chat_sessions
                        WHERE session_id = %s AND last_active_at < NOW() - %s RETURNING session_id
                        """,
                        (
                            session_id,
                            older_than,
                        ),
                    ).fetchone()
                if claimed:
                    self._saver.delete_thread(session_id)
                    purged += 1
            return purged


def make_session_store(checkpointer: BaseCheckpointSaver) -> SessionStore:
    if isinstance(checkpointer, PostgresSaver):
        store = PostgresSessionStore(checkpointer)
        store.setup()
        return store
    return NullSessionStore()
