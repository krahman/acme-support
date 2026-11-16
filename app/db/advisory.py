import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

log = logging.getLogger("genai.db")


@contextmanager
def advisory_lock(pool: ConnectionPool, name: str, timeout: float = 120.0) -> Iterator[None]:
    """Hold a named local shared by every replica using this database. Others wait until it is released.

    Waiters poll with a non-blocking try-lock instead of sleeping inside a blocking `pg_advisory_lock`. A statement
    blocked in the database is an open transaction, and a migration's `CREATE INDEX CONCURRENTLY` waits for every
    open transaction to finish, so blocked waiters and the lock holder would wait on each other forever.
    """
    key = "hashtextextended(%s, 0)"
    deadline = time.monotonic() + timeout

    with pool.connection() as conn:
        cur = conn.cursor(row_factory=tuple_row)
        while not cur.execute(f"select pg_try_advisory_lock({key})", (name,)).fetchone()[0]:
            if time.monotonic() > deadline:
                raise TimeoutError(f"another replica has held the lock {name!r} for too long")
            time.sleep(0.2)
        try:
            yield
        finally:
            try:
                cur.execute(f"select pg_advisory_unlock({key})", (name,))
            except psycopg.Error:
                log.warning("could not release the lock %r", name, exc_info=True)
                conn.close()  # a closed connection drops its lock


def schema_lock(pool: ConnectionPool):
    """Serialize schema setup across replicas that start at the same moment.
    `create table if not exists` is not safe to run concurrently: two sessions can both pass the "does it exist"
    check and then collide, Whoever holds this locks sets up; the others wait, then find it done.
    """
    return advisory_lock(pool, "acme-schema-setup")
