import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

from app.ai.rag.chunks import chunk_documents, index_fingerprint
from app.ai.rag.loader import load_documents
from app.ai.rag.store import get_vector_store, make_embeddings
from app.ai.rag.types import SimilarityStore
from app.config import Settings
from app.db.advisory import advisory_lock, schema_lock
from app.errors import KnowledgeBaseUnavailableError

log = logging.getLogger("genai.rag")

SCHEMA = (
    "create extension if not exists vector",
    """
    CREATE TABLE IF NOT EXISTS kb_indexes (
        fingerprint TEXT PRIMARY KEY,
        chunk_count INT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kb_chunks (
        fingerprint TEXT NOT NULL REFERENCES kb_indexes (fingerprint) ON DELETE CASCADE,
        chunk_id INT NOT NULL,
        source TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR NOT NULL,
        PRIMARY KEY (fingerprint, chunk_id)
    )
    """,
)


def _literal(vector: list[float]) -> str:
    """A vector as pgvector's text format, so no driver-side adapter (and no extra dependency) is needed."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgVectorStore:
    """Nearest-chunk search over one index (one knowledge-base fingerprint) in Porstgres."""

    def __init__(self, pool: ConnectionPool, embeddings: Embeddings, fingerprint: str) -> None:
        self._pool = pool
        self._embeddings = embeddings
        self.fingerprint = fingerprint

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        vector = _literal(self._embeddings.embed_query(query))
        with self._pool.connection() as conn:
            cur = conn.cursor(row_factory=tuple_row)
            rows = cur.execute(
                # <=> is cosine DISTANCE (lower is closer). Callers compare similarity (higher is closer), so convert.
                """
                SELECT source, content, 1 - (embedding <=> %s::vector) FROM kb_chunks
                WHERE fingerprint = %s ORDER BY embedding <=> %s::vector limit %s
                """,
                (vector, self.fingerprint, vector, k),
            ).fetchall()
            if not rows and not _index_exists(conn, self.fingerprint):
                # a purged index must fail loudly: an empty result would make the scope gate refuse every question
                raise KnowledgeBaseUnavailableError()
        return [
            (Document(page_content=content, metadata={"source": source}), float(sim)) for source, content, sim in rows
        ]


def _index_exists(conn: psycopg.Connection, fingerprint: str) -> bool:
    cur = conn.cursor(row_factory=tuple_row)
    return cur.execute("select 1 from kb_indexes where fingerprint = %s", (fingerprint,)).fetchone() is not None


def _create_schema(pool: ConnectionPool) -> None:
    with schema_lock(pool), pool.connection() as conn:
        try:
            for statement in SCHEMA:
                conn.execute(statement)
        except (psycopg.errors.FeatureNotSupported, psycopg.errors.UndefinedFile) as e:
            raise RuntimeError(
                "this Postgres has no pgvector extension: use the pgvector/pgvector image locally, "
                "or a managed Postgres that offers pgvector"
            ) from e


def ensure_index(pool: ConnectionPool, settings: Settings, embeddings: Embeddings) -> PgVectorStore:
    docs = load_documents(settings.kb_dir)
    fingerprint = index_fingerprint(settings, docs)
    _create_schema(pool)

    with pool.connection() as conn:
        found = _index_exists(conn, fingerprint)
    if found:
        log.info("found knowledge-base index %s", fingerprint)
    else:
        # one builder at a time per fingerprint: replicas starting together must not all pay to embed it
        with advisory_lock(pool, f"kb-index:{fingerprint}"):
            with pool.connection() as conn:
                found = _index_exists(conn, fingerprint)
            if found:
                log.info("found knowledge-base index %s (built by another replica)", fingerprint)
            else:
                _build(pool, settings, embeddings, docs, fingerprint)
    return PgVectorStore(pool, embeddings, fingerprint)


def _build(
    pool: ConnectionPool,
    settings: Settings,
    embeddings: Embeddings,
    docs: list[Document],
    fingerprint: str,
) -> None:
    chunks = chunk_documents(settings, docs)
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    rows = [
        (fingerprint, i, c.metadata["source"], c.page_content, _literal(v))
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    with pool.connection() as conn, conn.transaction():
        conn.execute("INSERT INTO kb_indexes (fingerprint, chunk_count) VALUES (%s, %s)", (fingerprint, len(rows)))
        conn.cursor().executemany(
            "INSERT INTO kb_chunks (fingerprint, chunk_id, source, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)",
            rows,
        )
    log.info("built knowledge-base index %s (%d chunks)", fingerprint, len(rows))


def purge_retired_indexes(pool: ConnectionPool, keep: int) -> int:
    """Delete all but the newest `keep` indexes (their chunks go with them). Returns how many were removed."""
    with pool.connection() as conn:
        cur = conn.execute(
            "DELETE FROM kb_indexes WHERE fingerprint NOT IN "
            "(SELECT fingerprint FROM kb_indexes ORDER BY created_at DESC LIMIT %s)",
            (keep,),
        )
        return cur.rowcount


@contextmanager
def open_kb_pool(settings: Settings) -> Iterator[ConnectionPool]:
    pool = ConnectionPool(
        settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_timeout_seconds,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True},
        open=False,
    )
    pool.open(wait=True, timeout=settings.db_timeout_seconds)
    try:
        yield pool
    finally:
        pool.close()


@contextmanager
def open_vector_store(settings: Settings, embeddings: Embeddings | None = None) -> Iterator[SimilarityStore]:
    """Postgres + pgvector when DATABASE_URL is set (every replica shares one index), else the local file index."""
    embeddings = embeddings or make_embeddings(settings)
    if settings.database_url is None:
        yield get_vector_store(settings, embeddings)
        return
    with open_kb_pool(settings) as pool:
        yield ensure_index(pool, settings, embeddings)
