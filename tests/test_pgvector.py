import hashlib
import os
import threading
import time
import uuid
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.guardrails import make_scope_checker
from app.ai.rag.chunks import index_fingerprint
from app.ai.rag.loader import load_documents
from app.ai.rag.store import get_vector_store
from app.ai.tools.orders import ORDERS, make_order_tool
from app.db.advisory import advisory_lock
from app.db.checkpointer import open_checkpointer
from app.db.pgvector import PgVectorStore, ensure_index, open_kb_pool, open_vector_store, purge_retired_indexes
from app.db.sessions import make_session_store
from app.dependencies import get_chat_service
from app.errors import KnowledgeBaseUnavailableError
from app.graph import build_graph
from app.main import app
from app.services.chat_service import ChatService
from tests.test_checkpointer import postgres, settings_for  # noqa: F401  (postgres is a fixture)

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://acme:acme@localhost:5433/acme_support")


class CountingEmbeddings(Embeddings):
    """Deterministic vectors. Counts embedded documents and can be slow, to widen race windows."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.docs_embedded = 0
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        time.sleep(self.delay)
        with self._lock:
            self.docs_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:16]]


@pytest.fixture
def pgvector_db(postgres):  # noqa: F811
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        if not conn.execute("select 1 from pg_available_extensions where name = 'vector'").fetchone():
            pytest.skip("this Postgres has no pgvector: use the pgvector/pgvector:pg17 image (docker compose up -d)")
    return postgres


@pytest.fixture
def kb(tmp_path, pgvector_db):
    """A knowledge base nobody else has: every document carries a unique marker, so its fingerprint is unique too."""
    marker = uuid.uuid4().hex
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    settings = pgvector_db.model_copy(update={"kb_dir": kb_dir, "index_dir": tmp_path / "index"})

    def write(name: str, text: str) -> None:
        (kb_dir / name).write_text(f"{text} [{marker}]")

    write("pricing.md", "The Team plan costs $50 per user per month.")
    write("sla.md", "The Pro plan has a best-effort uptime target.")
    write("refunds.md", "Full refunds are available within 14 days of purchase.")
    yield settings, write, marker
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:  # leave no test indexes behind
        conn.execute(
            "delete from kb_indexes where fingerprint in (select fingerprint from kb_chunks where content like %s)",
            (f"%{marker}%",),
        )


@pytest.fixture
def pool(kb):
    settings, _, _ = kb
    with open_kb_pool(settings) as p:
        yield p


@pytest.fixture
def empty_database(pgvector_db):
    """A brand-new database: the situation of a first deploy with several replicas."""
    parsed = urlparse(TEST_DB_URL)
    name = f"acme_fresh_{uuid.uuid4().hex[:10]}"
    admin = urlunparse(parsed._replace(path="/postgres"))
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"create database {name}")
    yield urlunparse(parsed._replace(path=f"/{name}"))
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"drop database if exists {name} with (force)")


def join_all(threads, timeout):
    """One shared deadline for all threads, so a hung replica fails the test instead of stalling it."""
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))


def count(sql: str, *params) -> int:
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        return conn.execute(sql, params).fetchone()[0]


def chunk_rows(marker: str) -> int:
    return count("select count(*) from kb_chunks where content like %s", f"%{marker}%")


# --- search results -------------------------------------------------------------


@pytest.mark.integration
def test_search_scores_match_the_in_memory_store(kb, pool):
    settings, _, _ = kb
    emb = CountingEmbeddings()
    in_memory = get_vector_store(settings, emb)
    pg = ensure_index(pool, settings, emb)
    for query in ("How much is the Team plan?", "uptime target", "refund within 14 days", "capital of France"):
        expected = in_memory.similarity_search_with_score(query, k=3)
        actual = pg.similarity_search_with_score(query, k=3)
        assert [d.page_content for d, _ in actual] == [d.page_content for d, _ in expected]
        assert [d.metadata for d, _ in actual] == [d.metadata for d, _ in expected]
        assert [s for _, s in actual] == pytest.approx([s for _, s in expected], abs=1e-4)


@pytest.mark.integration
def test_similarity_is_one_minus_distance_so_higher_means_closer(kb, pool):
    settings, _, _ = kb
    pg = ensure_index(pool, settings, CountingEmbeddings())
    exact_text = (settings.kb_dir / "pricing.md").read_text().strip()
    best = pg.similarity_search_with_score(exact_text, k=3)
    assert best[0][1] == pytest.approx(1.0, abs=1e-4)  # identical text: distance 0, similarity 1
    assert [s for _, s in best] == sorted((s for _, s in best), reverse=True)  # closest first
    assert best[0][1] > best[-1][1]


# --- building and sharing the index ---------------------------------------------


@pytest.mark.integration
def test_a_second_replica_reuses_the_index_without_embedding(kb, pool):
    settings, _, marker = kb
    first, second = CountingEmbeddings(), CountingEmbeddings()
    ensure_index(pool, settings, first)
    ensure_index(pool, settings, second)
    assert first.docs_embedded == 3
    assert second.docs_embedded == 0
    assert chunk_rows(marker) == 3


@pytest.mark.integration
def test_replicas_starting_together_embed_the_knowledge_base_once(kb):
    settings, _, marker = kb
    embeddings, errors, stores = CountingEmbeddings(delay=0.5), [], []
    barrier = threading.Barrier(6)

    def replica():
        try:
            barrier.wait(10)
            with open_kb_pool(settings) as pool:
                stores.append(ensure_index(pool, settings, embeddings))
                assert stores[-1].similarity_search_with_score("Team plan", k=1)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=replica, daemon=True) for _ in range(6)]
    [t.start() for t in threads]
    join_all(threads, 60)
    assert not errors
    assert len(stores) == 6
    assert embeddings.docs_embedded == 3  # one builder, not six
    assert chunk_rows(marker) == 3


@pytest.mark.integration
def test_a_failed_build_leaves_nothing_behind_and_the_next_replica_can_build(kb, pool):
    settings, write, marker = kb
    (settings.kb_dir / "sla.md").write_text(f"The second chunk has a character Postgres refuses \x00 [{marker}]")
    broken = index_fingerprint(settings, load_documents(settings.kb_dir))

    with pytest.raises(psycopg.Error):
        ensure_index(pool, settings, CountingEmbeddings())
    assert count("select count(*) from kb_indexes where fingerprint = %s", broken) == 0
    assert count("select count(*) from kb_chunks where fingerprint = %s", broken) == 0  # rolled back, none half-built

    write("sla.md", "The Pro plan has a best-effort uptime target.")  # fixed: the lock was released, so this builds
    assert ensure_index(pool, settings, CountingEmbeddings()).similarity_search_with_score("uptime", k=1)
    assert chunk_rows(marker) == 3


@pytest.mark.integration
def test_a_changed_knowledge_base_gets_a_new_index_and_the_old_one_keeps_working(kb, pool):
    settings, write, marker = kb
    emb = CountingEmbeddings()
    old = ensure_index(pool, settings, emb)
    write("pricing.md", "The Team plan now costs $60 per user per month.")
    new = ensure_index(pool, settings, emb)

    assert new.fingerprint != old.fingerprint
    assert chunk_rows(marker) == 6  # both versions side by side
    assert any("$50" in d.page_content for d, _ in old.similarity_search_with_score("Team plan price", k=3))
    assert any("$60" in d.page_content for d, _ in new.similarity_search_with_score("Team plan price", k=3))


# --- retiring old indexes -------------------------------------------------------


@pytest.mark.integration
def test_purge_keeps_the_newest_indexes_and_removes_the_rest_with_their_chunks(empty_database, tmp_path):
    # a brand-new database: purge works on the whole table, so it must never run against shared data
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    settings = settings_for(empty_database).model_copy(update={"kb_dir": kb_dir})
    fingerprints = []
    with open_kb_pool(settings) as pool:
        for version in range(3):
            (kb_dir / "pricing.md").write_text(f"The Team plan costs ${50 + version} per user per month.")
            fingerprints.append(ensure_index(pool, settings, CountingEmbeddings()).fingerprint)
            time.sleep(0.05)  # distinct creation times

        assert purge_retired_indexes(pool, keep=2) == 1

    with psycopg.connect(empty_database, autocommit=True) as conn:
        remaining = [r[0] for r in conn.execute("select fingerprint from kb_indexes")]
        orphans = conn.execute("select count(*) from kb_chunks where fingerprint = %s", (fingerprints[0],))
        orphan_count = orphans.fetchone()[0]
    assert sorted(remaining) == sorted(fingerprints[1:])  # the oldest is gone, the newest two stay
    assert orphan_count == 0  # its chunks went with it


@pytest.mark.integration
def test_searching_a_purged_index_fails_loudly_instead_of_returning_nothing(kb, pool):
    settings, _, _ = kb
    store = ensure_index(pool, settings, CountingEmbeddings())
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        conn.execute("delete from kb_indexes where fingerprint = %s", (store.fingerprint,))
    with pytest.raises(KnowledgeBaseUnavailableError):
        store.similarity_search_with_score("Team plan", k=1)


# --- choosing the backend -------------------------------------------------------


def test_without_a_database_url_the_local_file_index_is_used(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("The Team plan costs $50 per user per month.")
    settings = settings_for(None).model_copy(update={"kb_dir": kb_dir, "index_dir": tmp_path / "index"})
    with open_vector_store(settings, CountingEmbeddings()) as store:
        assert isinstance(store, InMemoryVectorStore)


@pytest.mark.integration
def test_with_a_database_url_the_pgvector_store_is_used(kb):
    settings, _, _ = kb
    with open_vector_store(settings, CountingEmbeddings()) as store:
        assert isinstance(store, PgVectorStore)
        assert store.similarity_search_with_score("Team plan", k=1)


# --- starting several replicas at once --------------------------------------------


@pytest.mark.integration
def test_replicas_starting_together_on_an_empty_database_all_start(empty_database, tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "pricing.md").write_text("The Team plan costs $50 per user per month.")
    settings = settings_for(empty_database).model_copy(update={"kb_dir": kb_dir})
    embeddings, errors, started = CountingEmbeddings(delay=0.3), [], []
    barrier = threading.Barrier(6)

    def replica():
        try:
            barrier.wait(10)
            with open_checkpointer(settings) as saver, open_vector_store(settings, embeddings) as store:
                make_session_store(saver)
                assert store.similarity_search_with_score("Team plan", k=1)
                started.append(True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=replica, daemon=True) for _ in range(6)]
    [t.start() for t in threads]
    join_all(threads, 90)
    assert not any(t.is_alive() for t in threads), "a replica is stuck: the startup locks deadlocked"
    assert not errors
    assert len(started) == 6
    assert embeddings.docs_embedded == 1  # one document, embedded once across all six replicas


@pytest.mark.integration
def test_replicas_creating_the_vector_schema_at_the_same_instant_do_not_collide(empty_database, tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "pricing.md").write_text("The Team plan costs $50 per user per month.")
    settings = settings_for(empty_database).model_copy(update={"kb_dir": kb_dir})
    errors, barrier = [], threading.Barrier(8)

    def replica():
        try:
            with open_kb_pool(settings) as pool:
                barrier.wait(10)  # all eight reach the schema step together
                ensure_index(pool, settings, CountingEmbeddings())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=replica, daemon=True) for _ in range(8)]
    [t.start() for t in threads]
    join_all(threads, 60)
    assert not errors


@pytest.mark.integration
def test_the_lock_makes_a_second_holder_wait_then_proceed_and_times_out_if_held_too_long(pool):
    order = []

    def second():
        with advisory_lock(pool, "test-lock-wait"):
            order.append("second")

    with advisory_lock(pool, "test-lock-wait"):
        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        time.sleep(0.6)
        assert order == []  # still waiting while the first holds it
        order.append("first")
    thread.join(10)
    assert order == ["first", "second"]

    with (
        advisory_lock(pool, "test-lock-timeout"),
        pytest.raises(TimeoutError),
        advisory_lock(pool, "test-lock-timeout", timeout=0.5),
    ):
        pass


@pytest.mark.integration
def test_a_holder_whose_connection_dies_neither_raises_nor_keeps_the_lock(pool):
    name = f"test-lock-crash-{uuid.uuid4().hex}"
    with advisory_lock(pool, name), psycopg.connect(TEST_DB_URL, autocommit=True) as admin:
        admin.execute(  # kill only the connection holding THIS lock
            "select pg_terminate_backend(pid) from pg_locks where locktype = 'advisory' and granted "
            "and ((classid::bigint << 32) | objid::bigint) = hashtextextended(%s, 0)",
            (name,),
        )
    # leaving the block unlocked a dead connection: that must be handled quietly, not raised
    with advisory_lock(pool, name, timeout=5):  # and the lock is free for the next holder
        pass


# --- how the API reports it -----------------------------------------------------


def test_an_unavailable_knowledge_base_is_a_503():
    class BrokenStore:
        def similarity_search_with_score(self, query, k=4):
            raise KnowledgeBaseUnavailableError()

    class UnusedLLM:
        def bind_tools(self, *args, **kwargs):
            return self

    graph = build_graph(
        UnusedLLM(),
        [make_order_tool(ORDERS)],
        make_scope_checker(BrokenStore(), 0.2),
        InMemorySaver(),
    )
    app.dependency_overrides[get_chat_service] = lambda: ChatService(graph)
    try:
        r = TestClient(app).post("/chat", json={"message": "How much is the Team plan?"})
    finally:
        app.dependency_overrides.clear()
    assert (r.status_code, r.json()["error"]["code"]) == (503, KnowledgeBaseUnavailableError.code)
