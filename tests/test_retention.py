import os
import uuid
from contextlib import contextmanager
from datetime import timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.ai.checkpointer import open_checkpointer
from app.ai.purge import purge
from app.ai.sessions import NullSessionStore, make_session_store
from app.bootstrap import run
from app.dependencies import get_chat_service
from app.errors import StorageUnavailableError
from app.main import app
from app.services.chat_service import ChatService
from tests.test_checkpointer import graph_on, postgres, settings_for  # noqa: F401  (postgres is a fixture)
from tests.test_sessions import grounded, make_graph, texts

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://acme:acme@localhost:5433/acme_support")
RETENTION = timedelta(days=30)


def sql(query: str, *params):
    with psycopg.connect(TEST_DB_URL, autocommit=True) as conn:
        cur = conn.execute(query, params)
        return cur.fetchall() if cur.description else None


def age(session_id: str, days: int) -> None:
    sql(
        "update chat_sessions set last_active_at = now() - make_interval(days => %s) where session_id = %s",
        days,
        session_id,
    )


def last_active(session_id: str):
    return sql("select last_active_at from chat_sessions where session_id = %s", session_id)[0][0]


def has_row(session_id: str) -> bool:
    return bool(sql("select 1 from chat_sessions where session_id = %s", session_id))


def checkpoint_rows(session_id: str) -> int:
    return sql("select count(*) from checkpoints where thread_id = %s", session_id)[0][0]


@pytest.fixture
def ids(postgres):  # noqa: F811
    made = [str(uuid.uuid4()) for _ in range(3)]
    yield made
    with open_checkpointer(postgres) as saver:
        for sid in made:
            saver.delete_thread(sid)
    sql("delete from chat_sessions where session_id = any(%s)", made)


@pytest.fixture
def store(postgres):  # noqa: F811
    with open_checkpointer(postgres) as saver:
        yield saver, make_session_store(saver)


# --- Postgres store -------------------------------------------------------------


@pytest.mark.integration
def test_touch_creates_the_session_row(store, ids):
    _, sessions = store
    sessions.touch(ids[0])
    assert has_row(ids[0])


@pytest.mark.integration
def test_touching_twice_within_a_minute_does_not_rewrite_the_row(store, ids):
    _, sessions = store
    sessions.touch(ids[0])
    first = last_active(ids[0])
    sessions.touch(ids[0])
    assert last_active(ids[0]) == first


@pytest.mark.integration
def test_touching_an_idle_session_resets_its_clock(store, ids):
    _, sessions = store
    sessions.touch(ids[0])
    age(ids[0], 10)
    aged = last_active(ids[0])
    sessions.touch(ids[0])
    assert last_active(ids[0]) > aged


@pytest.mark.integration
def test_purge_deletes_expired_sessions_and_their_conversations_but_keeps_active_ones(store, ids):
    saver, sessions = store
    old, fresh, _ = ids
    for sid in (old, fresh):
        graph, _llm = graph_on(saver, grounded("an answer"))
        sessions.touch(sid)
        run(graph, f"question for {sid[:8]}", session_id=sid)
    age(old, 40)

    assert sessions.purge_expired(RETENTION) == 1

    assert not has_row(old)
    assert checkpoint_rows(old) == 0  # the messages are really gone
    assert has_row(fresh)
    assert checkpoint_rows(fresh) > 0

    graph, llm = graph_on(saver, grounded("later answer"))
    run(graph, "a later question", session_id=fresh)
    assert f"question for {fresh[:8]}" in texts(llm.seen[0])  # the active conversation is intact


@pytest.mark.integration
def test_a_session_used_while_the_purge_runs_is_not_destroyed(store, ids, monkeypatch):
    saver, sessions = store
    graph, _llm = graph_on(saver, grounded("an answer"))
    sessions.touch(ids[0])
    run(graph, "a conversation in progress", session_id=ids[0])
    age(ids[0], 40)

    real_connection, fired = sessions._pool.connection, {"done": False}

    class ConnectionSpy:
        """Passes calls through, and after the first query (whichever connection runs it) the user comes back."""

        def __init__(self, conn):
            self._conn = conn

        def execute(self, *args, **kwargs):
            cursor = self._conn.execute(*args, **kwargs)
            if not fired["done"]:  # right after the "which sessions expired?" query
                fired["done"] = True
                sql("update chat_sessions set last_active_at = now() where session_id = %s", ids[0])
            return cursor

    @contextmanager
    def connection():
        with real_connection() as conn:
            yield ConnectionSpy(conn)

    monkeypatch.setattr(sessions._pool, "connection", connection)

    assert sessions.purge_expired(RETENTION) == 0
    assert fired["done"]  # the simulated return really happened mid-purge
    assert has_row(ids[0])
    assert checkpoint_rows(ids[0]) > 0


@pytest.mark.integration
def test_a_session_at_exactly_the_boundary_is_kept(store, ids):
    _, sessions = store
    sessions.touch(ids[0])
    age(ids[0], 29)
    assert sessions.purge_expired(RETENTION) == 0
    assert has_row(ids[0])


@pytest.mark.integration
def test_purging_twice_is_safe(store, ids):
    saver, sessions = store
    graph, _llm = graph_on(saver, grounded("an answer"))
    sessions.touch(ids[0])
    run(graph, "hello", session_id=ids[0])
    age(ids[0], 40)
    assert sessions.purge_expired(RETENTION) == 1
    assert sessions.purge_expired(RETENTION) == 0


@pytest.mark.integration
def test_conversations_that_predate_the_table_are_adopted_not_deleted(postgres, ids):  # noqa: F811
    with open_checkpointer(postgres) as saver:
        graph, _llm = graph_on(saver, grounded("an answer"))
        run(graph, "an old conversation", session_id=ids[0])  # no touch: no session row yet
        assert not has_row(ids[0])

        sessions = make_session_store(saver)  # what happens on every app start
        assert has_row(ids[0])  # adopted, clock starts now
        assert sessions.purge_expired(RETENTION) == 0
        assert checkpoint_rows(ids[0]) > 0

        age(ids[0], 40)
        assert sessions.purge_expired(RETENTION) == 1


@pytest.mark.integration
def test_the_purge_job_end_to_end(postgres, ids):  # noqa: F811
    with open_checkpointer(postgres) as saver:
        graph, _llm = graph_on(saver, grounded("an answer"))
        make_session_store(saver).touch(ids[0])
        run(graph, "hello", session_id=ids[0])
    age(ids[0], 40)

    assert purge(postgres.model_copy(update={"session_retention_days": 30})) >= 1
    assert not has_row(ids[0])
    assert checkpoint_rows(ids[0]) == 0


# --- no database ----------------------------------------------------------------


def test_the_null_store_has_nothing_to_track_or_purge():
    store = NullSessionStore()
    store.touch("anything")
    assert store.purge_expired(RETENTION) == 0


def test_purge_without_a_database_is_a_no_op():
    assert purge(settings_for(None)) == 0


def test_make_session_store_is_a_no_op_for_in_memory_checkpointers():
    with open_checkpointer(settings_for(None)) as saver:
        assert isinstance(make_session_store(saver), NullSessionStore)


# --- the chat flow touches the session -------------------------------------------


class SpySessions:
    def __init__(self, fail_with: Exception | None = None):
        self.touched: list[str] = []
        self.fail_with = fail_with

    def touch(self, session_id: str) -> None:
        if self.fail_with:
            raise self.fail_with
        self.touched.append(session_id)

    def purge_expired(self, older_than):
        return 0


@pytest.fixture
def client():
    def make(replies, sessions) -> TestClient:
        graph, _ = make_graph(replies)
        app.dependency_overrides[get_chat_service] = lambda: ChatService(graph, sessions)
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_chat_touches_a_generated_session_id_and_returns_it(client):
    spy = SpySessions()
    body = client(grounded("ok"), spy).post("/chat", json={"message": "status of A-1001"}).json()
    assert spy.touched == [body["session_id"]]


def test_chat_touches_the_session_id_the_client_sent(client):
    spy, sid = SpySessions(), str(uuid.uuid4())
    client(grounded("ok"), spy).post("/chat", json={"message": "status of A-1001", "session_id": sid})
    assert spy.touched == [sid]


def test_a_session_store_outage_is_a_clean_503(client):
    spy = SpySessions(fail_with=psycopg.OperationalError("connection refused"))
    r = client(grounded("ok"), spy).post("/chat", json={"message": "status of A-1001"})
    assert (r.status_code, r.json()["error"]["code"]) == (503, StorageUnavailableError.code)
