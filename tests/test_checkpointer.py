import os
import uuid

import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.checkpointer import open_checkpointer
from app.ai.tools.orders import ORDERS, make_order_tool
from app.bootstrap import run
from app.config import Settings
from app.graph import build_graph
from tests.test_sessions import ScriptedLLM, grounded, texts

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://acme:acme@localhost:5433/acme_support")


def settings_for(database_url: str | None) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="x",
        langfuse_public_key="x",
        langfuse_secret_key="x",
        database_url=database_url,
    )


def test_without_a_database_url_the_checkpointer_is_in_memory():
    with open_checkpointer(settings_for(None)) as saver:
        assert isinstance(saver, InMemorySaver)


@pytest.fixture
def postgres():
    try:
        psycopg.connect(TEST_DB_URL, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip("no Postgres reachable: run `docker compose up -d`")
    return settings_for(TEST_DB_URL)


@pytest.fixture
def session_id(postgres):
    sid = str(uuid.uuid4())
    yield sid
    with open_checkpointer(postgres) as saver:
        saver.delete_thread(sid)  # leave no test conversations behind


def graph_on(saver, replies):
    llm = ScriptedLLM(replies)
    return build_graph(llm, [make_order_tool(ORDERS)], lambda _: True, saver), llm


@pytest.mark.integration
def test_a_conversation_survives_a_restart(postgres, session_id):
    with open_checkpointer(postgres) as saver:  # "process 1"
        graph, _ = graph_on(saver, grounded("first answer"))
        run(graph, "Can I get a refund for A-1001?", session_id=session_id)

    with open_checkpointer(postgres) as saver:  # "process 2": new pool, new graph, same database
        graph, llm = graph_on(saver, grounded("second answer"))
        run(graph, "and the other one?", session_id=session_id)

    assert "Can I get a refund for A-1001?" in texts(llm.seen[0])


@pytest.mark.integration
def test_the_pool_recovers_after_the_database_drops_its_connections(postgres, session_id):
    """Reproduces a DB restart: the server kills every pooled connection, and the next turn must still work."""
    with open_checkpointer(postgres) as saver:
        graph, _ = graph_on(saver, grounded("first answer"))
        run(graph, "question before the outage", session_id=session_id)

        with psycopg.connect(TEST_DB_URL, autocommit=True) as admin:
            admin.execute(
                "select pg_terminate_backend(pid) from pg_stat_activity "
                "where datname = current_database() and pid <> pg_backend_pid()"
            )

        graph, llm = graph_on(saver, grounded("second answer"))
        run(graph, "question after the outage", session_id=session_id)

    assert "question before the outage" in texts(llm.seen[0])


@pytest.mark.integration
def test_deleting_a_thread_forgets_the_conversation(postgres, session_id):
    with open_checkpointer(postgres) as saver:
        graph, _ = graph_on(saver, grounded("first answer"))
        run(graph, "remember this question", session_id=session_id)
        saver.delete_thread(session_id)

        graph, llm = graph_on(saver, grounded("after delete"))
        run(graph, "a new question", session_id=session_id)

    assert "remember this question" not in texts(llm.seen[0])


@pytest.mark.integration
def test_sessions_are_isolated_in_postgres(postgres, session_id):
    other = str(uuid.uuid4())
    try:
        with open_checkpointer(postgres) as saver:
            graph, llm = graph_on(saver, grounded("a") + grounded("b"))
            run(graph, "private question from session one", session_id=session_id)
            run(graph, "hello from session two", session_id=other)
        assert "private question from session one" not in texts(llm.seen[2])
    finally:
        with open_checkpointer(postgres) as saver:
            saver.delete_thread(other)


def test_startup_fails_fast_when_the_database_is_unreachable():
    unreachable = settings_for("postgresql://acme:acme@127.0.0.1:1/none").model_copy(update={"db_timeout_seconds": 1})
    with pytest.raises(Exception) as exc, open_checkpointer(unreachable):
        pass
    assert "PoolTimeout" in type(exc.value).__name__
