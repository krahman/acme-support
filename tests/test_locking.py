import os
import threading
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.locks import InProcessLocks, open_session_locks
from app.ai.tools.orders import ORDERS, make_order_tool
from app.api.schema import ChatRequest, ChatResponse
from app.dependencies import get_chat_service
from app.errors import SessionBusyError, StorageUnavailableError
from app.graph import build_graph
from app.main import app
from app.services.chat_service import ChatService
from tests.test_checkpointer import postgres, settings_for  # noqa: F401  (postgres is a fixture)
from tests.test_sessions import lookup

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://acme:acme@localhost:5433/acme_support")


@pytest.fixture(params=["in-process", pytest.param("postgres", marks=pytest.mark.integration)])
def two_locks(request):
    """Two lock objects over the same backing store: like two app replicas (or two workers) sharing one database.

    Two separate Postgres pools matter: advisory locks are re-entrant on one connection, so a leaked lock would
    go unnoticed if the same pool handed the same connection straight back.
    """
    if request.param == "in-process":
        shared = InProcessLocks()
        yield shared, shared
        return
    settings = request.getfixturevalue("postgres")  # skips the test when no Postgres is reachable
    with open_session_locks(settings) as a, open_session_locks(settings) as b:
        yield a, b


# --- the lock itself, against both implementations ------------------------------


def test_a_session_that_is_busy_rejects_a_second_turn(two_locks):
    a, b = two_locks
    sid = str(uuid.uuid4())
    with a.hold(sid), pytest.raises(SessionBusyError), b.hold(sid):
        pass


def test_the_lock_is_free_again_after_the_turn(two_locks):
    a, b = two_locks
    sid = str(uuid.uuid4())
    with a.hold(sid):
        pass
    with b.hold(sid):
        pass


def test_the_lock_is_free_again_after_a_failed_turn(two_locks):
    a, b = two_locks
    sid = str(uuid.uuid4())
    with pytest.raises(RuntimeError), a.hold(sid):
        raise RuntimeError("the turn blew up")
    with b.hold(sid):
        pass


def test_different_sessions_do_not_block_each_other(two_locks):
    a, b = two_locks
    with a.hold(str(uuid.uuid4())), b.hold(str(uuid.uuid4())):
        pass


@pytest.mark.integration
def test_a_crashed_holder_frees_the_lock(postgres):  # noqa: F811
    sid = str(uuid.uuid4())
    with open_session_locks(postgres) as a, open_session_locks(postgres) as b, a.hold(sid):
        with psycopg.connect(TEST_DB_URL, autocommit=True) as admin:  # the holder's connection dies
            admin.execute(
                "select pg_terminate_backend(pid) from pg_locks "
                "where locktype = 'advisory' and granted and pid <> pg_backend_pid()"
            )
        with b.hold(sid):  # nobody is stuck waiting for a lock whose owner is gone
            pass


# --- the lock inside the chat flow ----------------------------------------------


class BlockingLLM:
    """Holds the first model call open until released, so a turn can be 'in progress' while we poke at it."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        if messages[-1].type == "tool":
            return AIMessage(content="done")
        self.entered.set()
        assert self.release.wait(5), "the test never released the turn"
        return lookup()


def chat_in_thread(service, request, results):
    def target():
        try:
            results.append(service.chat(request))
        except Exception as e:  # noqa: BLE001  (hand any failure back to the test)
            results.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    return thread


def test_a_message_sent_during_a_running_turn_is_rejected_and_nothing_is_lost(two_locks):
    a, b = two_locks
    llm = BlockingLLM()
    graph = build_graph(llm, [make_order_tool(ORDERS)], lambda _: True, InMemorySaver())
    sid = uuid.uuid4()
    first, second = ChatService(graph, locks=a), ChatService(graph, locks=b)

    results: list = []
    thread = chat_in_thread(first, ChatRequest(message="status of A-1001", session_id=sid), results)
    assert llm.entered.wait(5)  # turn one is now running and holds the session

    with pytest.raises(SessionBusyError):
        second.chat(ChatRequest(message="a second message", session_id=sid))

    llm.release.set()
    thread.join(5)
    assert isinstance(results[0], ChatResponse)

    retry = second.chat(ChatRequest(message="a second message", session_id=sid))  # free again: the retry works
    assert retry.session_id == str(sid)

    state = graph.get_state({"configurable": {"thread_id": str(sid)}}).values["messages"]
    assert [m.content for m in state if m.type == "human"] == ["status of A-1001", "a second message"]


def test_two_different_sessions_really_run_at_the_same_time(two_locks):
    a, b = two_locks
    barrier = threading.Barrier(2)

    class MeetingLLM:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            if messages[-1].type == "tool":
                return AIMessage(content="done")
            barrier.wait(5)  # only passes if both turns are inside the model at once
            return lookup()

    graph = build_graph(MeetingLLM(), [make_order_tool(ORDERS)], lambda _: True, InMemorySaver())
    results: list = []
    threads = [
        chat_in_thread(ChatService(graph, locks=locks), ChatRequest(message="status of A-1001"), results)
        for locks in (a, b)
    ]
    for t in threads:
        t.join(10)
    assert [type(r) for r in results] == [ChatResponse, ChatResponse]


# --- how the API reports it -----------------------------------------------------


class FailingLocks:
    def __init__(self, error: Exception):
        self.error = error

    def hold(self, session_id: str):
        raise self.error


@pytest.fixture
def client():
    def make(locks) -> TestClient:
        graph = build_graph(BlockingLLM(), [make_order_tool(ORDERS)], lambda _: True, InMemorySaver())
        app.dependency_overrides[get_chat_service] = lambda: ChatService(graph, locks=locks)
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_a_busy_session_is_a_409_with_a_clear_code(client):
    r = client(FailingLocks(SessionBusyError())).post("/chat", json={"message": "hi"})
    assert (r.status_code, r.json()["error"]["code"]) == (409, SessionBusyError.code)


def test_a_lock_store_outage_is_a_503(client):
    r = client(FailingLocks(psycopg.OperationalError("connection refused"))).post("/chat", json={"message": "hi"})
    assert (r.status_code, r.json()["error"]["code"]) == (503, StorageUnavailableError.code)
