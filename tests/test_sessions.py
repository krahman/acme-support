import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.ai.guardrails import REFUSAL
from app.ai.tools.orders import ORDERS, make_order_tool
from app.bootstrap import run
from app.dependencies import get_chat_service
from app.graph import build_graph
from app.main import app
from app.services.chat_service import ChatService


class ScriptedLLM:
    """Replays canned replies; records every message list it was sent and the tool_choice used."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen: list[list] = []
        self.tool_choices: list[str | None] = []

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return BoundLLM(self, tool_choice)


class BoundLLM:
    def __init__(self, parent: ScriptedLLM, tool_choice: str | None):
        self.parent = parent
        self.tool_choice = tool_choice

    def invoke(self, messages):
        self.parent.seen.append(messages)
        self.parent.tool_choices.append(self.tool_choice)
        return next(self.parent.replies)


def lookup(order_id="A-1001"):
    return AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": order_id}, "id": "c"}])


def grounded(answer):
    """One tool call, then a final answer: what a well-behaved turn looks like."""
    return [lookup(), AIMessage(content=answer)]


def make_graph(replies, in_scope=lambda _: True, max_history=20):
    llm = ScriptedLLM(replies)
    graph = build_graph(llm, [make_order_tool(ORDERS)], in_scope, InMemorySaver(), max_history)
    return graph, llm


def texts(messages):
    return [str(m.content) for m in messages]


def test_a_session_remembers_earlier_turns():
    graph, llm = make_graph(grounded("first answer") + grounded("second answer"))
    run(graph, "Can I get a refund for A-1001?", session_id="s1")
    run(graph, "and the other one?", session_id="s1")
    first_call_of_turn_2 = llm.seen[2]
    assert "Can I get a refund for A-1001?" in texts(first_call_of_turn_2)
    assert "first answer" in texts(first_call_of_turn_2)


def test_sessions_are_isolated_from_each_other():
    graph, llm = make_graph(grounded("a") + grounded("b"))
    run(graph, "secret question from session A", session_id="A")
    run(graph, "hello from session B", session_id="B")
    assert "secret question from session A" not in texts(llm.seen[2])


def test_without_a_session_id_each_call_is_a_fresh_conversation():
    graph, llm = make_graph(grounded("a") + grounded("b"))
    one = run(graph, "first", session_id=None)
    two = run(graph, "second", session_id=None)
    assert one["session_id"] != two["session_id"]
    uuid.UUID(one["session_id"])  # a valid id the client can send back
    assert "first" not in texts(llm.seen[2])


def test_tools_used_only_reports_the_current_turn():
    graph, _ = make_graph(grounded("a") + grounded("b"))
    run(graph, "turn one", session_id="s")
    second = run(graph, "turn two", session_id="s")
    assert second["tools_used"] == ["get_order_status"]  # not two entries


def test_the_first_model_call_of_every_turn_must_use_a_tool():
    graph, llm = make_graph(grounded("a") + grounded("b"))
    run(graph, "turn one", session_id="s")
    run(graph, "turn two", session_id="s")
    # per turn: forced tool call first, then a free final answer
    assert llm.tool_choices == ["required", None, "required", None]


def test_the_guard_still_refuses_if_the_model_ignores_the_forced_tool():
    graph, _ = make_graph(grounded("It was delivered.") + [AIMessage(content="As I said, it was delivered.")])
    run(graph, "status of A-1001", session_id="s")
    out = run(graph, "what did you just say?", session_id="s")
    assert out["answer"] == REFUSAL  # every answer needs evidence from the current turn


def test_gate_judges_a_follow_up_together_with_the_previous_question():
    graph, llm = make_graph(grounded("a") + grounded("b"), in_scope=lambda text: "A-1001" in text)
    run(graph, "status of A-1001", session_id="s")
    out = run(graph, "yes please", session_id="s")
    assert out["answer"] == "b"

    fresh, fresh_llm = make_graph([], in_scope=lambda text: "A-1001" in text)
    assert run(fresh, "yes please", session_id="new")["answer"] == REFUSAL
    assert fresh_llm.seen == []  # blocked before the model


def test_history_sent_to_the_model_is_trimmed_and_starts_on_a_user_message():
    graph, llm = make_graph([r for i in range(6) for r in grounded(f"answer {i}")], max_history=4)
    for i in range(6):
        run(graph, f"question {i}", session_id="long")
    for sent in llm.seen:
        history = sent[1:]  # [0] is the system prompt
        assert len(history) <= 4
        assert history[0].type == "human"  # never starts on an orphaned tool result


@pytest.fixture
def client():
    def make(replies) -> tuple[TestClient, ScriptedLLM]:
        graph, llm = make_graph(replies)
        app.dependency_overrides[get_chat_service] = lambda: ChatService(graph)
        return TestClient(app), llm  # no `with`: skips lifespan

    yield make
    app.dependency_overrides.clear()


def test_api_returns_a_session_id_that_continues_the_conversation(client):
    c, llm = client(grounded("one") + grounded("two"))
    first = c.post("/chat", json={"message": "status of A-1001"}).json()
    sid = first["session_id"]
    second = c.post("/chat", json={"message": "and again?", "session_id": sid}).json()
    assert second["session_id"] == sid
    assert "status of A-1001" in texts(llm.seen[2])


@pytest.mark.parametrize("bad", ["alice", "1", "s" * 65, "12345678-1234-1234-1234-123456789012"])
def test_api_rejects_session_ids_that_are_not_uuid4(client, bad):
    c, _ = client([])
    assert c.post("/chat", json={"message": "hi", "session_id": bad}).status_code == 422
