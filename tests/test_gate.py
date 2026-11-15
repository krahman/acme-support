import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.guardrails import REFUSAL, has_evidence, make_scope_checker
from app.ai.tools.knowledge import NO_RESULTS
from app.ai.tools.orders import ORDERS, make_order_tool
from app.graph import build_graph


class FakeStore:
    """Stands in for InMemoryVectorStore: returns a canned top-1 score."""

    def __init__(self, score: float | None):
        self.score = score

    def similarity_search_with_score(self, text, k=1):
        return [] if self.score is None else [(object(), self.score)]


class FakeLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_msg(content: str, **kw) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="1", **kw)


# --- make_scope_checker ---------------------------------------------------


@pytest.mark.parametrize("text", ["refund for A-1001?", "status of b-2002", "order Z-9999 please"])
def test_order_id_is_in_scope_without_touching_the_store(text):
    assert make_scope_checker(FakeStore(None), 0.2)(text) is True


def test_similar_to_knowledge_base_is_in_scope():
    assert make_scope_checker(FakeStore(0.5), 0.2)("How much is the Team plan?") is True


def test_dissimilar_is_out_of_scope():
    assert make_scope_checker(FakeStore(0.05), 0.2)("capital of France?") is False


def test_score_equal_to_threshold_is_in_scope():
    assert make_scope_checker(FakeStore(0.2), 0.2)("anything") is True


def test_empty_store_is_out_of_scope():
    assert make_scope_checker(FakeStore(None), 0.2)("anything") is False


@pytest.mark.parametrize("text", ["A-100", "A-10012", "AA-1001", "1001"])
def test_malformed_order_ids_do_not_bypass_the_store(text):
    assert make_scope_checker(FakeStore(0.0), 0.2)(text) is False


# --- has_evidence ---------------------------------------------------------

H = HumanMessage("q")


def test_tool_data_counts_as_evidence():
    assert has_evidence([H, tool_msg("Refunds within 14 days")]) is True


def test_no_results_is_not_evidence():
    assert has_evidence([H, tool_msg(NO_RESULTS)]) is False


def test_error_json_is_not_evidence():
    assert has_evidence([H, tool_msg('{"error": "order not found"}')]) is False


def test_error_status_is_not_evidence():
    assert has_evidence([H, tool_msg("boom", status="error")]) is False


def test_answer_without_tools_is_not_evidence():
    assert has_evidence([H, AIMessage(content="The capital is Madrid")]) is False


def test_tool_data_from_an_earlier_turn_is_not_evidence():
    assert has_evidence([H, tool_msg("old data"), AIMessage(content="a"), HumanMessage("q2")]) is False


def test_one_good_tool_among_failures_is_enough():
    assert has_evidence([H, tool_msg(NO_RESULTS), tool_msg("Order A-1001 delivered")]) is True


# --- graph wiring (fake LLM, no network) ----------------------------------


def make_graph(replies, in_scope):
    return build_graph(FakeLLM(messages=iter(replies)), [make_order_tool(ORDERS)], lambda _: in_scope)


def ask(graph, q="hi"):
    return graph.invoke({"messages": [("user", q)]})["messages"]


def test_out_of_scope_is_refused_and_llm_is_never_called():
    graph = make_graph([], in_scope=False)  # an empty iterator would blow up if invoked
    assert ask(graph)[-1].content == REFUSAL


def test_ungrounded_answer_is_replaced_by_refusal():
    graph = make_graph([AIMessage(content="The capital of Spain is Madrid")], in_scope=True)
    assert ask(graph)[-1].content == REFUSAL


def test_grounded_answer_is_kept():
    call = AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "A-1001"}, "id": "c1"}])
    graph = make_graph([call, AIMessage(content="Order A-1001 was delivered.")], in_scope=True)
    assert ask(graph)[-1].content == "Order A-1001 was delivered."
