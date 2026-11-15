from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.ai.guardrails import make_scope_checker
from app.ai.llm import build_llm
from app.ai.rag.store import get_vector_store
from app.ai.tools.knowledge import make_search_tool
from app.ai.tools.orders import ORDERS, make_order_tool
from app.config import Settings
from app.graph import build_graph


def build_agent(settings: Settings) -> CompiledStateGraph:
    store = get_vector_store(settings)
    tools = [
        make_search_tool(store, settings.retrieval_k, settings.retrieval_score_threshold),
        make_order_tool(ORDERS),
    ]
    is_in_scope = make_scope_checker(store, settings.scope_score_threshold)
    return build_graph(build_llm(settings), tools, is_in_scope)


def run(
    graph: CompiledStateGraph,
    question: str,
    callbacks: list | None = None,
    session_id: str | None = None,
) -> dict:
    """Single entrypoint shared by the API and the eval runner."""
    config: RunnableConfig = {"callbacks": callbacks or []}
    if session_id:
        config["metadata"] = {"langfuse_session_id": session_id}

    out = graph.invoke({"messages": [("user", question)]}, config=config)

    tools_used = [tc["name"] for m in out["messages"] for tc in getattr(m, "tool_calls", []) or []]
    return {"answer": out["messages"][-1].content, "tools_used": tools_used}
