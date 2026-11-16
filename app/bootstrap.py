import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.ai.guardrails import current_turn, make_scope_checker
from app.ai.llm import build_llm
from app.ai.rag.types import SimilarityStore
from app.ai.tools.knowledge import make_search_tool
from app.ai.tools.orders import ORDERS, make_order_tool
from app.config import Settings
from app.graph import build_graph


def build_agent(settings: Settings, checkpointer: BaseCheckpointSaver, store: SimilarityStore) -> CompiledStateGraph:
    tools = [
        make_search_tool(store, settings.retrieval_k, settings.retrieval_score_threshold),
        make_order_tool(ORDERS),
    ]
    is_in_scope = make_scope_checker(store, settings.scope_score_threshold)
    return build_graph(build_llm(settings), tools, is_in_scope, checkpointer, settings.history_max_messages)


def run(
    graph: CompiledStateGraph,
    question: str,
    callbacks: list | None = None,
    session_id: str | None = None,
) -> dict:
    """Single entrypoint shared by the API and the eval runner."""
    session_id = session_id or str(uuid.uuid4())  # no id: a fresh one-off conversation
    config: RunnableConfig = RunnableConfig(
        callbacks=callbacks or [],
        configurable={"thread_id": session_id},
        metadata={"langfuse_session_id": session_id},
    )
    out = graph.invoke({"messages": [("user", question)]}, config=config)

    tools_used = [tc["name"] for m in current_turn(out["messages"]) for tc in getattr(m, "tool_calls", []) or []]
    return {"answer": out["messages"][-1].content, "tools_used": tools_used, "session_id": session_id}
