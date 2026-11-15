"""RAG pipeline + tools + LangGraph agent graph."""
from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ---------- RAG pipeline: load -> split -> embed -> index -> retrieve ----------
KB = [
    Document(page_content="Refund policy: Pro and Team plans get a full refund within 14 days "
             "of purchase. After 14 days, refunds are prorated. Free plan has no refunds.",
             metadata={"source": "refund-policy.md"}),
    Document(page_content="Plans: Free ($0), Pro ($20/month), Team ($50/user/month). "
             "Team includes SSO and audit logs.", metadata={"source": "pricing.md"}),
    Document(page_content="SLA: Team plan has 99.9% uptime SLA with service credits. "
             "Pro plan is best-effort.", metadata={"source": "sla.md"}),
]
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(KB)
vector_store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ---------- Tools ----------
ORDERS = {  # mock order DB
    "A-1001": {"plan": "Pro", "days_since_purchase": 10, "amount_usd": 20},
    "A-1002": {"plan": "Team", "days_since_purchase": 30, "amount_usd": 150},
}


@tool
def search_knowledge_base(query: str) -> str:
    """Search Acme Cloud policy docs (refunds, pricing, SLA)."""
    docs = retriever.invoke(query)
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns plan, days since purchase and amount."""
    return ORDERS.get(order_id.upper(), {"error": f"order {order_id} not found"})


TOOLS = [search_knowledge_base, get_order_status]

# ---------- LangGraph: agent <-> tools loop ----------
SYSTEM = SystemMessage(content=(
    "You are Acme Cloud support. Always ground policy answers in search_knowledge_base "
    "and use get_order_status for order questions. Be concise and cite sources."))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent(state: State) -> dict:
    return {"messages": [llm.invoke([SYSTEM, *state["messages"]])]}


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()


def run(question: str, callbacks: list | None = None, session_id: str | None = None) -> dict:
    """Single entrypoint shared by the API and the eval runner."""
    out = graph.invoke(
        {"messages": [("user", question)]},
        config={"callbacks": callbacks or [],
                "metadata": {"langfuse_session_id": session_id} if session_id else {}},
    )
    tools_used = [tc["name"] for m in out["messages"] for tc in getattr(m, "tool_calls", []) or []]
    return {"answer": out["messages"][-1].content, "tools_used": tools_used}
