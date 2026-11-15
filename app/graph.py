from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.ai.guardrails import REFUSAL, has_evidence
from app.ai.prompts import SYSTEM


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(llm: BaseChatModel, tools: list[BaseTool], is_in_scope: Callable[[str], bool]) -> CompiledStateGraph:
    llm_with_tools = llm.bind_tools(tools)

    def route_gate(state: State) -> str:
        question = next(m.content for m in reversed(state["messages"]) if m.type == "human")
        return "agent" if is_in_scope(question) else "refuse"

    def refuse(state: State) -> dict:
        return {"messages": [AIMessage(content=REFUSAL)]}

    def guard(state: State) -> dict:
        return {} if has_evidence(state["messages"]) else {"messages": [AIMessage(content=REFUSAL)]}

    def agent(state: State) -> dict:
        return {"messages": [llm_with_tools.invoke([SYSTEM, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("refuse", refuse)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("guard", guard)
    builder.add_conditional_edges(START, route_gate, {"agent": "agent", "refuse": "refuse"})
    builder.add_edge("refuse", END)
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "guard"})
    builder.add_edge("tools", "agent")
    builder.add_edge("guard", END)
    return builder.compile()
