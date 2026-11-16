from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.ai.guardrails import REFUSAL, current_turn, has_evidence
from app.ai.prompts import SYSTEM


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    is_in_scope: Callable[[str], bool],
    checkpointer: BaseCheckpointSaver | None = None,
    max_history: int = 20,
) -> CompiledStateGraph:
    llm_with_tools = llm.bind_tools(tools)
    llm_must_use_tool = llm.bind_tools(tools, tool_choice="required")

    def route_gate(state: State) -> str:
        asked = [str(m.content) for m in state["messages"] if m.type == "human"]
        # a follow-up like "yes please" only makes sense next to the question before it
        return "agent" if is_in_scope(" ".join(asked[-2:])) else "refuse"

    def refuse(state: State) -> dict:
        return {"messages": [AIMessage(content=REFUSAL)]}

    def guard(state: State) -> dict:
        return {} if has_evidence(state["messages"]) else {"messages": [AIMessage(content=REFUSAL)]}

    def agent(state: State) -> dict:
        history = trim_messages(
            state["messages"],
            strategy="last",
            max_tokens=max_history,
            token_counter=len,
            start_on="human",
        )
        # first call of a turn: no tool has run yet, so force one. Every answer then rests on fresh evidence.
        model = llm_with_tools if current_turn(state["messages"]) else llm_must_use_tool
        return {"messages": [model.invoke([SYSTEM, *history])]}

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
    return builder.compile(checkpointer=checkpointer)
