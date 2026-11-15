from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.ai.prompts import SYSTEM


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(llm: BaseChatModel, tools: list[BaseTool]) -> CompiledStateGraph:
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: State) -> dict:
        return {"messages": [llm_with_tools.invoke([SYSTEM, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()
