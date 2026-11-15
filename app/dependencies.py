from typing import Annotated

from fastapi import Depends, Request
from langfuse import Langfuse
from langgraph.graph.state import CompiledStateGraph

from app.services.chat_service import ChatService


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_langfuse(request: Request) -> Langfuse:
    return request.app.state.langfuse

def get_chat_service(graph: GraphDependency) -> ChatService:
  return ChatService(graph)

GraphDependency = Annotated[CompiledStateGraph, Depends(get_graph)]
LangfuseDependency = Annotated[Langfuse, Depends(get_langfuse)]
ChatServiceDependency = Annotated[ChatService, Depends(get_chat_service)]
