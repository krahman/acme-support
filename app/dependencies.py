from typing import Annotated

from fastapi import Depends, Request
from langfuse import Langfuse
from langgraph.graph.state import CompiledStateGraph

from app.ai.sessions import SessionStore
from app.services.chat_service import ChatService


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_langfuse(request: Request) -> Langfuse:
    return request.app.state.langfuse


def get_chat_service(graph: GraphDependency, sessions: SessionsDependency) -> ChatService:
    return ChatService(graph, sessions)


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions


GraphDependency = Annotated[CompiledStateGraph, Depends(get_graph)]
SessionsDependency = Annotated[SessionStore, Depends(get_sessions)]
LangfuseDependency = Annotated[Langfuse, Depends(get_langfuse)]
ChatServiceDependency = Annotated[ChatService, Depends(get_chat_service)]
