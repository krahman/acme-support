from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from app.api.schema import ChatRequest, ChatResponse
from app.bootstrap import run


class ChatService:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    def chat(self, req: ChatRequest) -> ChatResponse:
        handler = CallbackHandler()
        result = run(self._graph, req.message, callbacks=[handler], session_id=req.session_id)
        return ChatResponse(**result, trace_id=handler.last_trace_id)
