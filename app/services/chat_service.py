import openai
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.api.schema import ChatRequest, ChatResponse
from app.bootstrap import run
from app.errors import AgentLoopError, UpstreamRejectedError, UpstreamTimeoutError, UpstreamUnavailableError


class ChatService:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    def chat(self, req: ChatRequest) -> ChatResponse:
        try:
            handler = CallbackHandler()
            result = run(self._graph, req.message, callbacks=[handler], session_id=req.session_id)
        except openai.APITimeoutError as e:  # before APIConnectionError: it is a subclass
            raise UpstreamTimeoutError() from e
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise UpstreamUnavailableError() from e
        except openai.APIStatusError as e:  # 400/401/403/404...: our request or config is wrong
            raise UpstreamRejectedError() from e
        except GraphRecursionError as e:
            raise AgentLoopError() from e
        return ChatResponse(**result, trace_id=handler.last_trace_id)
