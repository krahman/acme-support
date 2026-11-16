import uuid

import openai
import psycopg
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.ai.sessions import NullSessionStore, SessionStore
from app.api.schema import ChatRequest, ChatResponse
from app.bootstrap import run
from app.errors import (
    AgentLoopError,
    StorageUnavailableError,
    UpstreamRejectedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)


class ChatService:
    def __init__(self, graph: CompiledStateGraph, sessions: SessionStore | None = None) -> None:
        self._graph = graph
        self._sessions = sessions or NullSessionStore()

    def chat(self, req: ChatRequest) -> ChatResponse:
        session_id = str(req.session_id or uuid.uuid4())
        try:
            self._sessions.touch(session_id)  # before the turn: the clock starts even if the turn fails
            handler = CallbackHandler()
            result = run(
                self._graph,
                req.message,
                callbacks=[handler],
                session_id=session_id,
            )
        except openai.APITimeoutError as e:  # before APIConnectionError: it is a subclass
            raise UpstreamTimeoutError() from e
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise UpstreamUnavailableError() from e
        except openai.APIStatusError as e:  # 400/401/403/404...: our request or config is wrong
            raise UpstreamRejectedError() from e
        except psycopg.OperationalError as e:  # includes pool timeouts: the session store is unreachable
            raise StorageUnavailableError() from e
        except GraphRecursionError as e:
            raise AgentLoopError() from e
        return ChatResponse(
            **result,
            trace_id=handler.last_trace_id,
        )
