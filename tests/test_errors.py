import httpx
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient
from langgraph.errors import GraphRecursionError

from app.dependencies import get_chat_service
from app.main import app
from app.services.chat_service import ChatService

REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, status: int) -> openai.APIStatusError:
    return cls("secret provider detail", response=httpx.Response(status, request=REQ), body=None)


class FailingGraph:
    def __init__(self, exc: Exception):
        self.exc = exc

    def invoke(self, *args, **kwargs):
        raise self.exc


@pytest.fixture
def client():
    def make(exc: Exception) -> TestClient:
        app.dependency_overrides[get_chat_service] = lambda: ChatService(FailingGraph(exc))
        return TestClient(app)  # no `with`: skips lifespan, so no Langfuse/OpenAI needed

    yield make
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (openai.APITimeoutError(request=REQ), 504, "upstream_timeout"),
        (openai.APIConnectionError(request=REQ), 503, "upstream_unavailable"),
        (status_error(openai.RateLimitError, 429), 503, "upstream_unavailable"),
        (status_error(openai.InternalServerError, 500), 503, "upstream_unavailable"),
        (status_error(openai.AuthenticationError, 401), 502, "upstream_rejected"),
        (status_error(openai.BadRequestError, 400), 502, "upstream_rejected"),
        (GraphRecursionError("limit"), 500, "agent_loop"),
        (psycopg.OperationalError("connection refused"), 503, "storage_unavailable"),
    ],
)
def test_known_failures_map_to_status_and_code(client, exc, status, code):
    r = client(exc).post("/chat", json={"message": "hi"})
    assert r.status_code == status
    assert r.json()["error"]["code"] == code


def test_error_body_has_request_id_matching_the_header(client):
    r = client(openai.APITimeoutError(request=REQ)).post(
        "/chat", json={"message": "hi"}, headers={"x-request-id": "abc-123"}
    )
    assert r.json()["error"]["request_id"] == "abc-123"
    assert r.headers["x-request-id"] == "abc-123"


def test_provider_details_never_reach_the_client(client):
    r = client(status_error(openai.AuthenticationError, 401)).post("/chat", json={"message": "hi"})
    assert "secret provider detail" not in r.text


def test_unexpected_exception_is_a_clean_500_with_request_id(client):
    r = client(RuntimeError("secret internal detail")).post("/chat", json={"message": "hi"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text
    assert r.headers["x-request-id"] == r.json()["error"]["request_id"]


def test_failure_is_logged_with_the_cause(client, caplog):
    client(openai.APITimeoutError(request=REQ)).post("/chat", json={"message": "hi"})
    assert any(rec.name == "genai.errors" and rec.exc_info for rec in caplog.records)


@pytest.mark.parametrize(
    "body",
    [{"message": ""}, {"message": "x" * 2001}, {"message": "hi", "session_id": "s" * 65}, {}],
)
def test_invalid_input_is_rejected_before_the_llm(client, body):
    r = client(RuntimeError("must not be reached")).post("/chat", json=body)
    assert r.status_code == 422
