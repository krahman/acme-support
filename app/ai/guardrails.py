import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from app.ai.tools.knowledge import NO_RESULTS

REFUSAL = "I can only help with Acme Cloud policy and order questions."
ORDER_ID_RE = re.compile(r"\b[A-Z]-\d{4}\b", re.IGNORECASE)


def make_scope_checker(store: InMemoryVectorStore, threshold: float):
    def is_in_scope(text: str) -> bool:
        if ORDER_ID_RE.search(text):
            return True
        hits = store.similarity_search_with_score(text, k=1)
        return bool(hits) and hits[0][1] >= threshold

    return is_in_scope


def current_turn(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Everything after the latest user message, i.e. what this turn produced."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    return list(messages[last_human + 1 :])


def has_evidence(messages) -> bool:
    """True if a tool returned usable data during the current turn."""
    for m in current_turn(messages):
        if isinstance(m, ToolMessage) and m.status != "error":
            content = str(m.content)
            if content != NO_RESULTS and '"error"' not in content:
                return True
    return False
