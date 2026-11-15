import re

from langchain_core.messages import ToolMessage
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


def has_evidence(messages) -> bool:
    """True if a tool returned usable data since the latest user message."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    for m in messages[last_human + 1 :]:
        if isinstance(m, ToolMessage) and m.status != "error":
            content = str(m.content)
            if content != NO_RESULTS and '"error"' not in content:
                return True
    return False
