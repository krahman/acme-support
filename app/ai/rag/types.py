from typing import Protocol

from langchain_core.documents import Document


class SimilarityStore(Protocol):
    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """The k nearest chunks, each with a similarity score where higher means closer."""
