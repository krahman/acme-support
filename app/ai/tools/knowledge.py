from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

NO_RESULTS = "No relevant documents found."

def make_search_tool(store: InMemoryVectorStore, k: int, score_threshold: float):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search AcmeCloud policy docs (refunds, pricing, SLA)."""
        hits = store.similarity_search_with_score(query, k=k)
        docs = [doc for doc, score, in hits if score >= score_threshold]
        if not docs:
            return NO_RESULTS
        return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)

    return search_knowledge_base
