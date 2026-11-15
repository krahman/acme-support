import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from app.ai.rag.store import get_vector_store
from app.config import Settings


class FakeEmbeddings(Embeddings):
    """Deterministic vectors, and counts how many documents were embedded."""

    def __init__(self):
        self.docs_embedded = 0

    def embed_documents(self, texts):
        self.docs_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:8]]


@pytest.fixture
def settings(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "pricing.md").write_text("The Team plan costs $50 per month.")
    (kb / "sla.md").write_text("Pro has a best-effort uptime target.")
    return Settings(
        _env_file=None,
        openai_api_key="x",
        langfuse_public_key="x",
        langfuse_secret_key="x",
        kb_dir=kb,
        index_dir=tmp_path / "index",
    )


def index_files(settings):
    return sorted(settings.index_dir.glob("*.json"))


def test_first_call_builds_and_saves_the_index(settings):
    emb = FakeEmbeddings()
    get_vector_store(settings, emb)
    assert emb.docs_embedded > 0
    assert len(index_files(settings)) == 1


def test_second_call_loads_from_disk_without_embedding_documents(settings):
    get_vector_store(settings, FakeEmbeddings())
    emb = FakeEmbeddings()
    get_vector_store(settings, emb)
    assert emb.docs_embedded == 0


def test_loaded_store_returns_the_same_results_as_the_built_one(settings):
    built = get_vector_store(settings, FakeEmbeddings())
    loaded = get_vector_store(settings, FakeEmbeddings())
    q = "The Team plan costs $50 per month."
    b_doc, b_score = built.similarity_search_with_score(q, k=1)[0]
    l_doc, l_score = loaded.similarity_search_with_score(q, k=1)[0]
    assert (l_doc.page_content, l_doc.metadata) == (b_doc.page_content, b_doc.metadata)
    assert l_score == pytest.approx(b_score)


def test_editing_a_document_triggers_a_rebuild(settings):
    get_vector_store(settings, FakeEmbeddings())
    (settings.kb_dir / "sla.md").write_text("Pro now has a 99.9% uptime SLA.")
    emb = FakeEmbeddings()
    get_vector_store(settings, emb)
    assert emb.docs_embedded > 0
    assert len(index_files(settings)) == 2


def test_changing_chunk_size_triggers_a_rebuild(settings):
    get_vector_store(settings, FakeEmbeddings())
    emb = FakeEmbeddings()
    get_vector_store(settings.model_copy(update={"chunk_size": 100}), emb)
    assert emb.docs_embedded > 0


def test_corrupt_index_is_rebuilt(settings):
    get_vector_store(settings, FakeEmbeddings())
    index_files(settings)[0].write_text("{not json")
    emb = FakeEmbeddings()
    store = get_vector_store(settings, emb)
    assert emb.docs_embedded > 0
    assert store.similarity_search("plan", k=1)
    assert get_vector_store(settings, FakeEmbeddings())  # and the rewritten file loads again


def test_unwritable_index_dir_still_serves_from_memory(settings, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    store = get_vector_store(settings.model_copy(update={"index_dir": blocker}), FakeEmbeddings())
    assert store.similarity_search("plan", k=1)
