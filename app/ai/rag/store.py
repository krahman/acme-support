import logging
import os

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.ai.rag.chunks import chunk_documents, index_fingerprint
from app.ai.rag.loader import load_documents
from app.config import Settings

log = logging.getLogger("genai.rag")


def make_embeddings(settings: Settings) -> Embeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_vector_store(settings: Settings, embeddings: Embeddings | None = None) -> InMemoryVectorStore:
    """Load the index for the current KB from disk, or build and save it. No stale index is over served."""
    embeddings = embeddings or make_embeddings(settings)
    docs = load_documents(settings.kb_dir)
    path = settings.index_dir / f"{index_fingerprint(settings, docs)}.json"

    if path.exists():
        try:
            store = InMemoryVectorStore.load(str(path), embeddings)
            log.info("loaded vector index %s", path.name)
            return store
        except Exception:
            log.warning("vector index %s is unreadable, rebuilding", path.name, exc_info=True)

    chunks = chunk_documents(settings, docs)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    log.info("built vector index %s (%d chunks)", path.name, len(chunks))

    try:
        tmp = path.with_suffix(".tmp")
        store.dump(str(tmp))
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
    except OSError:
        log.warning("could not save vector index to %s, serving from memory", path, exc_info=True)
    return store
