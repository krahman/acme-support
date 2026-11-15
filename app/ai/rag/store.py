import hashlib
import json
import logging
import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ai.rag.loader import load_documents
from app.config import Settings

log = logging.getLogger("genai.rag")


def make_embeddings(settings: Settings) -> Embeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def index_fingerprint(settings: Settings, docs: list[Document]) -> str:
    """Hash of everything that changes the vectors: KB content and the settings that shape them."""
    h = hashlib.sha256(json.dumps([settings.embedding_model, settings.chunk_size, settings.chunk_overlap]).encode())
    for d in docs:
        h.update(d.metadata["source"].encode())
        h.update(d.page_content.encode())
    return h.hexdigest()[:16]


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

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    log.info("built vector index %s (%d chunks)", path.name, len(chunks))

    try:
        tmp = path.with_suffix(".tmp")
        store.dump(str(tmp))
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
    except OSError:
        log.warning("could not save vector index to %s, serving from memory", path, exc_info=True)
    return store
