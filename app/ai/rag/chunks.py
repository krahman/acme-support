import hashlib
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


def index_fingerprint(settings: Settings, docs: list[Document]) -> str:
    h = hashlib.sha256(json.dumps([settings.embedding_model, settings.chunk_size, settings.chunk_overlap]).encode())
    for doc in docs:
        h.update(doc.metadata["source"].encode())
        h.update(doc.page_content.encode())
    return h.hexdigest()[:16]


def chunk_documents(settings: Settings, docs: list[Document]) -> list[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(docs)
