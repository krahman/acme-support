from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ai.rag.loader import load_documents
from app.config import Settings


def build_vector_store(settings: Settings) -> InMemoryVectorStore:
    docs = load_documents(settings.kb_dir)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(docs)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return InMemoryVectorStore.from_documents(chunks, embeddings)
