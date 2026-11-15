from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    kb_dir = kb_dir or get_settings().kb_dir
    return [
        Document(page_content=d.read_text().strip(), metadata={"source": d.name}) for d in sorted(kb_dir.glob("*.md"))
    ]
