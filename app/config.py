from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # app
    log_level: str = "INFO"

    # secret (required: app fails to start if missing)
    openai_api_key: SecretStr
    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    langfuse_host: str = "http://localhost:3000"

    # models
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"

    # RAG
    kb_dir: Path = BASE_DIR / "knowledge_base"
    chunk_size: int = 300
    chunk_overlap: int = 30
    retrieval_k: int = 2
    retrieval_score_threshold: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
