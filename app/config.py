from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    llm_timeout_seconds: float = 30
    llm_max_retries: int = 2

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
    index_dir: Path = BASE_DIR / ".index"
    chunk_size: int = 300
    chunk_overlap: int = 30
    retrieval_k: int = 2
    retrieval_score_threshold: float = 0.2
    scope_score_threshold: float = 0.2

    # conversation
    history_max_messages: int = 10

    # database
    database_url: SecretStr | None = None  # unset: sessions live in memory
    db_pool_min_size: int = 1  # idle connections go stale on a DB restart; fewer means faster recovery
    db_pool_max_size: int = 10
    db_timeout_seconds: float = 5  # startup wait and per-request wait for a working connection


@lru_cache
def get_settings() -> Settings:
    return Settings()
