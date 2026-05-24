"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        ...,
        alias="DATABASE_URL",
        description="Async SQLAlchemy URL for application runtime.",
    )
    alembic_database_url: str = Field(
        ...,
        alias="ALEMBIC_DATABASE_URL",
        description="Sync SQLAlchemy URL for Alembic migrations.",
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Ensure DATABASE_URL uses a PostgreSQL driver."""
        if not value.startswith("postgresql"):
            msg = "DATABASE_URL must start with postgresql"
            raise ValueError(msg)
        return value


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class Neo4jSettings(BaseSettings):
    """Neo4j knowledge graph settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        alias="ELASTICSEARCH_URL",
    )
    es_index_name: str = Field(default="jobs", alias="ES_INDEX_NAME")


class LLMSettings(BaseSettings):
    """Google Gemini API settings (Phase 2+).

    Use ``llm_model_pro`` for quality-critical paths (resume extraction, match
    explanations). Use ``llm_model_flash`` for high-volume batch work (job field
    extraction and similar).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_ai_api_key: str = Field(default="", alias="GOOGLE_AI_API_KEY")
    llm_model_pro: str = Field(
        default="gemini-2.5-pro",
        alias="LLM_MODEL_PRO",
    )
    llm_model_flash: str = Field(
        default="gemini-2.5-flash",
        alias="LLM_MODEL_FLASH",
    )
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")


class EmbeddingSettings(BaseSettings):
    """Embedding model and artifact paths."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    faiss_index_path: str = Field(
        default="./data/faiss_indexes/",
        alias="FAISS_INDEX_PATH",
    )
    esco_embeddings_path: str = Field(
        default="./data/esco/skill_embeddings.npy",
        alias="ESCO_EMBEDDINGS_PATH",
    )
    esco_uri_index_path: str = Field(
        default="./data/esco/skill_uri_index.json",
        alias="ESCO_URI_INDEX_PATH",
    )


class GitHubSettings(BaseSettings):
    """GitHub API settings (Phase 2+)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str = Field(default="", alias="GITHUB_TOKEN")


class PathSettings(BaseSettings):
    """Filesystem paths for data and configs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    esco_data_path: str = Field(default="./data/esco/", alias="ESCO_DATA_PATH")
    scraper_config_path: str = Field(
        default="./config/scraper_config.yaml",
        alias="SCRAPER_CONFIG_PATH",
    )
    scraper_output_path: str = Field(
        default="./data/sample_jobs/scraped_jobs.jsonl",
        alias="SCRAPER_OUTPUT_PATH",
    )
    kaggle_jobs_path: str = Field(
        default="./data/sample_jobs/",
        alias="KAGGLE_JOBS_PATH",
    )


class ScraperSettings(BaseSettings):
    """Job scraper behavior settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    scraper_default_mode: Literal["jsonl", "db", "both"] = Field(
        default="both",
        alias="SCRAPER_DEFAULT_MODE",
    )


class SkillGraphSettings(BaseSettings):
    """ESCO skill graph expansion and linking thresholds."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    skill_expansion_max_hops: int = Field(default=3, alias="SKILL_EXPANSION_MAX_HOPS")
    skill_hop_decay_1: float = Field(default=1.0, alias="SKILL_HOP_DECAY_1")
    skill_hop_decay_2: float = Field(default=0.5, alias="SKILL_HOP_DECAY_2")
    skill_hop_decay_3: float = Field(default=0.25, alias="SKILL_HOP_DECAY_3")
    skill_broader_penalty: float = Field(default=0.8, alias="SKILL_BROADER_PENALTY")
    fuzzy_match_threshold: float = Field(default=0.85, alias="FUZZY_MATCH_THRESHOLD")
    semantic_match_threshold: float = Field(
        default=0.75,
        alias="SEMANTIC_MATCH_THRESHOLD",
    )
    skill_cache_size: int = Field(default=10000, alias="SKILL_CACHE_SIZE")


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bm25_top_k: int = Field(default=500, alias="BM25_TOP_K")
    es_index_batch_size: int = Field(default=500, alias="ES_INDEX_BATCH_SIZE")
    kaggle_batch_size: int = Field(default=1000, alias="KAGGLE_BATCH_SIZE")


class AppSettings(BaseSettings):
    """Application server settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    frontend_port: int = Field(default=8501, alias="FRONTEND_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


class Settings(BaseSettings):
    """Root settings aggregating all configuration groups."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    skill_graph: SkillGraphSettings = Field(default_factory=SkillGraphSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
