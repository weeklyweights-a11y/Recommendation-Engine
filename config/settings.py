"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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

    google_ai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_AI_API_KEY", "GOOGLE_API_KEY"),
    )
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
    domain_similarity_path: str = Field(
        default="./config/domain_similarity.yaml",
        alias="DOMAIN_SIMILARITY_PATH",
    )
    role_compatibility_path: str = Field(
        default="./config/role_compatibility.yaml",
        alias="ROLE_COMPATIBILITY_PATH",
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


class JobEmbeddingSettings(BaseSettings):
    """Batch job embedding pipeline settings (Phase 3)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    job_embed_batch_size: int = Field(default=64, alias="JOB_EMBED_BATCH_SIZE")
    job_llm_batch_size: int = Field(default=10, alias="JOB_LLM_BATCH_SIZE")
    job_skill_dict_path: str = Field(
        default="./data/esco/supplemental_aliases.csv",
        alias="JOB_SKILL_DICT_PATH",
    )
    faiss_flat_ip_max_jobs: int = Field(default=50_000, alias="FAISS_FLAT_IP_MAX_JOBS")
    faiss_index_type: str = Field(default="", alias="FAISS_INDEX_TYPE")
    faiss_ivf_nlist: int = Field(default=0, alias="FAISS_IVF_NLIST")
    faiss_nprobe: int = Field(default=10, alias="FAISS_NPROBE")


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bm25_top_k: int = Field(default=500, alias="BM25_TOP_K")
    hybrid_top_k: int = Field(default=500, alias="HYBRID_TOP_K")
    es_index_batch_size: int = Field(default=500, alias="ES_INDEX_BATCH_SIZE")
    kaggle_batch_size: int = Field(default=1000, alias="KAGGLE_BATCH_SIZE")
    vector_skill_weight: float = Field(default=0.35, alias="VECTOR_SKILL_WEIGHT")
    vector_domain_weight: float = Field(default=0.25, alias="VECTOR_DOMAIN_WEIGHT")
    vector_role_weight: float = Field(default=0.25, alias="VECTOR_ROLE_WEIGHT")
    vector_environment_weight: float = Field(default=0.15, alias="VECTOR_ENVIRONMENT_WEIGHT")
    vector_overfetch_multiplier: int = Field(default=2, alias="VECTOR_OVERFETCH_MULTIPLIER")
    vector_zero_norm_epsilon: float = Field(default=1e-6, alias="VECTOR_ZERO_NORM_EPSILON")
    graph_expansion_max_hops: int = Field(default=2, alias="GRAPH_EXPANSION_MAX_HOPS")
    graph_direct_match_weight: float = Field(default=1.0, alias="GRAPH_DIRECT_MATCH_WEIGHT")
    graph_one_hop_weight: float = Field(default=0.5, alias="GRAPH_ONE_HOP_WEIGHT")
    graph_two_hop_weight: float = Field(default=0.25, alias="GRAPH_TWO_HOP_WEIGHT")
    fusion_bm25_weight: float = Field(default=0.25, alias="FUSION_BM25_WEIGHT")
    fusion_vector_weight: float = Field(default=0.45, alias="FUSION_VECTOR_WEIGHT")
    fusion_graph_weight: float = Field(default=0.30, alias="FUSION_GRAPH_WEIGHT")
    fusion_strategy: Literal["rrf", "weighted_sum"] = Field(default="rrf", alias="FUSION_STRATEGY")
    rrf_k: int = Field(default=60, alias="RRF_K")


class IngestionSettings(BaseSettings):
    """Resume parsing, GitHub fetch, and profile merge settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    resume_max_file_bytes: int = Field(default=10485760, alias="RESUME_MAX_FILE_BYTES")
    resume_min_text_chars: int = Field(default=50, alias="RESUME_MIN_TEXT_CHARS")
    resume_max_text_chars: int = Field(default=100000, alias="RESUME_MAX_TEXT_CHARS")
    github_api_base_url: str = Field(
        default="https://api.github.com",
        alias="GITHUB_API_BASE_URL",
    )
    github_max_repos: int = Field(default=300, alias="GITHUB_MAX_REPOS")
    github_top_repos_languages: int = Field(default=20, alias="GITHUB_TOP_REPOS_LANGUAGES")
    github_top_repos_readme: int = Field(default=10, alias="GITHUB_TOP_REPOS_README")
    github_top_repos_signals: int = Field(default=20, alias="GITHUB_TOP_REPOS_SIGNALS")
    github_rate_limit_warn_threshold: int = Field(
        default=10,
        alias="GITHUB_RATE_LIMIT_WARN_THRESHOLD",
    )
    github_rate_limit_max_retries: int = Field(default=3, alias="GITHUB_RATE_LIMIT_MAX_RETRIES")
    github_fork_recency_days: int = Field(default=180, alias="GITHUB_FORK_RECENCY_DAYS")
    github_llm_summary_max_chars: int = Field(default=4000, alias="GITHUB_LLM_SUMMARY_MAX_CHARS")
    github_username_max_length: int = Field(default=39, alias="GITHUB_USERNAME_MAX_LENGTH")
    skill_depth_resume_mention: float = Field(default=0.25, alias="SKILL_DEPTH_RESUME_MENTION")
    skill_depth_resume_proficiency: float = Field(
        default=0.15,
        alias="SKILL_DEPTH_RESUME_PROFICIENCY",
    )
    skill_depth_github_presence: float = Field(default=0.25, alias="SKILL_DEPTH_GITHUB_PRESENCE")
    skill_depth_github_recency: float = Field(default=0.15, alias="SKILL_DEPTH_GITHUB_RECENCY")
    skill_depth_github_production: float = Field(
        default=0.10,
        alias="SKILL_DEPTH_GITHUB_PRODUCTION",
    )
    skill_depth_github_volume: float = Field(default=0.10, alias="SKILL_DEPTH_GITHUB_VOLUME")
    skill_depth_base_resume_only: float = Field(default=0.3, alias="SKILL_DEPTH_BASE_RESUME_ONLY")
    skill_depth_base_github_only: float = Field(default=0.2, alias="SKILL_DEPTH_BASE_GITHUB_ONLY")
    skill_expansion_max_hops_embed: int = Field(default=1, alias="SKILL_EXPANSION_MAX_HOPS_EMBED")
    embedding_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        alias="EMBEDDING_DEVICE",
    )
    embedding_chunk_token_limit: int = Field(default=512, alias="EMBEDDING_CHUNK_TOKEN_LIMIT")


class RerankerSettings(BaseSettings):
    """Multi-factor reranker settings (Phase 4)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    skill_fit_weight: float = Field(default=0.25, alias="RERANK_SKILL_FIT_WEIGHT")
    experience_alignment_weight: float = Field(
        default=0.15,
        alias="RERANK_EXPERIENCE_ALIGNMENT_WEIGHT",
    )
    domain_relevance_weight: float = Field(default=0.15, alias="RERANK_DOMAIN_RELEVANCE_WEIGHT")
    role_shape_weight: float = Field(default=0.15, alias="RERANK_ROLE_SHAPE_WEIGHT")
    location_fit_weight: float = Field(default=0.10, alias="RERANK_LOCATION_FIT_WEIGHT")
    company_stage_weight: float = Field(default=0.10, alias="RERANK_COMPANY_STAGE_WEIGHT")
    semantic_similarity_weight: float = Field(default=0.10, alias="RERANK_SEMANTIC_WEIGHT")
    freshness_48h_boost: float = Field(default=0.05, alias="RERANK_FRESHNESS_48H_BOOST")
    freshness_7d_boost: float = Field(default=0.03, alias="RERANK_FRESHNESS_7D_BOOST")
    diversity_min_industries: int = Field(default=3, alias="RERANK_DIVERSITY_MIN_INDUSTRIES")
    diversity_min_stages: int = Field(default=3, alias="RERANK_DIVERSITY_MIN_STAGES")
    diversity_top_n: int = Field(default=20, alias="RERANK_DIVERSITY_TOP_N")
    diversity_keep_top: int = Field(default=14, alias="RERANK_DIVERSITY_KEEP_TOP")
    diversity_inject_start: int = Field(default=15, alias="RERANK_DIVERSITY_INJECT_START")
    diversity_inject_end: int = Field(default=20, alias="RERANK_DIVERSITY_INJECT_END")
    diversity_scan_until: int = Field(default=100, alias="RERANK_DIVERSITY_SCAN_UNTIL")
    worth_exploring_semantic_min: float = Field(
        default=0.6,
        alias="RERANK_WORTH_EXPLORING_SEMANTIC_MIN",
    )
    worth_exploring_factor_max: float = Field(
        default=0.5,
        alias="RERANK_WORTH_EXPLORING_FACTOR_MAX",
    )
    worth_exploring_percentile: float = Field(
        default=0.2,
        alias="RERANK_WORTH_EXPLORING_PERCENTILE",
    )
    skill_direct_weight: float = Field(default=1.0, alias="RERANK_SKILL_DIRECT_WEIGHT")
    skill_one_hop_weight: float = Field(default=0.6, alias="RERANK_SKILL_ONE_HOP_WEIGHT")
    skill_two_hop_weight: float = Field(default=0.3, alias="RERANK_SKILL_TWO_HOP_WEIGHT")
    skill_default_required_count: int = Field(
        default=8,
        alias="RERANK_SKILL_DEFAULT_REQUIRED_COUNT",
    )
    skill_depth_threshold: float = Field(default=0.7, alias="RERANK_SKILL_DEPTH_THRESHOLD")
    skill_depth_boost_multiplier: float = Field(
        default=1.2,
        alias="RERANK_SKILL_DEPTH_BOOST_MULTIPLIER",
    )
    career_changer_role_shape_floor: float = Field(
        default=0.6,
        alias="RERANK_CAREER_CHANGER_ROLE_SHAPE_FLOOR",
    )
    stage_adjacent_score: float = Field(default=0.7, alias="RERANK_STAGE_ADJACENT_SCORE")
    stage_two_step_score: float = Field(default=0.4, alias="RERANK_STAGE_TWO_STEP_SCORE")
    stage_far_score: float = Field(default=0.2, alias="RERANK_STAGE_FAR_SCORE")
    stage_unknown_score: float = Field(default=0.5, alias="RERANK_STAGE_UNKNOWN_SCORE")
    stage_no_preference_score: float = Field(default=0.7, alias="RERANK_STAGE_NO_PREF_SCORE")
    experience_unknown_score: float = Field(default=0.7, alias="RERANK_EXPERIENCE_UNKNOWN_SCORE")


class HardFilterSettings(BaseSettings):
    """Hard constraint filter settings (Phase 4)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    min_results_warn: int = Field(default=50, alias="HARD_FILTER_MIN_RESULTS_WARN")
    pipeline_min_warn: int = Field(default=10, alias="HARD_FILTER_PIPELINE_MIN_WARN")
    all_work_models: str = Field(
        default="remote,hybrid,onsite",
        alias="HARD_FILTER_ALL_WORK_MODELS",
    )


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
    job_embedding: JobEmbeddingSettings = Field(default_factory=JobEmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    hard_filter: HardFilterSettings = Field(default_factory=HardFilterSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
