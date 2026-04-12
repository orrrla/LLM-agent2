from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class RedisSettings:
    host: str = _env("REDIS_HOST", "127.0.0.1")
    port: int = _env_int("REDIS_PORT", 6379)
    db: int = _env_int("REDIS_DB", 0)


@dataclass(frozen=True)
class ModelSettings:
    rewrite_model: str = _env("REWRITE_MODEL", "ep-20250206092527-ms2qn")
    arbitration_model: str = _env("ARBITRATION_MODEL", "ep-20250122160643-vj459")
    nlg_model: str = _env("NLG_MODEL", "ep-20241203180921-h2kgz")
    correlation_model: str = _env("CORRELATION_MODEL", "ep-20241203180921-h2kgz")
    nlu_fc_model: str = _env("NLU_FC_MODEL", "ep-20250106153928-kh8t7")
    bot_chat_model: str = _env("BOT_CHAT_MODEL", "bot-20250227131955-snjfg")
    deep_research_model: str = _env("DEEP_RESEARCH_MODEL", _env("NLG_MODEL", "ep-20241203180921-h2kgz"))
    deep_research_trigger_model: str = _env("DEEP_RESEARCH_TRIGGER_MODEL", "")
    distill_model: str = _env("DISTILL_MODEL", _env("NLU_FC_MODEL", "ep-20250106153928-kh8t7"))
    embedding_model: str = _env("MEMORY_EMBEDDING_MODEL", "")


@dataclass(frozen=True)
class AppSettings:
    api_key: str = _env("API_KEY")
    base_url: str = _env("BASE_URL")
    bot_url: str = _env("BOT_URL")
    reject_url: str = _env("REJECT_URL")
    intent_url: str = _env("INTENT_URL")
    nlu_url: str = _env("NLU_URL")
    project_root: Path = Path(__file__).resolve().parents[1]
    sessions_dir: Path = Path(_env("MEMORY_SESSIONS_DIR", str(Path(__file__).resolve().parents[1] / "memory_store" / "sessions")))
    bm25_index_dir: Path = Path(_env("BM25_INDEX_DIR", str(Path(__file__).resolve().parents[1] / "memory_store" / "bm25")))
    postgres_dsn: str = _env("POSTGRES_DSN")
    postgres_host: str = _env("POSTGRES_HOST", "localhost")
    postgres_port: int = _env_int("POSTGRES_PORT", 5432)
    postgres_user: str = _env("POSTGRES_USER", "postgres")
    postgres_password: str = _env("POSTGRES_PASSWORD", "")
    postgres_db: str = _env("POSTGRES_DB", "postgres")
    memory_backend: str = _env("MEMORY_BACKEND", "off").lower()
    memory_inject_mode: str = _env("MEMORY_V2_INJECT", "always").lower()
    memory_inject_top_k: int = _env_int("MEMORY_V2_INJECT_TOP_K", 3)
    memory_dense_top_k: int = _env_int("MEMORY_V2_DENSE_TOP_K", 20)
    memory_keyword_top_k: int = _env_int("MEMORY_V2_KEYWORD_TOP_K", 20)
    memory_rrf_k: int = _env_int("MEMORY_V2_RRF_K", 60)
    memory_dense_weight: float = _env_float("MEMORY_V2_DENSE_WEIGHT", 0.35)
    memory_keyword_weight: float = _env_float("MEMORY_V2_KEYWORD_WEIGHT", 0.65)
    memory_min_exchange_chars: int = _env_int("MEMORY_V2_MIN_EXCHANGE_CHARS", 40)
    memory_distill_timeout: float = _env_float("DISTILL_TIMEOUT", 10.0)
    memory_search_enabled: bool = _env_bool("MEMORY_V2_SEARCH_ENABLED", True)
    memory_init_schema: bool = _env_bool("MEMORY_V2_INIT_SCHEMA", True)
    memory_embedding_url: str = _env("MEMORY_EMBEDDING_URL", _env("EMBEDDING_URL", ""))
    memory_embedding_api_key: str = _env("MEMORY_EMBEDDING_API_KEY", _env("API_KEY", ""))
    memory_embedding_dimensions: int = _env_int("MEMORY_EMBEDDING_DIMENSIONS", 1024)
    deep_research_enabled: bool = _env_bool("DEEP_RESEARCH_ENABLED", False)
    deep_research_timeout: float = _env_float("DEEP_RESEARCH_TIMEOUT", 18.0)
    deep_research_max_results: int = _env_int("DEEP_RESEARCH_MAX_RESULTS", 5)
    deep_research_fetch_top_n: int = _env_int("DEEP_RESEARCH_FETCH_TOP_N", 3)
    tavily_api_key: str = _env("TAVILY_API_KEY")

    @property
    def postgres_resolved_dsn(self) -> str:
        if self.postgres_dsn:
            return self.postgres_dsn
        password = f":{self.postgres_password}" if self.postgres_password else ""
        return (
            f"postgresql://{self.postgres_user}{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def get_app_settings() -> AppSettings:
    return AppSettings()


def get_model_settings() -> ModelSettings:
    return ModelSettings()


def get_redis_settings() -> RedisSettings:
    return RedisSettings()
