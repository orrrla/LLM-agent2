from __future__ import annotations

from dataclasses import dataclass

from config.runtime import get_app_settings


@dataclass(frozen=True)
class MemoryConfig:
    backend: str
    inject_mode: str
    inject_top_k: int
    dense_top_k: int
    keyword_top_k: int
    rrf_k: int
    dense_weight: float
    keyword_weight: float
    min_exchange_chars: int

    @property
    def enabled(self) -> bool:
        return self.backend == "v2"


def get_memory_config() -> MemoryConfig:
    settings = get_app_settings()
    return MemoryConfig(
        backend=settings.memory_backend,
        inject_mode=settings.memory_inject_mode,
        inject_top_k=settings.memory_inject_top_k,
        dense_top_k=settings.memory_dense_top_k,
        keyword_top_k=settings.memory_keyword_top_k,
        rrf_k=settings.memory_rrf_k,
        dense_weight=settings.memory_dense_weight,
        keyword_weight=settings.memory_keyword_weight,
        min_exchange_chars=settings.memory_min_exchange_chars,
    )
