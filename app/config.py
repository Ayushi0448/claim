"""Central configuration.

Every tunable is driven by an environment variable with a safe default, so the
same image runs locally, in Docker and on a free-tier host without code edits.
No secret is ever hard-coded; credentials are read from the environment only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class RetrievalSettings:
    """Hybrid retrieval tuning knobs."""

    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 12))
    bm25_top_k: int = field(default_factory=lambda: _env_int("BM25_TOP_K", 12))
    fusion_top_k: int = field(default_factory=lambda: _env_int("FUSION_TOP_K", 16))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 6))
    # Evidence guaranteed to survive from EACH dimension's own ranking. Without
    # this floor, a narrow dimension loses the global contest and its governing
    # clause never reaches the rule engine. See docs/failure-analysis.md F-01.
    per_dimension_k: int = field(default_factory=lambda: _env_int("PER_DIMENSION_K", 3))
    # Reciprocal Rank Fusion smoothing constant (Cormack et al., 2009).
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 1.0))
    bm25_weight: float = field(default_factory=lambda: _env_float("BM25_WEIGHT", 1.0))
    # A reranked chunk below this score is not considered usable evidence.
    min_rerank_score: float = field(
        default_factory=lambda: _env_float("MIN_RERANK_SCORE", 0.12)
    )


@dataclass(frozen=True)
class ChunkingSettings:
    """Structure-aware chunker limits (characters, not tokens, for determinism)."""

    max_chunk_chars: int = field(default_factory=lambda: _env_int("MAX_CHUNK_CHARS", 1600))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 180))
    overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 160))


@dataclass(frozen=True)
class ModelSettings:
    """Pluggable model backends.

    ``auto`` probes for the heavyweight backend and silently falls back to the
    offline one, which is what keeps the system runnable on a free tier and in
    network-restricted CI.
    """

    # auto | sentence_transformer | lsa
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "auto").lower()
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("LSA_EMBEDDING_DIM", 192))

    # auto | cross_encoder | lexical
    reranker_backend: str = field(
        default_factory=lambda: _env("RERANKER_BACKEND", "auto").lower()
    )
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )

    # LLM is optional: it enriches phrasing and adds a second-opinion validation
    # pass. It never decides admissibility on its own. See docs/architecture.md.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 20.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider in {"", "none", "off", "disabled"}:
            return False
        if self.llm_provider == "ollama":  # local, no key needed
            return True
        return bool(self.llm_api_key)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Aptino Policy-Aware Multi-Agent RAG Claim Decision Engine"
    version: str = "1.0.0"

    policy_pdf: Path = field(
        default_factory=lambda: Path(
            _env(
                "POLICY_PDF_PATH",
                str(
                    PROJECT_ROOT
                    / "data"
                    / "policy"
                    / "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
                ),
            )
        )
    )
    policy_source_name: str = field(
        default_factory=lambda: _env("POLICY_SOURCE_NAME", "USGIC-CSC-Individual-Health-Insurance.pdf")
    )
    policy_id: str = field(default_factory=lambda: _env("POLICY_ID", "UNIHLIP18004V011718"))
    index_dir: Path = field(
        default_factory=lambda: Path(_env("INDEX_DIR", str(PROJECT_ROOT / "data" / "index")))
    )

    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    models: ModelSettings = field(default_factory=ModelSettings)

    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))
    request_timeout_s: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT_S", 60.0)
    )
    max_request_bytes: int = field(
        default_factory=lambda: _env_int("MAX_REQUEST_BYTES", 256_000)
    )

    # Decision thresholds. Documented in README §14.
    abstain_below_confidence: float = field(
        default_factory=lambda: _env_float("ABSTAIN_BELOW_CONFIDENCE", 0.45)
    )
    max_validation_revisions: int = field(
        default_factory=lambda: _env_int("MAX_VALIDATION_REVISIONS", 1)
    )
    strict_citation_mode: bool = field(
        default_factory=lambda: _env_bool("STRICT_CITATION_MODE", True)
    )

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npz"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
