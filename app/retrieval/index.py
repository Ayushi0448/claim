"""The hybrid retriever: build, persist, load and query.

Pipeline per query:  dense ──┐
                             ├─ RRF fusion ─ rerank ─ threshold ─ evidence
                     BM25  ──┘

Multiple queries (one per decision dimension) are fused a second time so the
evidence pool reflects the whole investigation rather than any single question.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.models.schemas import (
    DecisionDimension,
    EvidenceItem,
    PolicyChunk,
    RetrievalQuery,
)
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunking import chunk_policy, embedding_text
from app.retrieval.dense import DenseIndex
from app.retrieval.embeddings import EmbeddingBackend, get_embedding_backend
from app.retrieval.fusion import FusedResult, merge_multi_query, reciprocal_rank_fusion
from app.retrieval.ingestion import load_policy_pages, policy_fingerprint
from app.retrieval.reranker import RerankCandidate, Reranker, get_reranker

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 2


class IndexNotBuiltError(RuntimeError):
    """Raised when a query arrives before the policy has been indexed."""


@dataclass
class RetrievalStats:
    dense_results: int = 0
    bm25_results: int = 0
    fused_results: int = 0
    reranked_results: int = 0
    dropped_below_threshold: int = 0
    queries_executed: int = 0
    elapsed_ms: int = 0


class HybridRetriever:
    """Owns the chunk corpus and both retrieval arms."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.chunks: list[PolicyChunk] = []
        self._embedder: EmbeddingBackend | None = None
        self._dense = DenseIndex()
        self._bm25 = BM25Index()
        self._reranker: Reranker | None = None
        self._embeddings: np.ndarray | None = None
        self._by_id: dict[str, int] = {}
        self._ready = False

    # ------------------------------------------------------------------ #
    # Build / persist / load
    # ------------------------------------------------------------------ #

    def build(self, *, persist: bool = True) -> "HybridRetriever":
        started = time.perf_counter()
        cfg = self.settings

        pages = load_policy_pages(cfg.policy_pdf)
        self.chunks = chunk_policy(
            pages, source_name=cfg.policy_source_name, settings=cfg.chunking
        )
        if not self.chunks:
            raise RuntimeError("Chunking produced no chunks; check the policy PDF.")

        corpus = [embedding_text(c) for c in self.chunks]

        self._embedder = get_embedding_backend(cfg.models)
        self._embedder.fit(corpus)
        self._embeddings = self._embedder.encode(corpus)
        self._dense.build(self._embeddings)

        self._bm25.fit(corpus)
        self._reranker = get_reranker(cfg.models, bm25=self._bm25)
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._ready = True

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "Index built: %d chunks, dense=%s(%s,%dd), sparse=bm25, reranker=%s in %.0f ms",
            len(self.chunks), self._embedder.name, self._dense.backend,
            self._embeddings.shape[1], self._reranker.name, elapsed,
        )

        if persist:
            self.save()
        return self

    def save(self) -> None:
        cfg = self.settings
        cfg.index_dir.mkdir(parents=True, exist_ok=True)

        cfg.chunks_path.write_text(
            json.dumps([c.model_dump() for c in self.chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self._embeddings is not None:
            np.savez_compressed(cfg.embeddings_path, embeddings=self._embeddings)
        with (cfg.index_dir / "bm25.pkl").open("wb") as fh:
            pickle.dump(self._bm25.state_dict(), fh)
        if self._embedder is not None:
            self._embedder.save(cfg.index_dir / "embedder.pkl")

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "policy_source": cfg.policy_source_name,
            "policy_fingerprint": policy_fingerprint(cfg.policy_pdf),
            "policy_id": cfg.policy_id,
            "n_chunks": len(self.chunks),
            "embedding_backend": self._embedder.name if self._embedder else "unknown",
            "embedding_dim": int(self._embeddings.shape[1]) if self._embeddings is not None else 0,
            "dense_backend": self._dense.backend,
            "reranker_backend": self._reranker.name if self._reranker else "unknown",
            "sections": sorted({c.section for c in self.chunks}),
            "pages_indexed": sorted({c.page for c in self.chunks}),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Index persisted to %s", cfg.index_dir)

    def load(self) -> bool:
        """Load a persisted index, rejecting it if the policy PDF has changed."""
        cfg = self.settings
        try:
            if not (cfg.manifest_path.exists() and cfg.chunks_path.exists()):
                return False
            manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != INDEX_FORMAT_VERSION:
                logger.info("Index format changed; rebuild required")
                return False
            if cfg.policy_pdf.exists():
                if manifest.get("policy_fingerprint") != policy_fingerprint(cfg.policy_pdf):
                    logger.info("Policy PDF changed since index was built; rebuild required")
                    return False

            raw = json.loads(cfg.chunks_path.read_text(encoding="utf-8"))
            self.chunks = [PolicyChunk(**c) for c in raw]

            with np.load(cfg.embeddings_path) as data:
                self._embeddings = data["embeddings"]
            self._dense.build(self._embeddings)

            with (cfg.index_dir / "bm25.pkl").open("rb") as fh:
                self._bm25 = BM25Index.from_state(pickle.load(fh))

            self._embedder = get_embedding_backend(cfg.models)
            embedder_path = cfg.index_dir / "embedder.pkl"
            if self._embedder.name == "lsa" and not self._embedder.load(embedder_path):
                logger.info("LSA embedder state missing; rebuild required")
                return False
            if self._embedder.name != "lsa":
                self._embedder.load(embedder_path)

            self._reranker = get_reranker(cfg.models, bm25=self._bm25)
            self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            self._ready = True
            logger.info("Loaded persisted index with %d chunks", len(self.chunks))
            return True
        except Exception as exc:
            logger.warning("Could not load persisted index (%s); rebuilding", exc)
            return False

    def ensure_ready(self) -> "HybridRetriever":
        if self._ready:
            return self
        if not self.load():
            self.build()
        return self

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def _search_one(self, query: str) -> list[FusedResult]:
        cfg = self.settings.retrieval
        q_vec = self._embedder.encode([query])  # type: ignore[union-attr]
        dense_hits = self._dense.search(q_vec[0], top_k=cfg.dense_top_k)
        sparse_hits = self._bm25.search(query, top_k=cfg.bm25_top_k)
        return reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_weight=cfg.bm25_weight,
            top_k=cfg.fusion_top_k,
        )

    def retrieve(
        self,
        queries: list[RetrievalQuery],
        *,
        top_k: int | None = None,
    ) -> tuple[list[EvidenceItem], RetrievalStats]:
        """Run the full hybrid pipeline for a set of dimension-scoped queries.

        Evidence selection is two-phase, which matters when a claim engages a
        dozen dimensions:

        Phase 1 (recall guarantee) — every dimension is reranked against *its
        own* query and its best ``per_dimension_k`` chunks are reserved. This
        stops a narrow but decisive dimension (say the ambulance cap) from being
        crowded out by a broad one (the exclusions sweep).

        Phase 2 (precision fill) — the remaining budget is filled from the
        multi-query fused pool, which favours chunks relevant across several
        dimensions.
        """
        if not self._ready:
            raise IndexNotBuiltError(
                "Policy index is not ready. Run `python -m scripts.ingest_policy` first."
            )
        if not queries:
            return [], RetrievalStats()

        started = time.perf_counter()
        cfg = self.settings.retrieval
        stats = RetrievalStats(queries_executed=len(queries))

        per_query: list[list[FusedResult]] = []
        dim_map: dict[int, set[DecisionDimension]] = {}
        query_map: dict[int, set[str]] = {}
        fused_by_index: dict[int, FusedResult] = {}
        # Best rerank score achieved by each chunk, and the pairing that did it.
        best_score: dict[int, float] = {}
        reserved: list[int] = []

        for rq in queries:
            expanded = " ".join([rq.query, *rq.expansions]).strip()
            fused = self._search_one(expanded)
            per_query.append(fused)
            stats.dense_results += sum(1 for f in fused if f.dense_rank is not None)
            stats.bm25_results += sum(1 for f in fused if f.bm25_rank is not None)

            for f in fused:
                dim_map.setdefault(f.index, set()).add(rq.dimension)
                query_map.setdefault(f.index, set()).add(rq.query)
                prior = fused_by_index.get(f.index)
                if prior is None or f.fusion_score > prior.fusion_score:
                    fused_by_index[f.index] = f

            # Phase 1: rerank this dimension's own candidates against its query.
            ranked = self._reranker.rerank(  # type: ignore[union-attr]
                [
                    RerankCandidate(
                        index=f.index,
                        chunk=self.chunks[f.index],
                        query=expanded,
                        dimension=rq.dimension,
                        dense_score=f.dense_score,
                        fusion_score=f.fusion_score,
                    )
                    for f in fused
                ]
            )
            kept = 0
            for result in ranked:
                if result.score > best_score.get(result.index, -1.0):
                    best_score[result.index] = result.score
                if kept >= cfg.per_dimension_k:
                    continue
                if result.score < cfg.min_rerank_score:
                    continue
                if result.index not in reserved:
                    reserved.append(result.index)
                kept += 1

        # Phase 2: fill from the cross-dimension fused pool.
        pool = merge_multi_query(per_query, k=cfg.rrf_k, top_k=cfg.fusion_top_k * 2)
        stats.fused_results = len(pool)
        for f in pool:
            if f.index not in fused_by_index:
                fused_by_index[f.index] = f

        limit = top_k or cfg.rerank_top_k
        selected = list(reserved)
        for f in pool:
            if len(selected) >= limit:
                break
            if f.index in selected:
                continue
            if best_score.get(f.index, 0.0) < cfg.min_rerank_score:
                stats.dropped_below_threshold += 1
                continue
            selected.append(f.index)

        # Strongest evidence first, so a reviewer reads the governing clause first.
        selected.sort(key=lambda i: best_score.get(i, 0.0), reverse=True)
        selected = selected[:limit]

        evidence: list[EvidenceItem] = []
        for index in selected:
            fused = fused_by_index[index]
            chunk = self.chunks[index]
            evidence.append(
                EvidenceItem(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page=chunk.page,
                    section=chunk.section,
                    heading=chunk.heading,
                    clause_ref=chunk.clause_ref,
                    source=chunk.source,
                    retrieval_score=round(fused.fusion_score, 6),
                    rerank_score=round(best_score.get(index, 0.0), 6),
                    dense_score=fused.dense_score,
                    bm25_score=fused.bm25_score,
                    dense_rank=fused.dense_rank,
                    bm25_rank=fused.bm25_rank,
                    fusion_score=round(fused.fusion_score, 6),
                    retrieval_method=fused.retrieval_method,  # type: ignore[arg-type]
                    matched_dimensions=sorted(dim_map.get(index, set()), key=lambda d: d.value),
                    queries=sorted(query_map.get(index, set())),
                )
            )

        stats.reranked_results = len(evidence)
        stats.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return evidence, stats

    # ------------------------------------------------------------------ #

    def get_chunk(self, chunk_id: str) -> PolicyChunk | None:
        idx = self._by_id.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    @property
    def ready(self) -> bool:
        return self._ready

    def backend_info(self) -> dict[str, str]:
        return {
            "embedding": self._embedder.name if self._embedder else "none",
            "dense_index": self._dense.backend,
            "sparse": "bm25",
            "reranker": self._reranker.name if self._reranker else "none",
            "chunks": str(len(self.chunks)),
        }


_RETRIEVER: HybridRetriever | None = None


def get_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Process-wide singleton; the index is loaded once per worker."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever(settings).ensure_ready()
    return _RETRIEVER


def reset_retriever() -> None:
    """Test hook."""
    global _RETRIEVER
    _RETRIEVER = None
