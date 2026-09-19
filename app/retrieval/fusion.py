"""Result fusion for hybrid retrieval.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) is used rather
than a weighted sum of raw scores, because BM25 scores are unbounded and
corpus-dependent while cosine similarities sit in [-1, 1]. Normalising those two
onto a common scale requires calibration that would have to be re-tuned whenever
the embedding backend changes. RRF only consumes *ranks*, so it is invariant to
both — which matters here precisely because the dense backend is pluggable.

A small normalised-score term is blended in as a tie-breaker so that, among
chunks at the same rank, the more strongly-matching one wins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.retrieval.bm25 import BM25Result
from app.retrieval.dense import DenseResult


@dataclass
class FusedResult:
    index: int
    fusion_score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    methods: set[str] = field(default_factory=set)

    @property
    def retrieval_method(self) -> str:
        if len(self.methods) > 1:
            return "fused"
        return next(iter(self.methods)) if self.methods else "fused"


def _normalise(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= 0:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def reciprocal_rank_fusion(
    dense: list[DenseResult],
    sparse: list[BM25Result],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    tie_break_weight: float = 0.05,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse one dense and one sparse ranking into a single ordering."""
    merged: dict[int, FusedResult] = {}

    dense_norm = _normalise({r.index: r.score for r in dense})
    sparse_norm = _normalise({r.index: r.score for r in sparse})
    scores: dict[int, float] = defaultdict(float)

    for rank, result in enumerate(dense, start=1):
        entry = merged.setdefault(result.index, FusedResult(index=result.index, fusion_score=0.0))
        entry.dense_score = result.score
        entry.dense_rank = rank
        entry.methods.add("dense")
        scores[result.index] += dense_weight * (1.0 / (k + rank))
        scores[result.index] += tie_break_weight * dense_weight * dense_norm.get(result.index, 0.0)

    for rank, result in enumerate(sparse, start=1):
        entry = merged.setdefault(result.index, FusedResult(index=result.index, fusion_score=0.0))
        entry.bm25_score = result.score
        entry.bm25_rank = rank
        entry.methods.add("bm25")
        scores[result.index] += bm25_weight * (1.0 / (k + rank))
        scores[result.index] += tie_break_weight * bm25_weight * sparse_norm.get(result.index, 0.0)

    for index, score in scores.items():
        merged[index].fusion_score = score

    ordered = sorted(merged.values(), key=lambda r: r.fusion_score, reverse=True)
    return ordered[:top_k] if top_k else ordered


def merge_multi_query(
    per_query: list[list[FusedResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse the per-dimension rankings into one candidate pool.

    The Case Analysis Agent emits one query per decision dimension, so this is
    what stops a chunk that is rank-1 for a single narrow dimension from
    dominating the pool: a chunk relevant to several dimensions accumulates
    reciprocal-rank mass from each of them.
    """
    merged: dict[int, FusedResult] = {}
    scores: dict[int, float] = defaultdict(float)

    for results in per_query:
        for rank, result in enumerate(results, start=1):
            entry = merged.get(result.index)
            if entry is None:
                entry = FusedResult(
                    index=result.index,
                    fusion_score=0.0,
                    dense_score=result.dense_score,
                    bm25_score=result.bm25_score,
                    dense_rank=result.dense_rank,
                    bm25_rank=result.bm25_rank,
                    methods=set(result.methods),
                )
                merged[result.index] = entry
            else:
                entry.methods |= result.methods
                # Keep the best evidence of each arm across queries.
                if result.dense_score is not None and (
                    entry.dense_score is None or result.dense_score > entry.dense_score
                ):
                    entry.dense_score = result.dense_score
                    entry.dense_rank = result.dense_rank
                if result.bm25_score is not None and (
                    entry.bm25_score is None or result.bm25_score > entry.bm25_score
                ):
                    entry.bm25_score = result.bm25_score
                    entry.bm25_rank = result.bm25_rank
            scores[result.index] += 1.0 / (k + rank)

    for index, score in scores.items():
        merged[index].fusion_score = score

    ordered = sorted(merged.values(), key=lambda r: r.fusion_score, reverse=True)
    return ordered[:top_k] if top_k else ordered
