"""Reranking stage.

The retrieval arms score a chunk *independently* of the query (bi-encoder
cosine) or with a bag-of-words assumption (BM25). A reranker's job is to look at
the query and the chunk **together** and re-score the shortlist with a more
expensive function. That is the property that matters, not whether the function
happens to be a neural network.

Two implementations, selected by ``RERANKER_BACKEND``:

* ``CrossEncoderReranker`` — a BGE cross-encoder, used when weights load.
* ``LexicalSemanticReranker`` — the offline default. It scores query-document
  interaction directly: IDF-weighted term coverage, bigram/phrase adjacency,
  numeric-and-unit agreement (decisive in a policy full of "48 months", "1.0%",
  "24 hours"), a section prior derived from the decision dimension, and the
  dense similarity as one input among several.

The lexical reranker is weaker than a trained cross-encoder on paraphrase, and
the README says so plainly. It is deterministic, needs no download, and is
reproducible in CI — which is what makes the evaluation numbers trustworthy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import ModelSettings, get_settings
from app.models.schemas import DecisionDimension, PolicyChunk
from app.retrieval.bm25 import BM25Index
from app.utils.text import bigrams, tokenize

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    index: int
    chunk: PolicyChunk
    query: str
    dimension: DecisionDimension | None = None
    dense_score: float | None = None
    fusion_score: float = 0.0


@dataclass
class RerankResult:
    index: int
    score: float
    components: dict[str, float]


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, candidates: list[RerankCandidate]) -> list[RerankResult]: ...


# Which policy sections are a priori relevant to which decision dimension.
# This encodes document structure, not insurance knowledge: it says "waiting
# period questions are answered in the exclusions section of this document",
# which is a fact about the supplied PDF's layout.
_SECTION_PRIORS: dict[DecisionDimension, dict[str, float]] = {
    DecisionDimension.HOSPITALIZATION_DEFINITION: {"Definitions": 1.0, "Scope of Cover": 0.6},
    DecisionDimension.HOSPITAL_DEFINITION: {"Definitions": 1.0},
    DecisionDimension.SCOPE_OF_COVER: {"Scope of Cover": 1.0, "Preamble": 0.5},
    DecisionDimension.DAY_CARE: {"Definitions": 0.8, "Scope of Cover": 1.0},
    DecisionDimension.DOMICILIARY: {"Definitions": 0.9, "Scope of Cover": 1.0, "What We Exclude": 0.7},
    DecisionDimension.WAITING_PERIOD_INITIAL: {"What We Exclude": 1.0},
    DecisionDimension.WAITING_PERIOD_FIRST_YEAR: {"What We Exclude": 1.0},
    DecisionDimension.PRE_EXISTING_DISEASE: {"What We Exclude": 1.0, "Definitions": 0.7},
    DecisionDimension.PORTABILITY_CONTINUITY: {
        "What We Exclude": 0.9, "Standard Terms and Conditions": 1.0, "Definitions": 0.6,
    },
    DecisionDimension.EXCLUSIONS: {"What We Exclude": 1.0},
    DecisionDimension.EXPERIMENTAL_TREATMENT: {"Definitions": 1.0, "What We Exclude": 0.9},
    DecisionDimension.ROOM_RENT_LIMIT: {"Scope of Cover": 1.0, "Definitions": 0.5},
    DecisionDimension.CATEGORY_SUBLIMITS: {"Scope of Cover": 1.0},
    DecisionDimension.AMBULANCE_LIMIT: {"Scope of Cover": 1.0},
    DecisionDimension.PRE_POST_HOSPITALIZATION: {"Scope of Cover": 1.0, "Definitions": 0.8},
    DecisionDimension.MEDICAL_NECESSITY: {"Definitions": 1.0},
    DecisionDimension.SUM_INSURED_AGGREGATE: {"Preamble": 0.9, "Scope of Cover": 1.0},
}

_NUMERIC_TOKEN = {"%", "months", "month", "years", "year", "days", "day", "hours", "hrs", "hour"}


class LexicalSemanticReranker:
    """Deterministic cross-encoder-style scorer. No model download required."""

    name = "lexical_semantic"

    def __init__(self, bm25: BM25Index | None = None) -> None:
        self._bm25 = bm25

    # ------------------------------------------------------------------ #

    def _idf(self, term: str) -> float:
        if self._bm25 is None:
            return 1.0
        # Floor at a small positive value so unseen query terms still count.
        return max(0.15, self._bm25.idf(term))

    def _term_coverage(self, q_tokens: list[str], d_tokens: set[str]) -> float:
        if not q_tokens:
            return 0.0
        total = sum(self._idf(t) for t in q_tokens)
        if total <= 0:
            return 0.0
        hit = sum(self._idf(t) for t in q_tokens if t in d_tokens)
        return hit / total

    @staticmethod
    def _phrase_score(q_bigrams: list[str], d_bigrams: set[str]) -> float:
        if not q_bigrams:
            return 0.0
        return sum(1.0 for b in q_bigrams if b in d_bigrams) / len(q_bigrams)

    @staticmethod
    def _numeric_agreement(q_tokens: list[str], d_tokens: list[str]) -> float:
        """Reward chunks sharing the query's numbers *and* their units.

        "48 months" matching "48 months" is near-decisive for a pre-existing
        disease question; "48" alone is not.
        """
        q_nums = {t for t in q_tokens if any(ch.isdigit() for ch in t)}
        if not q_nums:
            return 0.0
        d_nums = {t for t in d_tokens if any(ch.isdigit() for ch in t)}
        overlap = q_nums & d_nums
        if not overlap:
            return 0.0
        base = len(overlap) / len(q_nums)
        d_set = set(d_tokens)
        unit_bonus = 0.25 if (d_set & _NUMERIC_TOKEN) and (set(q_tokens) & _NUMERIC_TOKEN) else 0.0
        return min(1.0, base + unit_bonus)

    @staticmethod
    def _section_prior(dimension: DecisionDimension | None, section: str) -> float:
        if dimension is None:
            return 0.5
        priors = _SECTION_PRIORS.get(dimension)
        if not priors:
            return 0.5
        return priors.get(section, 0.15)

    @staticmethod
    def _length_penalty(char_count: int, *, is_complete_clause: bool = False) -> float:
        """Very short fragments and very long blobs are both poor evidence.

        Exception: a numbered clause is a *complete legal statement* however
        short it is. "7. Dental treatment or surgery of any kind." is 42
        characters and completely disposes of a claim. Penalising it as a
        fragment pushed it out of retrieval and made the engine abstain on a
        clear-cut exclusion (docs/failure-analysis.md F-05), so chunks carrying
        a clause reference are exempt from the short-text penalty.
        """
        if char_count < 80 and not is_complete_clause:
            return 0.6
        if char_count > 1400:
            return 0.85
        return 1.0

    # ------------------------------------------------------------------ #

    def rerank(self, candidates: list[RerankCandidate]) -> list[RerankResult]:
        results: list[RerankResult] = []
        for cand in candidates:
            q_tokens = tokenize(cand.query)
            d_tokens = tokenize(cand.chunk.text + " " + cand.chunk.heading)
            d_token_set = set(d_tokens)
            d_bigrams = set(bigrams(d_tokens))

            coverage = self._term_coverage(q_tokens, d_token_set)
            phrase = self._phrase_score(bigrams(q_tokens), d_bigrams)
            numeric = self._numeric_agreement(q_tokens, d_tokens)
            prior = self._section_prior(cand.dimension, cand.chunk.section)
            dense = max(0.0, cand.dense_score or 0.0)

            raw = (
                0.42 * coverage
                + 0.18 * phrase
                + 0.16 * numeric
                + 0.14 * prior
                + 0.10 * dense
            ) * self._length_penalty(
                cand.chunk.char_count,
                is_complete_clause=bool(cand.chunk.clause_ref),
            )

            # Squash into a stable 0-1 band so MIN_RERANK_SCORE is meaningful
            # regardless of which dense backend produced the candidate.
            score = 1.0 / (1.0 + math.exp(-8.0 * (raw - 0.35)))

            results.append(
                RerankResult(
                    index=cand.index,
                    score=round(score, 6),
                    components={
                        "term_coverage": round(coverage, 4),
                        "phrase_match": round(phrase, 4),
                        "numeric_agreement": round(numeric, 4),
                        "section_prior": round(prior, 4),
                        "dense_similarity": round(dense, 4),
                        "raw": round(raw, 4),
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results


class CrossEncoderReranker:
    """BGE cross-encoder reranker, used when the weights are reachable."""

    name = "cross_encoder"

    def __init__(self, model_name: str, fallback: Reranker | None = None) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = fallback

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(self, candidates: list[RerankCandidate]) -> list[RerankResult]:
        if not candidates:
            return []
        try:
            self._ensure()
            pairs = [(c.query, c.chunk.text) for c in candidates]
            raw_scores = self._model.predict(pairs)
            results = [
                RerankResult(
                    index=c.index,
                    score=float(1.0 / (1.0 + math.exp(-float(s)))),
                    components={"cross_encoder_logit": float(s)},
                )
                for c, s in zip(candidates, raw_scores)
            ]
            results.sort(key=lambda r: r.score, reverse=True)
            return results
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Cross-encoder rerank failed (%s); using lexical reranker", exc)
            if self._fallback is None:
                self._fallback = LexicalSemanticReranker()
            return self._fallback.rerank(candidates)


def _cross_encoder_available(model_name: str) -> bool:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except Exception:
        return False
    try:
        CrossEncoder(model_name)
        return True
    except Exception as exc:
        logger.info(
            "Reranker model %r unavailable (%s); using lexical reranker",
            model_name, type(exc).__name__,
        )
        return False


def get_reranker(
    settings: ModelSettings | None = None, *, bm25: BM25Index | None = None
) -> Reranker:
    cfg = settings or get_settings().models
    lexical = LexicalSemanticReranker(bm25=bm25)
    choice = cfg.reranker_backend

    if choice in {"lexical", "heuristic"}:
        return lexical
    if choice in {"cross_encoder", "bge", "cross-encoder"}:
        return CrossEncoderReranker(cfg.reranker_model, fallback=lexical)

    if _cross_encoder_available(cfg.reranker_model):
        logger.info("Reranker: cross-encoder %s", cfg.reranker_model)
        return CrossEncoderReranker(cfg.reranker_model, fallback=lexical)
    return lexical
