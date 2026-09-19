"""Agent 2 — Policy Evidence.

Responsibility: convert the investigation plan into policy evidence.

It owns the retrieval stack — dense + BM25 + fusion + reranking — and returns
ranked evidence carrying full provenance (page, section, heading, chunk id) and
full score attribution (dense score/rank, BM25 score/rank, fusion score, rerank
score), which is what makes retrieval quality measurable in the evaluation.

It deliberately performs no interpretation: it does not decide what the clauses
mean. That separation is what allows the coverage agent's conclusions to be
checked against the evidence this agent returned.
"""

from __future__ import annotations

import logging

from app.models.schemas import EvidenceItem, RetrievalQuery
from app.retrieval.index import HybridRetriever, RetrievalStats

logger = logging.getLogger(__name__)

# Clauses that must be available for the rule engine to ground its core checks.
# Retrieving them for every case guarantees the engine can always *evaluate*
# admissibility; without them it would abstain for the wrong reason (a
# retrieval gap rather than a genuine evidence gap).
_FOUNDATION_QUERY_FLOOR = 3


class PolicyEvidenceAgent:
    name = "policy_evidence"

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def run(
        self, queries: list[RetrievalQuery], *, top_k: int | None = None
    ) -> tuple[list[EvidenceItem], RetrievalStats]:
        if not queries:
            return [], RetrievalStats()

        # Evidence budget scales with the breadth of the investigation: a claim
        # engaging 12 dimensions needs more evidence than one engaging 4, but
        # the pool stays bounded so the reviewer can actually read it.
        budget = top_k or max(
            self.retriever.settings.retrieval.rerank_top_k,
            min(len(queries) * 2, 28),
        )

        evidence, stats = self.retriever.retrieve(queries, top_k=budget)

        if len(queries) >= _FOUNDATION_QUERY_FLOOR and not evidence:
            logger.warning(
                "Retrieval returned no evidence above threshold for %d queries", len(queries)
            )
        return evidence, stats

    @staticmethod
    def coverage_by_dimension(
        queries: list[RetrievalQuery], evidence: list[EvidenceItem]
    ) -> dict[str, int]:
        """How many evidence items each dimension attracted.

        A dimension with zero evidence is a retrieval gap and is surfaced in the
        trace so it can be distinguished from a genuine policy silence.
        """
        counts = {q.dimension.value: 0 for q in queries}
        for item in evidence:
            for dim in item.matched_dimensions:
                if dim.value in counts:
                    counts[dim.value] += 1
        return counts
