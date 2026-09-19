"""Retrieval tests: BM25, dense, fusion, reranking and the hybrid pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas import DecisionDimension as D, RetrievalQuery
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.fusion import merge_multi_query, reciprocal_rank_fusion
from app.retrieval.reranker import LexicalSemanticReranker, RerankCandidate
from app.utils.text import contains_phrase

CORPUS = [
    "Pre-existing diseases will not be covered until 48 months of continuous coverage have elapsed.",
    "A waiting period of 30 days will apply to all claims unless continuously insured.",
    "Normal Room expenses: 1.0% of Basic Sum Insured per day.",
    "Ambulance charges limited to 1.0% of Basic Sum Insured or Rupees 1000 whichever is less.",
    "The geographical scope of this Policy will be India and all claims payable in Indian currency.",
]


class TestBM25:
    @pytest.fixture(scope="class")
    def index(self):
        return BM25Index().fit(CORPUS)

    def test_ranks_lexically_matching_document_first(self, index):
        results = index.search("pre-existing disease 48 months waiting", top_k=3)
        assert results and results[0].index == 0

    def test_numeric_tokens_are_preserved(self, index):
        """Percentages and durations are decisive evidence in a policy."""
        results = index.search("1.0% of Basic Sum Insured room", top_k=3)
        assert results[0].index == 2
        assert any("1.0%" in term or "1" in term for term in results[0].matched_terms)

    def test_returns_nothing_for_unmatched_query(self, index):
        assert index.search("xylophone quantum submarine", top_k=5) == []

    def test_empty_query_is_safe(self, index):
        assert index.search("", top_k=5) == []

    def test_idf_is_higher_for_rarer_terms(self, index):
        # "insured" appears in three documents of the toy corpus; "ambulance"
        # in one, so the rare term must carry more weight.
        assert index.idf("ambulance") > index.idf("insured")
        assert index.idf("never-appears-anywhere") == 0.0

    def test_state_round_trip(self, index):
        restored = BM25Index.from_state(index.state_dict())
        assert restored.n_docs == index.n_docs
        a = index.search("waiting period 30 days", top_k=2)
        b = restored.search("waiting period 30 days", top_k=2)
        assert [r.index for r in a] == [r.index for r in b]


class TestDenseIndex:
    def test_cosine_search_returns_nearest(self):
        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32
        )
        index = DenseIndex(use_faiss=False).build(vectors)
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
        assert results[0].index == 0
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        assert [r.index for r in results] == sorted(
            [r.index for r in results], key=lambda i: -results[[x.index for x in results].index(i)].score
        )

    def test_empty_index_returns_nothing(self):
        assert DenseIndex().search(np.array([1.0, 0.0], dtype=np.float32)) == []

    def test_zero_query_vector_is_safe(self):
        index = DenseIndex(use_faiss=False).build(np.array([[1.0, 0.0]], dtype=np.float32))
        assert index.search(np.array([0.0, 0.0], dtype=np.float32)) == []

    def test_top_k_larger_than_corpus(self):
        index = DenseIndex(use_faiss=False).build(np.array([[1.0, 0.0]], dtype=np.float32))
        assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=50)) == 1


class TestFusion:
    def test_document_found_by_both_arms_outranks_either_alone(self):
        from app.retrieval.bm25 import BM25Result
        from app.retrieval.dense import DenseResult

        dense = [DenseResult(index=1, score=0.9), DenseResult(index=2, score=0.8)]
        sparse = [BM25Result(index=3, score=9.0), BM25Result(index=1, score=8.0)]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        assert fused[0].index == 1, "Document in both rankings should fuse to the top"
        assert fused[0].retrieval_method == "fused"

    def test_single_arm_results_are_labelled(self):
        from app.retrieval.dense import DenseResult

        fused = reciprocal_rank_fusion([DenseResult(index=5, score=0.5)], [], k=60)
        assert fused[0].retrieval_method == "dense"
        assert fused[0].bm25_score is None

    def test_rrf_is_scale_invariant(self):
        """BM25 scores are unbounded; fusion must not be swayed by magnitude."""
        from app.retrieval.bm25 import BM25Result
        from app.retrieval.dense import DenseResult

        dense = [DenseResult(index=1, score=0.9), DenseResult(index=2, score=0.85)]
        small = [BM25Result(index=2, score=2.0), BM25Result(index=1, score=1.0)]
        huge = [BM25Result(index=2, score=2000.0), BM25Result(index=1, score=1000.0)]
        assert [f.index for f in reciprocal_rank_fusion(dense, small, k=60)] == [
            f.index for f in reciprocal_rank_fusion(dense, huge, k=60)
        ]

    def test_multi_query_merge_rewards_cross_dimension_relevance(self):
        from app.retrieval.fusion import FusedResult

        q1 = [FusedResult(index=7, fusion_score=0.1), FusedResult(index=1, fusion_score=0.09)]
        q2 = [FusedResult(index=2, fusion_score=0.1), FusedResult(index=7, fusion_score=0.09)]
        merged = merge_multi_query([q1, q2], k=60)
        assert merged[0].index == 7

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], [], k=60) == []
        assert merge_multi_query([], k=60) == []


class TestReranker:
    def test_query_document_interaction_beats_topical_similarity(self, chunks):
        reranker = LexicalSemanticReranker()
        target = next(c for c in chunks if contains_phrase(c.text, "48 months"))
        other = next(c for c in chunks if c.section == "Grievance and Ombudsman")
        query = "pre-existing disease 48 months of continuous coverage"
        results = reranker.rerank([
            RerankCandidate(index=0, chunk=target, query=query, dimension=D.PRE_EXISTING_DISEASE),
            RerankCandidate(index=1, chunk=other, query=query, dimension=D.PRE_EXISTING_DISEASE),
        ])
        assert results[0].index == 0
        assert results[0].score > results[1].score

    def test_scores_are_bounded(self, chunks):
        reranker = LexicalSemanticReranker()
        results = reranker.rerank([
            RerankCandidate(index=i, chunk=c, query="room rent limit", dimension=D.ROOM_RENT_LIMIT)
            for i, c in enumerate(chunks[:20])
        ])
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_short_complete_clause_is_not_penalised(self, chunks):
        """Regression test for F-05: a 43-character exclusion is decisive."""
        short_clause = next(
            c for c in chunks
            if c.clause_ref == "Exclusion 7" and contains_phrase(c.text, "Dental treatment")
        )
        assert short_clause.char_count < 80
        reranker = LexicalSemanticReranker()
        result = reranker.rerank([
            RerankCandidate(
                index=0, chunk=short_clause,
                query="Dental treatment or surgery of any kind",
                dimension=D.EXCLUSIONS,
            )
        ])
        assert result[0].score > 0.7, "Short but complete clause was penalised"

    def test_output_is_sorted_descending(self, chunks):
        reranker = LexicalSemanticReranker()
        results = reranker.rerank([
            RerankCandidate(index=i, chunk=c, query="waiting period", dimension=D.EXCLUSIONS)
            for i, c in enumerate(chunks[:25])
        ])
        assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)

    def test_empty_candidate_list(self):
        assert LexicalSemanticReranker().rerank([]) == []


class TestHybridPipeline:
    def test_index_is_populated(self, retriever):
        assert retriever.ready
        assert len(retriever.chunks) > 100

    @pytest.mark.parametrize(
        "dimension,query,expected_phrase",
        [
            (D.PRE_EXISTING_DISEASE, "pre-existing disease waiting period 48 months",
             "Pre-existing diseases will not be covered until 48 months"),
            (D.WAITING_PERIOD_INITIAL, "30 day waiting period all claims",
             "waiting period of 30 days will apply to all claims"),
            (D.ROOM_RENT_LIMIT, "room rent limit per day percentage of sum insured",
             "Normal Room expenses: 1.0% of Basic Sum Insured"),
            (D.HOSPITAL_DEFINITION, "definition of hospital registered minimum criteria beds",
             "Hospital means any institution established for in-patient care"),
            (D.DOMICILIARY, "domiciliary hospitalisation aggregate sub-limit",
             "maximum aggregate sub-limit of 20% of the Basic Sum Insured"),
            (D.EXPERIMENTAL_TREATMENT, "unproven experimental treatment definition",
             "Unproven/Experimental Treatment means"),
            (D.AMBULANCE_LIMIT, "ambulance charges limit rupees 1000",
             "Ambulance charges in connection with any admissible claim limited to"),
        ],
    )
    def test_governing_clause_is_retrieved(self, retriever, dimension, query, expected_phrase):
        evidence, _ = retriever.retrieve(
            [RetrievalQuery(dimension=dimension, query=query, rationale="test")], top_k=5
        )
        assert any(contains_phrase(e.text, expected_phrase) for e in evidence), (
            f"{dimension.value}: expected clause not retrieved"
        )

    def test_evidence_carries_full_provenance_and_scores(self, retriever):
        evidence, stats = retriever.retrieve(
            [RetrievalQuery(dimension=D.ROOM_RENT_LIMIT, query="room rent 1% sum insured",
                            rationale="test")],
            top_k=5,
        )
        assert evidence
        for item in evidence:
            assert item.chunk_id and item.source and item.section and item.heading
            assert item.page >= 1
            assert 0.0 <= item.rerank_score <= 1.0
            assert item.retrieval_method in {"dense", "bm25", "fused"}
            assert item.matched_dimensions
        assert stats.queries_executed == 1
        assert stats.reranked_results == len(evidence)

    def test_both_arms_contribute(self, retriever):
        evidence, stats = retriever.retrieve(
            [RetrievalQuery(dimension=D.EXCLUSIONS,
                            query="cosmetic aesthetic treatment plastic surgery excluded",
                            rationale="test")],
            top_k=8,
        )
        assert stats.dense_results > 0, "Dense arm contributed nothing"
        assert stats.bm25_results > 0, "Sparse arm contributed nothing"

    def test_per_dimension_recall_guarantee(self, retriever):
        """Regression test for F-01: narrow dimensions must survive a wide sweep."""
        queries = [
            RetrievalQuery(dimension=D.AMBULANCE_LIMIT,
                           query="ambulance charges limited to rupees 1000", rationale="t"),
            RetrievalQuery(dimension=D.PRE_EXISTING_DISEASE,
                           query="pre-existing diseases 48 months continuous coverage", rationale="t"),
            RetrievalQuery(dimension=D.EXCLUSIONS,
                           query="what we exclude cosmetic dental pregnancy HIV war nuclear", rationale="t"),
            RetrievalQuery(dimension=D.DOMICILIARY,
                           query="domiciliary treatment at home 20% sub-limit", rationale="t"),
            RetrievalQuery(dimension=D.ROOM_RENT_LIMIT,
                           query="normal room expenses 1.0% basic sum insured", rationale="t"),
        ]
        evidence, _ = retriever.retrieve(queries, top_k=20)
        covered = {d for item in evidence for d in item.matched_dimensions}
        for query in queries:
            assert query.dimension in covered, f"{query.dimension.value} got no evidence"

    def test_empty_query_list(self, retriever):
        evidence, stats = retriever.retrieve([])
        assert evidence == [] and stats.queries_executed == 0

    def test_persisted_index_round_trip(self, settings):
        from app.retrieval.index import HybridRetriever

        fresh = HybridRetriever()
        assert fresh.load(), "Persisted index failed to load"
        assert len(fresh.chunks) > 100
        evidence, _ = fresh.retrieve(
            [RetrievalQuery(dimension=D.PRE_EXISTING_DISEASE,
                            query="pre-existing 48 months", rationale="t")],
            top_k=3,
        )
        assert evidence
