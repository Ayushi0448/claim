"""Agent behaviour and state-transition tests."""

from __future__ import annotations

import pytest

from app.agents.case_analysis import CaseAnalysisAgent
from app.agents.coverage_exclusion import CoverageExclusionAgent
from app.models.schemas import (
    ClaimCase,
    Decision,
    DecisionDimension as D,
    FindingStatus,
    Severity,
)
from app.policy.evidence_binding import Anchor, resolve_anchors
from app.policy.limits import hospital_days


class TestCaseAnalysisAgent:
    def test_derives_waiting_period_facts(self, case_by_id):
        analysis, _ = CaseAnalysisAgent().run(ClaimCase(**case_by_id["PUB-002"]))
        assert analysis.derived_facts["days_since_policy_start"] == 19
        assert analysis.derived_facts["months_covered"] == 0

    def test_dimension_selection_is_claim_driven(self, case_by_id):
        agent = CaseAnalysisAgent()
        domiciliary, _ = agent.run(ClaimCase(**case_by_id["PUB-004"]))
        inpatient, _ = agent.run(ClaimCase(**case_by_id["PUB-001"]))

        assert D.DOMICILIARY in domiciliary.decision_dimensions
        assert D.DOMICILIARY not in inpatient.decision_dimensions
        assert D.HOSPITAL_DEFINITION in inpatient.decision_dimensions

    def test_day_care_dimension_for_short_admission(self, case_by_id):
        analysis, _ = CaseAnalysisAgent().run(ClaimCase(**case_by_id["PUB-005"]))
        assert D.DAY_CARE in analysis.decision_dimensions

    def test_experimental_dimension_only_when_flagged(self, case_by_id):
        agent = CaseAnalysisAgent()
        flagged, _ = agent.run(ClaimCase(**case_by_id["PUB-012"]))
        plain, _ = agent.run(ClaimCase(**case_by_id["PUB-001"]))
        assert D.EXPERIMENTAL_TREATMENT in flagged.decision_dimensions
        assert D.EXPERIMENTAL_TREATMENT not in plain.decision_dimensions

    def test_multiple_queries_never_one(self, case_by_id):
        """RULE 7: a claim is never resolved from a single retrieval query."""
        for case_id in ["PUB-001", "PUB-006", "PUB-011"]:
            _, queries = CaseAnalysisAgent().run(ClaimCase(**case_by_id[case_id]))
            assert len(queries) >= 5, f"{case_id} produced only {len(queries)} queries"
            assert len({q.query for q in queries}) == len(queries), "duplicate queries"

    def test_anchor_directed_queries_are_generated(self, case_by_id):
        """Regression test for F-06: rules declare the clauses they need."""
        _, queries = CaseAnalysisAgent().run(ClaimCase(**case_by_id["CUS-003"]))
        texts = {q.query for q in queries}
        assert "Dental treatment or surgery of any kind" in texts

    def test_missing_fields_detected(self, case_by_id):
        analysis, _ = CaseAnalysisAgent().run(ClaimCase(**case_by_id["PUB-006"]))
        assert any("medical_necessity_confirmed" in f for f in analysis.missing_fields)

    def test_irrelevant_attributes_are_identified_not_used(self, case_by_id):
        """RULE 9: non-policy attributes are named and discounted."""
        analysis, _ = CaseAnalysisAgent().run(ClaimCase(**case_by_id["CUS-010"]))
        assert analysis.irrelevant_attributes
        assert any("hospital.name" in a for a in analysis.irrelevant_attributes)

    def test_contradictory_dates_produce_a_warning(self):
        case = ClaimCase(
            case_id="T-WARN",
            policy_start_date="2026-01-01",
            claim_date="2026-03-01",
            continuous_coverage_months=40,  # disagrees with 2 months elapsed
            sum_insured_inr=500000,
        )
        analysis, _ = CaseAnalysisAgent().run(case)
        assert analysis.input_warnings

    def test_unparseable_date_is_reported_not_crashed(self):
        case = ClaimCase(case_id="T-DATE", policy_start_date="not-a-date", claim_date="2026-01-01")
        analysis, _ = CaseAnalysisAgent().run(case)
        assert any("Unparseable" in w for w in analysis.input_warnings)


class TestHospitalDays:
    @pytest.mark.parametrize(
        "hours,expected", [(96, 4), (24, 1), (25, 2), (168, 7), (8, 1), (0, 0), (None, 0)]
    )
    def test_billable_days_round_up(self, hours, expected):
        case = ClaimCase(case_id="T", treatment={"admission_hours": hours})
        assert hospital_days(case) == expected


class TestEvidenceBinding:
    def test_anchor_resolves_only_against_retrieved_text(self, analyses):
        evidence = analyses["PUB-003"].evidence
        hit = resolve_anchors(
            evidence, [Anchor(phrase="Pre-existing diseases will not be covered until 48 months")]
        )
        assert hit.resolved and hit.primary is not None

        miss = resolve_anchors(
            evidence, [Anchor(phrase="this sentence appears nowhere in the policy")]
        )
        assert not miss.resolved
        assert miss.missing_anchors

    def test_require_all_semantics(self, analyses):
        evidence = analyses["PUB-003"].evidence
        result = resolve_anchors(
            evidence,
            [
                Anchor(phrase="Pre-existing diseases will not be covered until 48 months"),
                Anchor(phrase="entirely fabricated clause text"),
            ],
            require_all=True,
        )
        assert not result.resolved

    def test_citations_are_built_only_from_retrieved_chunks(self, analyses):
        """RULE 3: no citation may reference a chunk retrieval did not return."""
        for case_id, response in analyses.items():
            retrieved = {e.chunk_id for e in response.evidence}
            for citation in response.citations:
                assert citation.chunk_id in retrieved, (
                    f"{case_id}: citation {citation.chunk_id} was never retrieved"
                )


class TestCoverageExclusionAgent:
    def test_rule_cannot_fire_without_its_clause(self, case_by_id):
        """The central guarantee: no evidence, no finding — only abstention."""
        case = ClaimCase(**case_by_id["PUB-003"])
        agent = CoverageExclusionAgent()
        from app.agents.case_analysis import CaseAnalysisAgent as CA

        analysis, _ = CA().run(case)
        assessment = agent.run(case, analysis, evidence=[])

        ped = [f for f in assessment.findings if f.rule_id == "pre_existing_disease_48m"]
        assert ped, "Rule should still be evaluated"
        assert ped[0].status == FindingStatus.UNRESOLVED
        assert ped[0].citations == []
        assert ped[0].evidence_supported is False

    def test_limits_are_not_applied_without_their_clause(self, case_by_id):
        case = ClaimCase(**case_by_id["PUB-007"])
        from app.agents.case_analysis import CaseAnalysisAgent as CA

        analysis, _ = CA().run(case)
        assessment = CoverageExclusionAgent().run(case, analysis, evidence=[])
        assert assessment.applied_limits == [], "A cap was applied without a citable clause"

    def test_room_rent_arithmetic(self, analyses):
        """1% of 500,000 = 5,000/day x 4 days = 20,000 against 30,000 claimed."""
        limits = {l.limit_id: l for l in analyses["PUB-001"].applicable_limits}
        room = limits["limit_room_rent"]
        assert room.limit_amount_inr == pytest.approx(20000)
        assert room.claimed_amount_inr == pytest.approx(30000)
        assert room.deduction_inr == pytest.approx(10000)
        assert room.binding is True

    def test_ambulance_takes_the_lesser_cap(self, analyses):
        """1% of 1,000,000 is 10,000, but the policy caps at 1,000."""
        limits = {l.limit_id: l for l in analyses["PUB-007"].applicable_limits}
        ambulance = limits["limit_ambulance"]
        assert ambulance.limit_amount_inr == pytest.approx(1000)
        assert ambulance.deduction_inr == pytest.approx(500)

    def test_icu_uses_its_own_per_day_rate(self, analyses):
        """2% of 300,000 = 6,000/day x 7 days = 42,000 against 90,000 claimed."""
        limits = {l.limit_id: l for l in analyses["CUS-002"].applicable_limits}
        icu = limits["limit_icu"]
        assert icu.limit_amount_inr == pytest.approx(42000)
        assert icu.deduction_inr == pytest.approx(48000)

    def test_domiciliary_uses_aggregate_cap_not_category_caps(self, analyses):
        limits = {l.limit_id for l in analyses["PUB-004"].applicable_limits}
        assert "limit_domiciliary_aggregate" in limits
        assert "limit_room_rent" not in limits

    def test_every_applied_limit_carries_a_citation(self, analyses):
        for case_id, response in analyses.items():
            for limit in response.applicable_limits:
                assert limit.citations, f"{case_id}: {limit.limit_id} has no citation"
                assert limit.evidence_chunk_ids


class TestDecisionAgent:
    def test_blocking_violation_yields_rejection(self, analyses):
        response = analyses["PUB-008"]
        assert response.decision == Decision.NOT_ADMISSIBLE
        assert any(
            f.severity == Severity.BLOCKING and f.status == FindingStatus.VIOLATED
            for f in response.key_findings
        )

    def test_unresolved_condition_yields_abstention(self, analyses):
        response = analyses["PUB-011"]
        assert response.decision == Decision.NEEDS_REVIEW
        assert response.abstained
        assert "INSUFFICIENT_EVIDENCE" in (response.abstain_reason or "")

    def test_binding_cap_yields_admissible_with_limits(self, analyses):
        response = analyses["PUB-001"]
        assert response.decision == Decision.ADMISSIBLE_WITH_LIMITS
        assert any(l.binding for l in response.applicable_limits)

    def test_clean_claim_is_admissible(self, analyses):
        response = analyses["CUS-001"]
        assert response.decision == Decision.ADMISSIBLE
        assert response.total_deduction_inr == pytest.approx(0.0)

    def test_partial_exclusion_yields_partially_admissible(self, analyses):
        response = analyses["CUS-006"]
        assert response.decision == Decision.PARTIALLY_ADMISSIBLE

    def test_decisive_finding_is_ranked_first(self, analyses):
        response = analyses["PUB-002"]
        assert response.key_findings[0].severity == Severity.BLOCKING
        assert response.key_findings[0].status == FindingStatus.VIOLATED

    def test_confidence_is_computed_not_asserted(self, analyses):
        breakdown = analyses["PUB-001"].confidence_breakdown
        assert breakdown.weights
        expected_raw = sum(
            breakdown.weights[k] * getattr(breakdown, k)
            for k in ["retrieval_quality", "evidence_coverage", "citation_support",
                      "decision_consistency"]
        )
        assert breakdown.raw_score == pytest.approx(expected_raw, abs=1e-3)

    def test_confidence_lower_when_abstaining(self, analyses):
        assert analyses["CUS-005"].confidence < analyses["CUS-001"].confidence


class TestWorkflowState:
    def test_all_five_agents_run_in_order(self, analyses):
        for case_id, response in analyses.items():
            agents = [e.agent for e in response.trace]
            assert agents[:5] == [
                "case_analysis", "policy_evidence", "coverage_exclusion", "decision", "validation"
            ], f"{case_id}: unexpected agent order {agents}"

    def test_trace_exposes_metrics_not_reasoning(self, analyses):
        """RULE 10: the trace must carry auditable facts, never chain-of-thought."""
        forbidden = {"reasoning", "thought", "chain_of_thought", "rationale_text", "prompt"}
        for case_id, response in analyses.items():
            for event in response.trace:
                assert set(event.metrics) & forbidden == set(), f"{case_id}: trace leaked reasoning"
                assert len(event.action) < 120
                assert event.elapsed_ms >= 0

    def test_retrieval_trace_reports_arm_counts(self, analyses):
        event = next(e for e in analyses["PUB-001"].trace if e.agent == "policy_evidence")
        for key in ["dense_results", "bm25_results", "fused_results", "reranked_results"]:
            assert key in event.metrics

    def test_structured_state_is_typed_not_free_text(self, analyses):
        response = analyses["PUB-001"]
        assert response.case_analysis is not None
        assert isinstance(response.case_analysis.decision_dimensions, list)
        assert all(isinstance(d, D) for d in response.case_analysis.decision_dimensions)
        assert all(isinstance(f.status, FindingStatus) for f in response.key_findings)
