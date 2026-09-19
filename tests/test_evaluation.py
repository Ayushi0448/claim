"""Evaluation-harness tests and the end-to-end accuracy gate.

These tests assert against the hand-derived ground truth in
``evaluation/expected_results/``. If a change to retrieval or the rules breaks a
case, it fails here rather than silently degrading the reported metrics.
"""

from __future__ import annotations

import json

import pytest

from app.models.schemas import Decision
from evaluation.metrics import (
    CaseResult,
    aggregate,
    citation_correctness,
    citation_hit_rate,
    evidence_recall_at_k,
)
from evaluation.run_evaluation import score_case


class TestSuppliedCasesUnmodified:
    def test_all_twelve_public_cases_present(self, public_cases):
        assert len(public_cases) == 12
        assert [c["case_id"] for c in public_cases] == [f"PUB-{i:03d}" for i in range(1, 13)]

    def test_public_cases_retain_their_original_shape(self, public_cases):
        """Guards the instruction not to modify the supplied cases."""
        for case in public_cases:
            assert case["policy_id"] == "USGIC-CSC-2017-2018"
            for field in ["case_id", "policy_start_date", "claim_date", "sum_insured_inr",
                          "patient", "hospital", "treatment", "expenses_inr", "documents", "task"]:
                assert field in case, f"{case['case_id']} is missing {field}"

    def test_custom_cases_are_separate_and_sufficient(self, custom_cases):
        assert len(custom_cases) >= 5, "At least five candidate cases are required"
        assert all(c["case_id"].startswith("CUS-") for c in custom_cases)


class TestGroundTruth:
    def test_every_case_has_a_label(self, public_cases, custom_cases,
                                    expected_public, expected_custom):
        for case in public_cases:
            assert case["case_id"] in expected_public
        for case in custom_cases:
            assert case["case_id"] in expected_custom

    def test_labels_are_valid_statuses(self, expected_public, expected_custom):
        allowed = {d.value for d in Decision}
        for spec in {**expected_public, **expected_custom}.values():
            assert spec["expected_decision"] in allowed

    def test_every_label_cites_a_clause(self, expected_public, expected_custom):
        """Ground truth must be justified, not asserted."""
        for case_id, spec in {**expected_public, **expected_custom}.items():
            assert spec.get("clause_basis"), f"{case_id} has no clause justification"
            assert len(spec["clause_basis"]) >= 1

    def test_at_least_two_abstention_cases(self, expected_public, expected_custom):
        combined = {**expected_public, **expected_custom}
        abstentions = [c for c, s in combined.items() if s["expected_decision"] == "NEEDS_REVIEW"]
        assert len(abstentions) >= 2

    def test_all_five_statuses_are_exercised(self, expected_public, expected_custom):
        combined = {**expected_public, **expected_custom}
        covered = {s["expected_decision"] for s in combined.values()}
        assert covered == {d.value for d in Decision}, f"Not exercised: {set(d.value for d in Decision) - covered}"


class TestMetrics:
    def test_evidence_recall_detects_a_missing_clause(self, analyses):
        response = analyses["PUB-001"]
        recall, found, missing = evidence_recall_at_k(
            response,
            [
                {"must_contain": "Normal Room expenses: 1.0% of Basic Sum Insured", "page": 7},
                {"must_contain": "a clause that is not in this policy at all", "page": 7},
            ],
        )
        assert recall == pytest.approx(0.5)
        assert len(found) == 1 and len(missing) == 1

    def test_recall_checks_the_page_too(self, analyses):
        response = analyses["PUB-001"]
        recall, _, _ = evidence_recall_at_k(
            response,
            [{"must_contain": "Normal Room expenses: 1.0% of Basic Sum Insured", "page": 99}],
        )
        assert recall == 0.0, "A clause matched on the wrong page must not count"

    def test_recall_is_none_without_requirements(self, analyses):
        recall, _, _ = evidence_recall_at_k(analyses["PUB-001"], [])
        assert recall is None

    def test_citation_correctness_penalises_tampering(self, analyses):
        import copy

        response = copy.deepcopy(analyses["PUB-001"])
        assert citation_correctness(response) == pytest.approx(1.0)
        response.citations[0].page = 99
        assert citation_correctness(response) < 1.0

    def test_citation_hit_rate_requires_actual_use(self, analyses):
        rate = citation_hit_rate(
            analyses["PUB-003"],
            [{"must_contain": "Pre-existing diseases will not be covered until 48 months",
              "page": 8}],
        )
        assert rate == pytest.approx(1.0)

    def test_aggregate_computes_abstention_confusion(self):
        results = [
            CaseResult("A", "t", "NEEDS_REVIEW", "NEEDS_REVIEW", True, 0.5, "PASS",
                       None, 0.0, None, 5, 2, 1.0, abstained=True, expected_abstention=True),
            CaseResult("B", "t", "ADMISSIBLE", "NEEDS_REVIEW", False, 0.4, "PASS",
                       None, 0.0, None, 5, 2, 1.0, abstained=True, expected_abstention=False),
            CaseResult("C", "t", "ADMISSIBLE", "ADMISSIBLE", True, 0.9, "PASS",
                       None, 0.0, None, 5, 2, 1.0, abstained=False, expected_abstention=False),
        ]
        report = aggregate(results)
        # aggregate() rounds to 4 decimal places for report readability.
        assert report["decision_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
        assert report["abstention"]["true_positives"] == 1
        assert report["abstention"]["false_positives"] == 1
        assert report["abstention"]["recall"] == pytest.approx(1.0)


class TestEndToEndAccuracy:
    """The accuracy gate. A regression in retrieval or rules fails here."""

    @pytest.fixture(scope="class")
    def scored(self, analyses, expected_public, expected_custom):
        combined = {**expected_public, **expected_custom}
        return {
            case_id: score_case(
                response, combined.get(case_id),
                "public" if case_id.startswith("PUB") else "custom",
            )
            for case_id, response in analyses.items()
        }

    def test_every_public_case_matches_ground_truth(self, scored):
        failures = [
            f"{r.case_id}: expected {r.expected_decision}, produced {r.actual_decision}"
            for r in scored.values()
            if r.suite == "public" and r.decision_correct is False
        ]
        assert not failures, "Public-suite regressions: " + "; ".join(failures)

    def test_every_custom_case_matches_ground_truth(self, scored):
        failures = [
            f"{r.case_id}: expected {r.expected_decision}, produced {r.actual_decision}"
            for r in scored.values()
            if r.suite == "custom" and r.decision_correct is False
        ]
        assert not failures, "Custom-suite regressions: " + "; ".join(failures)

    def test_monetary_deductions_match(self, scored):
        failures = [
            f"{r.case_id}: expected {r.expected_deduction_inr}, produced {r.actual_deduction_inr}"
            for r in scored.values()
            if r.deduction_correct is False
        ]
        assert not failures, "Deduction regressions: " + "; ".join(failures)

    def test_no_fabricated_citations_anywhere(self, scored):
        for case_id, result in scored.items():
            assert result.citation_correctness in (None, 1.0), (
                f"{case_id}: citation correctness {result.citation_correctness}"
            )

    def test_required_clauses_are_retrieved(self, scored):
        gaps = {r.case_id: r.evidence_missing for r in scored.values() if r.evidence_missing}
        assert not gaps, f"Retrieval gaps: {gaps}"

    def test_validation_passes_for_all(self, scored):
        failures = [r.case_id for r in scored.values() if r.validation_status != "PASS"]
        assert not failures, f"Validation failures: {failures}"

    def test_aggregate_accuracy_gate(self, scored):
        report = aggregate(list(scored.values()))
        assert report["decision_accuracy"] >= 0.9, (
            f"Decision accuracy regressed to {report['decision_accuracy']}"
        )
        assert report["citation_correctness"] == pytest.approx(1.0)
        assert report["abstention"]["false_negatives"] == 0, (
            "The system decided a case where it should have abstained"
        )

    def test_confidence_is_higher_on_correct_decisions(self, scored):
        correct = [r.confidence for r in scored.values() if r.decision_correct]
        assert correct and sum(correct) / len(correct) > 0.6


class TestHarnessArtifacts:
    def test_results_files_are_written(self, tmp_path):
        from evaluation.run_evaluation import main

        exit_code = main(["--suite", "public", "--case", "PUB-001",
                          "--output-dir", str(tmp_path), "--quiet"])
        assert exit_code == 0
        for name in ["results.json", "results.csv", "summary.md", "decisions.json"]:
            assert (tmp_path / name).exists(), f"{name} not written"

        report = json.loads((tmp_path / "results.json").read_text())
        assert report["overall"]["n_cases"] == 1
        assert "backends" in report
