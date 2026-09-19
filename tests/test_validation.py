"""Validation-agent tests, including deliberate tampering.

The point of the Validation Agent is to catch a decision whose statements are
not supported by evidence. These tests manufacture exactly that situation --
fabricated citations, drifted quotes, broken arithmetic, an affirmative
decision taken over an unresolved condition -- and assert that it is caught.
"""

from __future__ import annotations

import copy

import pytest

from app.agents.validation import ValidationAgent, force_review
from app.models.schemas import (
    AppliedLimit,
    Citation,
    Decision,
    DecisionDimension as D,
    Finding,
    FindingStatus,
    Severity,
    ValidationStatus,
)


@pytest.fixture
def clean(analyses):
    return copy.deepcopy(analyses["PUB-001"])


class TestValidationPasses:
    def test_untampered_decision_passes(self, clean):
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.PASS
        assert report.unsupported_claims == []
        assert report.citation_integrity is True

    def test_all_checks_run(self, clean):
        report = ValidationAgent().run(clean, clean.evidence)
        for check in [
            "citation_integrity", "material_support", "quote_provenance",
            "limit_arithmetic", "decision_alignment", "abstention_discipline",
        ]:
            assert check in report.checks_run

    def test_every_case_passes_validation(self, analyses):
        for case_id, response in analyses.items():
            assert response.validation.status == ValidationStatus.PASS, (
                f"{case_id} failed validation: {response.validation.unsupported_claims}"
            )


class TestValidationCatchesFabrication:
    def test_citation_to_nonexistent_chunk_is_caught(self, clean):
        """RULE 3: a citation naming a chunk that was never retrieved."""
        clean.citations.append(
            Citation(
                claim="The policy covers unlimited dental implants.",
                source="policy.pdf",
                page=99,
                section="Invented Section",
                heading="Fabricated",
                chunk_id="p99-entirely-made-up-999",
                quote="This text does not exist.",
            )
        )
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "citation_integrity" in report.checks_failed
        assert not report.citation_integrity
        assert any("not in the retrieved evidence" in c for c in report.unsupported_claims)

    def test_drifted_page_number_is_caught(self, clean):
        """A citation pointing at a real chunk but the wrong page."""
        clean.citations[0].page = 42
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "citation_integrity" in report.checks_failed

    def test_altered_section_is_caught(self, clean):
        clean.citations[0].section = "Some Other Section"
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL

    def test_invented_quote_is_caught(self, clean):
        """RULE 6: a quote assembled rather than extracted from the chunk."""
        target = next(f for f in clean.key_findings if f.citations)
        target.citations[0].quote = (
            "The insurer shall pay all expenses without limit or deduction whatsoever, "
            "notwithstanding any other provision of this policy."
        )
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "quote_provenance" in report.checks_failed

    def test_material_finding_without_citation_is_caught(self, clean):
        clean.key_findings.append(
            Finding(
                rule_id="fabricated_rule",
                dimension=D.EXCLUSIONS,
                status=FindingStatus.VIOLATED,
                severity=Severity.BLOCKING,
                statement="This claim is excluded under a clause that was never retrieved.",
                evidence_chunk_ids=[],
                citations=[],
                evidence_supported=False,
            )
        )
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "material_support" in report.checks_failed

    def test_broken_limit_arithmetic_is_caught(self, clean):
        clean.applicable_limits.append(
            AppliedLimit(
                limit_id="bogus_limit",
                category="room",
                description="Tampered limit",
                basis="fabricated",
                limit_amount_inr=20000,
                claimed_amount_inr=30000,
                allowed_amount_inr=30000,   # should be 20000
                deduction_inr=0,            # should be 10000
                binding=False,
            )
        )
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "limit_arithmetic" in report.checks_failed


class TestValidationCatchesBadDecisions:
    def test_rejection_without_a_blocking_finding_is_caught(self, clean):
        clean.decision = Decision.NOT_ADMISSIBLE
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "decision_alignment" in report.checks_failed

    def test_affirmative_decision_over_unresolved_condition_is_caught(self, clean):
        """RULE 4/5: abstention discipline."""
        clean.key_findings.append(
            Finding(
                rule_id="hospital_definition",
                dimension=D.HOSPITAL_DEFINITION,
                status=FindingStatus.UNRESOLVED,
                severity=Severity.ABSTAIN,
                statement="Facility eligibility could not be established.",
                evidence_chunk_ids=[],
                citations=[],
                evidence_supported=False,
            )
        )
        clean.decision = Decision.ADMISSIBLE
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "abstention_discipline" in report.checks_failed

    def test_abstention_without_a_reason_is_caught(self, clean):
        clean.decision = Decision.NEEDS_REVIEW
        clean.missing_evidence = []
        clean.abstain_reason = None
        clean.key_findings = [
            f for f in clean.key_findings if f.status != FindingStatus.UNRESOLVED
        ]
        report = ValidationAgent().run(clean, clean.evidence)
        assert report.status == ValidationStatus.FAIL
        assert "decision_alignment" in report.checks_failed


class TestForcedReview:
    def test_unverifiable_decision_becomes_needs_review(self, clean):
        clean.decision = Decision.ADMISSIBLE
        report = ValidationAgent().run(clean, clean.evidence)
        forced = force_review(clean, report)
        assert forced.decision == Decision.NEEDS_REVIEW
        assert forced.abstained is True
        assert "INSUFFICIENT_EVIDENCE" in forced.abstain_reason

    def test_revision_loop_terminates(self, analyses):
        """Regression test for F-07: the validation loop must always terminate."""
        for case_id, response in analyses.items():
            decisions = [e for e in response.trace if e.agent == "decision"]
            assert len(decisions) <= 3, f"{case_id}: decision ran {len(decisions)} times"
            assert response.validation.revisions_applied <= 2


class TestAbstentionBehaviour:
    def test_abstention_cases_abstain(self, analyses, expected_public, expected_custom):
        expected = {**expected_public, **expected_custom}
        abstaining = [
            cid for cid, spec in expected.items() if spec.get("abstention_case")
        ]
        assert len(abstaining) >= 2, "The suite must contain at least two abstention cases"
        for case_id in abstaining:
            response = analyses[case_id]
            assert response.decision == Decision.NEEDS_REVIEW, f"{case_id} did not abstain"
            assert response.abstained is True

    def test_abstention_names_what_is_missing(self, analyses):
        for case_id in ["PUB-006", "PUB-011", "CUS-005", "CUS-009"]:
            response = analyses[case_id]
            assert response.missing_evidence, f"{case_id} abstained without naming a gap"
            assert any(m.blocking for m in response.missing_evidence)

    def test_no_case_asserts_support_it_does_not_have(self, analyses):
        for case_id, response in analyses.items():
            for finding in response.key_findings:
                if finding.evidence_supported:
                    assert finding.citations, (
                        f"{case_id}: {finding.rule_id} claims support but cites nothing"
                    )
