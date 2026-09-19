"""Agent 4 — Decision.

Responsibility: combine the specialists' structured output into one decision.

Precedence (each step is decisive; later steps only run if earlier ones do not fire):

1. A **blocking violation** -> NOT_ADMISSIBLE.
   A confirmed exclusion or an unexpired waiting period disposes of the claim
   regardless of what else is unresolved: the answer is the same either way.
2. An **unresolved required condition** -> NEEDS_REVIEW.
   RULE 5: a condition precedent the evidence cannot establish means no safe
   decision exists.
3. A **partially excluded portion** -> PARTIALLY_ADMISSIBLE.
   Part of the claim is outside cover while the remainder stands.
4. A **binding monetary cap** -> ADMISSIBLE_WITH_LIMITS.
5. Otherwise -> ADMISSIBLE.

Finally, if confidence falls below ``ABSTAIN_BELOW_CONFIDENCE``, any
affirmative decision is downgraded to NEEDS_REVIEW. Rejections are not
downgraded: a rejection already grounded in a cited blocking clause is not made
safer by converting it to a review.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.models.schemas import (
    AnalysisResponse,
    CaseAnalysis,
    Citation,
    ClaimCase,
    ConfidenceBreakdown,
    CoverageAssessment,
    Decision,
    EvidenceItem,
    Finding,
    FindingStatus,
    Severity,
    ValidationReport,
)
from app.services.confidence import compute_confidence
from app.services.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

_AFFIRMATIVE = {
    Decision.ADMISSIBLE,
    Decision.ADMISSIBLE_WITH_LIMITS,
    Decision.PARTIALLY_ADMISSIBLE,
}


class DecisionAgent:
    name = "decision"

    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client()

    # ------------------------------------------------------------------ #

    def _classify(self, assessment: CoverageAssessment) -> tuple[Decision, str | None]:
        findings = assessment.findings

        blocking = [
            f for f in findings
            if f.severity == Severity.BLOCKING and f.status == FindingStatus.VIOLATED
        ]
        if blocking:
            return Decision.NOT_ADMISSIBLE, None

        abstaining = [
            f for f in findings
            if f.severity == Severity.ABSTAIN and f.status == FindingStatus.UNRESOLVED
        ]
        blocking_gaps = [m for m in assessment.missing_evidence if m.blocking]
        if abstaining or blocking_gaps:
            reasons = [f.statement for f in abstaining] or [
                f"Required evidence not available: {m.item}" for m in blocking_gaps
            ]
            return Decision.NEEDS_REVIEW, "INSUFFICIENT_EVIDENCE: " + " ".join(reasons[:2])

        partial = [
            f for f in findings
            if f.severity == Severity.LIMITING and f.status == FindingStatus.VIOLATED
        ]
        if partial:
            return Decision.PARTIALLY_ADMISSIBLE, None

        if any(l.binding for l in assessment.applied_limits):
            return Decision.ADMISSIBLE_WITH_LIMITS, None

        return Decision.ADMISSIBLE, None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _key_findings(assessment: CoverageAssessment, decision: Decision) -> list[Finding]:
        """Order findings by decision relevance so the reviewer reads the point first."""
        priority = {
            Severity.BLOCKING: 0,
            Severity.ABSTAIN: 1,
            Severity.LIMITING: 2,
            Severity.INFORMATIONAL: 3,
        }

        def sort_key(f: Finding) -> tuple[int, int, str]:
            decisive = 0 if (
                (decision == Decision.NOT_ADMISSIBLE
                 and f.severity == Severity.BLOCKING
                 and f.status == FindingStatus.VIOLATED)
                or (decision == Decision.NEEDS_REVIEW
                    and f.status == FindingStatus.UNRESOLVED)
            ) else 1
            return (decisive, priority.get(f.severity, 9), f.rule_id)

        return sorted(assessment.findings, key=sort_key)

    @staticmethod
    def _collect_citations(findings: list[Finding], assessment: CoverageAssessment) -> list[Citation]:
        """Deduplicate citations while preserving the order findings introduced them."""
        seen: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for source in (findings, [l for l in assessment.applied_limits]):
            for item in source:
                for citation in item.citations:
                    key = (citation.chunk_id, citation.claim[:80])
                    if key not in seen:
                        seen.add(key)
                        citations.append(citation)
        return citations

    # ------------------------------------------------------------------ #

    def _summary(
        self,
        case: ClaimCase,
        decision: Decision,
        assessment: CoverageAssessment,
        key_findings: list[Finding],
        abstain_reason: str | None,
    ) -> str:
        """Deterministic reviewer summary; optionally rephrased by the LLM.

        The template version is always computed first so the system reads
        identically with or without a key.
        """
        parts: list[str] = []
        decisive = next(
            (f for f in key_findings
             if f.status in {FindingStatus.VIOLATED, FindingStatus.UNRESOLVED}
             and f.severity in {Severity.BLOCKING, Severity.ABSTAIN}),
            None,
        )

        if decision == Decision.NOT_ADMISSIBLE:
            parts.append(f"Claim {case.case_id} is not admissible.")
            if decisive:
                parts.append(decisive.statement)
        elif decision == Decision.NEEDS_REVIEW:
            parts.append(
                f"Claim {case.case_id} cannot be decided on the supplied evidence and is "
                "referred for review."
            )
            if decisive:
                parts.append(decisive.statement)
            elif abstain_reason:
                parts.append(abstain_reason.replace("INSUFFICIENT_EVIDENCE: ", ""))
        elif decision == Decision.PARTIALLY_ADMISSIBLE:
            parts.append(f"Claim {case.case_id} is partially admissible.")
            excluded = [
                f.statement for f in assessment.findings
                if f.severity == Severity.LIMITING and f.status == FindingStatus.VIOLATED
            ]
            parts.extend(excluded[:2])
        elif decision == Decision.ADMISSIBLE_WITH_LIMITS:
            binding = [l for l in assessment.applied_limits if l.binding]
            names = ", ".join(l.category.replace("_", " ") for l in binding)
            parts.append(
                f"Claim {case.case_id} is admissible subject to policy limits on {names}."
            )
            if assessment.total_deduction_inr:
                parts.append(
                    f"Total deduction INR {assessment.total_deduction_inr:,.0f} against a claimed "
                    f"INR {assessment.claimed_total_inr:,.0f}."
                )
        else:
            parts.append(
                f"Claim {case.case_id} is admissible; no policy limit reduces the payable amount."
            )

        if assessment.missing_evidence and decision != Decision.NEEDS_REVIEW:
            parts.append(
                f"{len(assessment.missing_evidence)} evidence item(s) noted but not decisive."
            )

        summary = " ".join(p.rstrip(".") + "." for p in parts if p)
        return self._llm_polish(summary, decision, key_findings) or summary

    def _llm_polish(
        self, summary: str, decision: Decision, key_findings: list[Finding]
    ) -> str | None:
        """Ask the LLM to tighten the wording. It may not add or change facts."""
        if not self.llm.enabled:
            return None
        try:
            result = self.llm.complete(
                system=(
                    "You rewrite insurance claim decision summaries for clarity. "
                    "You must not add, remove or alter any fact, amount, clause reference or "
                    "conclusion. Return two sentences at most, plain text, no preamble."
                ),
                user=(
                    f"Decision: {decision.value}\n"
                    f"Supported statements:\n"
                    + "\n".join(f"- {f.statement}" for f in key_findings[:4])
                    + f"\n\nDraft summary: {summary}"
                ),
                max_tokens=180,
            )
            if result.ok and result.content.strip():
                return result.content.strip()
        except Exception as exc:  # pragma: no cover - optional path
            logger.debug("LLM summary polish skipped: %s", exc)
        return None

    # ------------------------------------------------------------------ #

    def run(
        self,
        case: ClaimCase,
        analysis: CaseAnalysis,
        assessment: CoverageAssessment,
        evidence: list[EvidenceItem],
        *,
        validation: ValidationReport | None = None,
    ) -> AnalysisResponse:
        decision, abstain_reason = self._classify(assessment)

        confidence: ConfidenceBreakdown = compute_confidence(
            assessment=assessment,
            evidence=evidence,
            dimensions=analysis.decision_dimensions,
            decision=decision,
            validation_status=validation.status if validation else None,
        )

        # Uncertainty becomes abstention, but only for affirmative decisions.
        if (
            decision in _AFFIRMATIVE
            and confidence.final_score < self.settings.abstain_below_confidence
        ):
            logger.info(
                "Downgrading %s to NEEDS_REVIEW for %s (confidence %.2f < %.2f)",
                decision.value, case.case_id, confidence.final_score,
                self.settings.abstain_below_confidence,
            )
            decision = Decision.NEEDS_REVIEW
            abstain_reason = (
                f"INSUFFICIENT_EVIDENCE: evidential support for an affirmative decision is "
                f"below the required threshold (confidence {confidence.final_score:.2f} < "
                f"{self.settings.abstain_below_confidence:.2f})."
            )
            confidence = compute_confidence(
                assessment=assessment,
                evidence=evidence,
                dimensions=analysis.decision_dimensions,
                decision=decision,
                validation_status=validation.status if validation else None,
            )

        key_findings = self._key_findings(assessment, decision)
        citations = self._collect_citations(key_findings, assessment)

        return AnalysisResponse(
            case_id=case.case_id,
            decision=decision,
            confidence=confidence.final_score,
            summary=self._summary(case, decision, assessment, key_findings, abstain_reason),
            key_findings=key_findings,
            applicable_limits=assessment.applied_limits,
            missing_evidence=assessment.missing_evidence,
            citations=citations,
            validation=validation or ValidationReport(),
            confidence_breakdown=confidence,
            evidence=evidence,
            case_analysis=analysis,
            payable_estimate_inr=assessment.payable_estimate_inr,
            claimed_total_inr=assessment.claimed_total_inr,
            total_deduction_inr=assessment.total_deduction_inr,
            abstained=decision == Decision.NEEDS_REVIEW,
            abstain_reason=abstain_reason,
            engine_version=self.settings.version,
        )
