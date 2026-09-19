"""Confidence scoring.

Confidence here is a measurement of *evidential support*, not a number the model
was asked to produce. It answers: "how well did the evidence actually support
the decision we reached?"

    raw = 0.25 * retrieval_quality
        + 0.25 * evidence_coverage
        + 0.25 * citation_support
        + 0.25 * decision_consistency

    final = clamp(raw * validation_factor - missing_evidence_penalty, 0, 1)

Components
----------
retrieval_quality     Mean rerank score of the evidence actually cited by
                      findings (falling back to the top-k mean). Measures how
                      well the clauses we relied on matched their queries.
evidence_coverage     Share of investigated dimensions that produced at least
                      one retrieved chunk. Measures investigation completeness.
citation_support      Share of material findings carrying >=1 citation.
                      Structurally near 1.0 because unsupported findings cannot
                      be produced — it is a regression alarm, not a lever.
decision_consistency  Whether the findings agree. Unanimity scores 1.0;
                      contradictory signals (a blocking exclusion alongside an
                      unresolved dimension) reduce it.
validation_factor     1.0 on PASS, 0.55 on FAIL. A decision whose statements
                      failed verification cannot be high-confidence.
missing_evidence      0.08 per blocking gap, 0.03 per non-blocking gap, capped
                      at 0.35.

Low confidence does not merely annotate the answer: below
``ABSTAIN_BELOW_CONFIDENCE`` the Decision Agent downgrades an affirmative
decision to NEEDS_REVIEW, so uncertainty becomes abstention.
"""

from __future__ import annotations

from app.models.schemas import (
    ConfidenceBreakdown,
    CoverageAssessment,
    Decision,
    EvidenceItem,
    Finding,
    FindingStatus,
    Severity,
    ValidationStatus,
)

WEIGHTS = {
    "retrieval_quality": 0.25,
    "evidence_coverage": 0.25,
    "citation_support": 0.25,
    "decision_consistency": 0.25,
}

_MATERIAL_SEVERITIES = {Severity.BLOCKING, Severity.ABSTAIN, Severity.LIMITING}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def retrieval_quality(evidence: list[EvidenceItem], findings: list[Finding]) -> float:
    if not evidence:
        return 0.0
    cited = {cid for f in findings for cid in f.evidence_chunk_ids}
    scores = [e.rerank_score for e in evidence if e.chunk_id in cited]
    if not scores:
        scores = [e.rerank_score for e in evidence[:5]]
    return max(0.0, min(1.0, _mean(scores)))


def evidence_coverage(dimensions: list, evidence: list[EvidenceItem]) -> float:
    if not dimensions:
        return 0.0
    covered = {d for item in evidence for d in item.matched_dimensions}
    return len(covered & set(dimensions)) / len(dimensions)


def citation_support(findings: list[Finding]) -> float:
    material = [f for f in findings if f.severity in _MATERIAL_SEVERITIES]
    if not material:
        material = findings
    if not material:
        return 0.0
    supported = sum(1 for f in material if f.citations)
    return supported / len(material)


def decision_consistency(assessment: CoverageAssessment, decision: Decision) -> float:
    """Do the findings point the same way as the decision?"""
    findings = assessment.findings
    if not findings:
        return 0.0

    blocking = [
        f for f in findings
        if f.severity == Severity.BLOCKING and f.status == FindingStatus.VIOLATED
    ]
    abstaining = [
        f for f in findings
        if f.severity == Severity.ABSTAIN and f.status == FindingStatus.UNRESOLVED
    ]
    limiting = [l for l in assessment.applied_limits if l.binding]

    score = 1.0
    if decision == Decision.NOT_ADMISSIBLE:
        if not blocking:
            score -= 0.6
        # A rejection made while a dimension is unresolved is less clean.
        if abstaining:
            score -= 0.12
    elif decision == Decision.NEEDS_REVIEW:
        if not (abstaining or assessment.missing_evidence):
            score -= 0.5
    elif decision == Decision.ADMISSIBLE_WITH_LIMITS:
        if not limiting:
            score -= 0.4
        if blocking:
            score -= 0.6
    elif decision == Decision.ADMISSIBLE:
        if limiting:
            score -= 0.35
        if blocking or abstaining:
            score -= 0.6
    elif decision == Decision.PARTIALLY_ADMISSIBLE:
        partial = [
            f for f in findings
            if f.severity == Severity.LIMITING and f.status == FindingStatus.VIOLATED
        ]
        if not partial:
            score -= 0.4

    return max(0.0, min(1.0, score))


def missing_evidence_penalty(assessment: CoverageAssessment) -> float:
    penalty = 0.0
    for item in assessment.missing_evidence:
        penalty += 0.08 if item.blocking else 0.03
    return min(0.35, penalty)


def compute_confidence(
    *,
    assessment: CoverageAssessment,
    evidence: list[EvidenceItem],
    dimensions: list,
    decision: Decision,
    validation_status: ValidationStatus | None = None,
) -> ConfidenceBreakdown:
    rq = retrieval_quality(evidence, assessment.findings)
    ec = evidence_coverage(dimensions, evidence)
    cs = citation_support(assessment.findings)
    dc = decision_consistency(assessment, decision)
    penalty = missing_evidence_penalty(assessment)
    vf = 0.55 if validation_status == ValidationStatus.FAIL else 1.0

    raw = (
        WEIGHTS["retrieval_quality"] * rq
        + WEIGHTS["evidence_coverage"] * ec
        + WEIGHTS["citation_support"] * cs
        + WEIGHTS["decision_consistency"] * dc
    )
    final = max(0.0, min(1.0, raw * vf - penalty))

    notes: list[str] = []
    if penalty > 0:
        notes.append(f"Reduced by {penalty:.2f} for {len(assessment.missing_evidence)} evidence gap(s).")
    if vf < 1.0:
        notes.append("Reduced because validation did not pass.")
    if ec < 0.75:
        notes.append(
            f"Only {ec:.0%} of investigated dimensions returned policy evidence."
        )
    if rq < 0.5 and evidence:
        notes.append("Evidence matched its queries weakly; citations should be reviewed.")

    return ConfidenceBreakdown(
        retrieval_quality=round(rq, 4),
        evidence_coverage=round(ec, 4),
        citation_support=round(cs, 4),
        decision_consistency=round(dc, 4),
        validation_factor=vf,
        missing_evidence_penalty=round(penalty, 4),
        weights=dict(WEIGHTS),
        raw_score=round(raw, 4),
        final_score=round(final, 4),
        notes=notes,
    )
