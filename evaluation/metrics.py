"""Evaluation metrics.

Every metric is computed against the hand-derived ground truth in
``expected_results/``, never against the system's own output. Where a metric
cannot be computed honestly (no label, no required evidence listed) the case is
excluded from that metric's denominator and the exclusion is reported, rather
than being silently scored as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from app.models.schemas import AnalysisResponse
from app.utils.text import contains_phrase

ABSTAIN_DECISION = "NEEDS_REVIEW"


@dataclass
class CaseResult:
    """Per-case evaluation record."""

    case_id: str
    suite: str
    expected_decision: str | None
    actual_decision: str
    decision_correct: bool | None
    confidence: float
    validation_status: str
    expected_deduction_inr: float | None
    actual_deduction_inr: float
    deduction_correct: bool | None
    n_evidence: int
    n_citations: int
    evidence_recall: float | None
    evidence_found: list[str] = field(default_factory=list)
    evidence_missing: list[str] = field(default_factory=list)
    citation_hit_rate: float | None = None
    citation_correctness: float | None = None
    unsupported_claims: int = 0
    latency_ms: int = 0
    abstained: bool = False
    expected_abstention: bool = False
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suite": self.suite,
            "expected_decision": self.expected_decision or "",
            "actual_decision": self.actual_decision,
            "decision_correct": "" if self.decision_correct is None else self.decision_correct,
            "confidence": round(self.confidence, 4),
            "validation": self.validation_status,
            "expected_deduction_inr": (
                "" if self.expected_deduction_inr is None else self.expected_deduction_inr
            ),
            "actual_deduction_inr": round(self.actual_deduction_inr, 2),
            "deduction_correct": "" if self.deduction_correct is None else self.deduction_correct,
            "n_evidence": self.n_evidence,
            "n_citations": self.n_citations,
            "evidence_recall": "" if self.evidence_recall is None else round(self.evidence_recall, 4),
            "citation_hit_rate": (
                "" if self.citation_hit_rate is None else round(self.citation_hit_rate, 4)
            ),
            "citation_correctness": (
                "" if self.citation_correctness is None else round(self.citation_correctness, 4)
            ),
            "unsupported_claims": self.unsupported_claims,
            "latency_ms": self.latency_ms,
            "abstained": self.abstained,
            "expected_abstention": self.expected_abstention,
            "evidence_missing": "; ".join(self.evidence_missing),
        }


# --------------------------------------------------------------------------- #
# Per-case metrics
# --------------------------------------------------------------------------- #


def evidence_recall_at_k(
    response: AnalysisResponse, required: list[dict]
) -> tuple[float | None, list[str], list[str]]:
    """Fraction of required policy clauses present in the retrieved evidence.

    Matching is by verbatim fragment rather than chunk id, so the metric stays
    valid when the chunker changes. The page is also checked when supplied,
    which catches a fragment matched in the wrong part of the document.
    """
    if not required:
        return None, [], []

    found: list[str] = []
    missing: list[str] = []
    for requirement in required:
        phrase = requirement.get("must_contain", "")
        expected_page = requirement.get("page")
        hit = any(
            contains_phrase(item.text, phrase)
            and (expected_page is None or item.page == expected_page)
            for item in response.evidence
        )
        (found if hit else missing).append(phrase)

    return len(found) / len(required), found, missing


def citation_hit_rate(response: AnalysisResponse, required: list[dict]) -> float | None:
    """Fraction of required clauses that the decision actually *cited*.

    Stricter than recall: retrieving the governing clause but never relying on
    it is a real failure, and this is the metric that exposes it.
    """
    if not required:
        return None
    cited_ids = {c.chunk_id for c in response.citations}
    cited_text = " ".join(
        item.text for item in response.evidence if item.chunk_id in cited_ids
    )
    if not cited_text:
        return 0.0
    hits = sum(1 for r in required if contains_phrase(cited_text, r.get("must_contain", "")))
    return hits / len(required)


def citation_correctness(response: AnalysisResponse) -> float | None:
    """Fraction of citations that genuinely support their statement.

    A citation is correct when (a) its chunk is in the retrieved evidence,
    (b) its page and section match that chunk, and (c) its quoted text appears
    verbatim in that chunk. This is what detects a fabricated or drifted
    citation, which is the failure mode the assignment cares most about.
    """
    if not response.citations:
        return None

    by_id = {item.chunk_id: item for item in response.evidence}
    correct = 0
    for citation in response.citations:
        item = by_id.get(citation.chunk_id)
        if item is None:
            continue
        if citation.page != item.page or citation.section != item.section:
            continue
        if citation.quote:
            probe = " ".join(citation.quote.split()).strip("… ")
            if len(probe) > 40:
                probe = probe[10:90]
            if probe and not contains_phrase(item.text, probe):
                continue
        correct += 1
    return correct / len(response.citations)


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #


def _safe_mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(mean(clean), 4) if clean else None


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    labelled = [r for r in results if r.decision_correct is not None]
    correct = [r for r in labelled if r.decision_correct]

    # Abstention is scored as a binary classification: did the system abstain
    # exactly when it should have?
    tp = sum(1 for r in results if r.expected_abstention and r.abstained)
    fn = sum(1 for r in results if r.expected_abstention and not r.abstained)
    fp = sum(1 for r in results if not r.expected_abstention and r.abstained)
    tn = sum(1 for r in results if not r.expected_abstention and not r.abstained)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )

    deduction_checked = [r for r in results if r.deduction_correct is not None]
    latencies = [r.latency_ms for r in results]

    by_decision: dict[str, dict[str, int]] = {}
    for r in labelled:
        entry = by_decision.setdefault(
            r.expected_decision or "?", {"expected": 0, "correct": 0}
        )
        entry["expected"] += 1
        if r.decision_correct:
            entry["correct"] += 1

    confusion: dict[str, dict[str, int]] = {}
    for r in labelled:
        row = confusion.setdefault(r.expected_decision or "?", {})
        row[r.actual_decision] = row.get(r.actual_decision, 0) + 1

    return {
        "n_cases": len(results),
        "n_labelled": len(labelled),
        "decision_accuracy": round(len(correct) / len(labelled), 4) if labelled else None,
        "decision_accuracy_by_expected": by_decision,
        "confusion_matrix": confusion,
        "deduction_accuracy": (
            round(
                sum(1 for r in deduction_checked if r.deduction_correct)
                / len(deduction_checked),
                4,
            )
            if deduction_checked
            else None
        ),
        "deduction_cases_checked": len(deduction_checked),
        "evidence_recall_at_k": _safe_mean([r.evidence_recall for r in results]),
        "citation_hit_rate": _safe_mean([r.citation_hit_rate for r in results]),
        "citation_correctness": _safe_mean([r.citation_correctness for r in results]),
        "validation_pass_rate": round(
            sum(1 for r in results if r.validation_status == "PASS") / len(results), 4
        )
        if results
        else None,
        "unsupported_claims_total": sum(r.unsupported_claims for r in results),
        "abstention": {
            "expected_abstentions": tp + fn,
            "actual_abstentions": tp + fp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        },
        "confidence": {
            "mean": _safe_mean([r.confidence for r in results]),
            "mean_when_correct": _safe_mean([r.confidence for r in correct]),
            "mean_when_incorrect": _safe_mean(
                [r.confidence for r in labelled if not r.decision_correct]
            ),
        },
        "latency_ms": {
            "mean": round(mean(latencies), 1) if latencies else None,
            "median": round(median(latencies), 1) if latencies else None,
            "max": max(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
        },
    }
