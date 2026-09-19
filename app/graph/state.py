"""The shared workflow state.

This is the structured channel through which agents communicate. Each agent
reads specific keys and writes specific keys; none of them passes free-form
prose to the next. The ownership table below is the contract:

    key                  written by            read by
    -------------------  --------------------  ------------------------------
    case                 (input)               all
    analysis             case_analysis         policy_evidence, coverage, decision
    queries              case_analysis         policy_evidence
    evidence             policy_evidence       coverage_exclusion, validation
    retrieval_stats      policy_evidence       decision (confidence), trace
    assessment           coverage_exclusion    decision, validation
    decision_draft       decision              validation
    validation           validation            decision (on revision)
    trace                all                   API response
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from app.models.schemas import (
    AnalysisResponse,
    CaseAnalysis,
    ClaimCase,
    CoverageAssessment,
    EvidenceItem,
    RetrievalQuery,
    TraceEvent,
    ValidationReport,
)


def _append(left: list, right: list) -> list:
    """Reducer so parallel/looping nodes accumulate trace events."""
    return [*(left or []), *(right or [])]


class ClaimWorkflowState(TypedDict, total=False):
    """LangGraph state object passed between agent nodes."""

    case: ClaimCase
    analysis: CaseAnalysis
    queries: list[RetrievalQuery]
    evidence: list[EvidenceItem]
    retrieval_stats: dict[str, Any]
    assessment: CoverageAssessment
    decision_draft: AnalysisResponse
    validation: ValidationReport
    revision_count: int
    trace: Annotated[list[TraceEvent], _append]
    errors: Annotated[list[str], _append]
