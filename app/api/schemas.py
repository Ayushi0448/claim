"""API request/response envelopes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import AnalysisResponse, ClaimCase


class AnalyzeRequest(BaseModel):
    """POST /analyze body.

    A claim may be sent either bare (the supplied dataset's own shape) or
    wrapped in a ``case`` object. Both are accepted so a reviewer can paste a
    case straight from ``public_test_cases.json`` without reshaping it.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "case_id": "PUB-001",
                "policy_id": "USGIC-CSC-2017-2018",
                "policy_start_date": "2025-01-01",
                "claim_date": "2026-03-14",
                "sum_insured_inr": 500000,
                "continuous_coverage_months": 14,
                "prior_insurer_continuous_years": 0,
                "patient": {"age": 34},
                "hospital": {"name": "Sunrise Multispeciality", "network_provider": True},
                "treatment": {
                    "type": "inpatient",
                    "admission_hours": 96,
                    "diagnosis": "Acute appendicitis",
                    "procedure": "Appendectomy",
                    "pre_existing": False,
                    "experimental": False,
                },
                "expenses_inr": {
                    "room": 30000,
                    "doctor_fees": 30000,
                    "medicines_diagnostics": 90000,
                    "pre_hospitalization": 5000,
                    "post_hospitalization": 7000,
                    "ambulance": 1200,
                },
                "documents": [
                    "claim_form", "discharge_summary", "itemized_bill", "doctor_prescription"
                ],
                "task": "Determine whether the hospitalization is admissible.",
            }
        },
    )

    case: ClaimCase | None = Field(
        default=None, description="Optional wrapper; omit to post the claim fields directly."
    )
    include_evidence: bool = Field(
        default=True, description="Return the full retrieved evidence set."
    )
    include_trace: bool = Field(default=True, description="Return the agent execution trace.")

    def resolve_case(self) -> ClaimCase:
        """Accept both the wrapped and the bare shape."""
        if self.case is not None:
            return self.case
        payload: dict[str, Any] = {
            k: v
            for k, v in (self.model_extra or {}).items()
            if k not in {"include_evidence", "include_trace"}
        }
        return ClaimCase(**payload)


class BatchAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[ClaimCase] = Field(min_length=1, max_length=50)
    include_evidence: bool = False
    include_trace: bool = True


class BatchAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[AnalysisResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    version: str
    index_ready: bool
    policy_indexed: bool
    chunks_indexed: int
    policy_source: str
    policy_id: str
    backends: dict[str, str]
    uptime_seconds: float
    detail: str | None = None


class PolicyChunkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    page: int
    section: str
    heading: str
    char_count: int


class IndexInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_chunks: int
    sections: dict[str, int]
    pages: list[int]
    backends: dict[str, str]
    sample: list[PolicyChunkSummary]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str
    detail: str | None = None
    field_errors: list[dict[str, Any]] = Field(default_factory=list)
    hint: str | None = None
