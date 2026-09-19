"""Typed contracts shared by every agent, the API and the evaluation harness.

These models *are* the inter-agent protocol. Agents never pass free-form text to
one another; they read and write the fields defined here. That is what makes the
workflow auditable and testable rather than a chain of prompts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Decision(str, Enum):
    """The five decision statuses required by the assignment (§6)."""

    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DecisionDimension(str, Enum):
    """Independent axes a claim must be investigated along.

    Each dimension drives its own retrieval query, so a claim is never resolved
    from a single lookup (assignment §5 / RULE 7).
    """

    HOSPITALIZATION_DEFINITION = "hospitalization_definition"
    HOSPITAL_DEFINITION = "hospital_definition"
    SCOPE_OF_COVER = "scope_of_cover"
    CLAIM_DOCUMENTATION = "claim_documentation"
    DAY_CARE = "day_care"
    DOMICILIARY = "domiciliary"
    WAITING_PERIOD_INITIAL = "waiting_period_initial"
    WAITING_PERIOD_FIRST_YEAR = "waiting_period_first_year"
    PRE_EXISTING_DISEASE = "pre_existing_disease"
    PORTABILITY_CONTINUITY = "portability_continuity"
    EXCLUSIONS = "exclusions"
    EXPERIMENTAL_TREATMENT = "experimental_treatment"
    ROOM_RENT_LIMIT = "room_rent_limit"
    CATEGORY_SUBLIMITS = "category_sublimits"
    AMBULANCE_LIMIT = "ambulance_limit"
    PRE_POST_HOSPITALIZATION = "pre_post_hospitalization"
    MEDICAL_NECESSITY = "medical_necessity"
    SUM_INSURED_AGGREGATE = "sum_insured_aggregate"


class FindingStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    LIMIT_APPLIES = "LIMIT_APPLIES"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNRESOLVED = "UNRESOLVED"


class Severity(str, Enum):
    """How a finding influences the final decision."""

    BLOCKING = "BLOCKING"        # forces NOT_ADMISSIBLE
    ABSTAIN = "ABSTAIN"          # forces NEEDS_REVIEW
    LIMITING = "LIMITING"        # reduces payable amount
    INFORMATIONAL = "INFORMATIONAL"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


# --------------------------------------------------------------------------- #
# Input case
# --------------------------------------------------------------------------- #


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int | None = Field(default=None, ge=0, le=120)


class Hospital(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str | None = None
    network_provider: bool | None = None


class Treatment(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str | None = Field(
        default=None,
        description="inpatient | day_care | domiciliary | outpatient",
    )
    admission_hours: float | None = Field(default=None, ge=0)
    diagnosis: str | None = None
    procedure: str | None = None
    pre_existing: bool | None = None
    experimental: bool | None = None
    hospital_room_unavailable: bool | None = None
    patient_cannot_be_moved: bool | None = None
    treatment_days: float | None = None


class Expenses(BaseModel):
    """All monetary fields are Indian Rupees, matching the supplied dataset."""

    model_config = ConfigDict(extra="allow")
    room: float = 0.0
    icu: float = 0.0
    doctor_fees: float = 0.0
    medicines_diagnostics: float = 0.0
    pre_hospitalization: float = 0.0
    post_hospitalization: float = 0.0
    ambulance: float = 0.0

    @field_validator("*", mode="before")
    @classmethod
    def _null_to_zero(cls, v: Any) -> Any:
        return 0.0 if v is None else v

    def total(self) -> float:
        return float(
            self.room
            + self.icu
            + self.doctor_fees
            + self.medicines_diagnostics
            + self.pre_hospitalization
            + self.post_hospitalization
            + self.ambulance
        )


class ExpenseTiming(BaseModel):
    model_config = ConfigDict(extra="allow")
    pre_hospitalization_days_before_admission: int | None = None
    post_hospitalization_days_after_discharge: int | None = None
    same_condition_confirmed: bool | None = None


class PriorPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")
    insurer_type: str | None = None
    continuous_years: float | None = None
    database_and_claim_history_received: bool | None = None
    previous_sum_insured_inr: float | None = None


class EvidenceContext(BaseModel):
    """Externally-established facts the policy requires but the claim may lack.

    ``None`` means *unknown* and is materially different from ``False``: unknown
    drives abstention, False drives a negative finding.
    """

    model_config = ConfigDict(extra="allow")
    hospital_registered: bool | None = None
    hospital_minimum_criteria_documented: bool | None = None
    medical_necessity_confirmed: bool | None = None
    day_care_procedure_listed: bool | None = None
    technological_advancement_certified: bool | None = None


class ClaimCase(BaseModel):
    """Input schema, permissive by design.

    ``extra="allow"`` implements RULE 9: unknown, non-critical attributes are
    tolerated and simply ignored by the decision logic rather than rejected.
    """

    model_config = ConfigDict(extra="allow")

    case_id: str = Field(min_length=1, max_length=120)
    policy_id: str | None = None
    policy_start_date: str | None = None
    claim_date: str | None = None
    sum_insured_inr: float | None = Field(default=None, gt=0)
    continuous_coverage_months: float | None = Field(default=None, ge=0)
    prior_insurer_continuous_years: float | None = Field(default=None, ge=0)

    patient: Patient = Field(default_factory=Patient)
    hospital: Hospital = Field(default_factory=Hospital)
    treatment: Treatment = Field(default_factory=Treatment)
    expenses_inr: Expenses = Field(default_factory=Expenses)

    documents: list[str] = Field(default_factory=list)
    task: str | None = None

    prior_policy: PriorPolicy | None = None
    evidence_context: EvidenceContext = Field(default_factory=EvidenceContext)
    expense_timing: ExpenseTiming | None = None

    @field_validator("case_id")
    @classmethod
    def _clean_case_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("case_id must not be blank")
        return v


# --------------------------------------------------------------------------- #
# Retrieval artefacts
# --------------------------------------------------------------------------- #


class PolicyChunk(BaseModel):
    """An indexed unit of the policy with full provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    page: int
    page_end: int | None = None
    section: str
    heading: str
    clause_ref: str | None = None
    source: str
    char_count: int = 0
    token_estimate: int = 0
    ordinal: int = 0


class EvidenceItem(BaseModel):
    """A retrieved chunk plus the scores that justify its presence."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    page: int
    section: str
    heading: str
    clause_ref: str | None = None
    source: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float = 0.0
    retrieval_method: Literal["dense", "bm25", "fused"] = "fused"
    matched_dimensions: list[DecisionDimension] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)

    def snippet(self, limit: int = 320) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: DecisionDimension
    query: str
    rationale: str
    expansions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent outputs
# --------------------------------------------------------------------------- #


class InvestigationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: DecisionDimension
    question: str
    why_it_matters: str
    required_inputs: list[str] = Field(default_factory=list)
    blocking_if_unresolved: bool = False


class CaseAnalysis(BaseModel):
    """Output of Agent 1."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    facts: dict[str, Any] = Field(default_factory=dict)
    derived_facts: dict[str, Any] = Field(default_factory=dict)
    decision_dimensions: list[DecisionDimension] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    irrelevant_attributes: list[str] = Field(default_factory=list)
    investigation_plan: list[InvestigationItem] = Field(default_factory=list)
    input_warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """The evidence contract from assignment §5."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    source: str
    page: int
    section: str
    heading: str | None = None
    chunk_id: str
    quote: str | None = None
    rule_id: str | None = None


class Finding(BaseModel):
    """A single policy-grounded conclusion about one decision dimension."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    dimension: DecisionDimension
    status: FindingStatus
    severity: Severity
    statement: str
    detail: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_supported: bool = True


class AppliedLimit(BaseModel):
    """A monetary cap the policy imposes, with the arithmetic made explicit."""

    model_config = ConfigDict(extra="forbid")

    limit_id: str
    category: str
    description: str
    basis: str
    limit_amount_inr: float | None = None
    claimed_amount_inr: float | None = None
    allowed_amount_inr: float | None = None
    deduction_inr: float = 0.0
    binding: bool = False
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class MissingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str
    dimension: DecisionDimension
    why_required: str
    blocking: bool = False
    suggested_document: str | None = None


class CoverageAssessment(BaseModel):
    """Output of Agent 3."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    applied_limits: list[AppliedLimit] = Field(default_factory=list)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list)
    unresolved_dimensions: list[DecisionDimension] = Field(default_factory=list)
    payable_estimate_inr: float | None = None
    claimed_total_inr: float | None = None
    total_deduction_inr: float = 0.0


class ConfidenceBreakdown(BaseModel):
    """Every component of the confidence score, so it can be audited (§8)."""

    model_config = ConfigDict(extra="forbid")

    retrieval_quality: float = 0.0
    evidence_coverage: float = 0.0
    citation_support: float = 0.0
    decision_consistency: float = 0.0
    validation_factor: float = 1.0
    missing_evidence_penalty: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    raw_score: float = 0.0
    final_score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Output of Agent 5."""

    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus = ValidationStatus.PASS
    unsupported_claims: list[str] = Field(default_factory=list)
    revision_required: bool = False
    revisions_applied: int = 0
    checks_run: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    citation_integrity: bool = True
    notes: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """Auditable, chain-of-thought-free execution record (§9 / RULE 10)."""

    model_config = ConfigDict(extra="allow")

    agent: str
    action: str
    elapsed_ms: int = 0
    status: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """The structured decision contract returned by POST /analyze (§5, §7)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    key_findings: list[Finding] = Field(default_factory=list)
    applicable_limits: list[AppliedLimit] = Field(default_factory=list)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = Field(default_factory=list)
    case_analysis: CaseAnalysis | None = None
    payable_estimate_inr: float | None = None
    claimed_total_inr: float | None = None
    total_deduction_inr: float = 0.0
    abstained: bool = False
    abstain_reason: str | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    engine_version: str = "1.0.0"
    backends: dict[str, str] = Field(default_factory=dict)
    total_elapsed_ms: int = 0
