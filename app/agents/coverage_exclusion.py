"""Agent 3 — Coverage & Exclusion.

Responsibility: interpret the retrieved evidence against the claim.

It runs the evidence-bound rule engine (coverage, definitions, waiting periods,
exclusions, conditions) and the limit calculators (room rent, category caps,
ambulance, domiciliary aggregate, sum insured), producing structured findings
and applied limits — each carrying the chunk ids and citations that support it.

It never reaches a verdict. It reports what the policy says about this claim and
hands that to the Decision Agent, which is what keeps "what the policy says"
separable from "what we therefore decide".
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    AppliedLimit,
    CaseAnalysis,
    ClaimCase,
    CoverageAssessment,
    DecisionDimension,
    EvidenceItem,
    Finding,
    FindingStatus,
    Severity,
)
from app.policy import limits as L
from app.policy.rules import RuleContext, evaluate_rules

logger = logging.getLogger(__name__)


class CoverageExclusionAgent:
    name = "coverage_exclusion"

    def run(
        self,
        case: ClaimCase,
        analysis: CaseAnalysis,
        evidence: list[EvidenceItem],
    ) -> CoverageAssessment:
        facts = {**analysis.facts, **analysis.derived_facts}
        ctx = RuleContext(case=case, facts=facts, evidence=evidence)

        findings = evaluate_rules(ctx)
        applied = self._compute_limits(case, facts, evidence, findings)

        claimed_total = case.expenses_inr.total()
        total_deduction = round(sum(l.deduction_inr for l in applied), 2)
        payable = self._payable_estimate(case, applied, findings, claimed_total)

        unresolved = sorted(
            {
                f.dimension
                for f in findings
                if f.status == FindingStatus.UNRESOLVED and f.severity == Severity.ABSTAIN
            },
            key=lambda d: d.value,
        )

        return CoverageAssessment(
            findings=findings,
            applied_limits=applied,
            missing_evidence=ctx.missing_evidence,
            unresolved_dimensions=unresolved,
            payable_estimate_inr=payable,
            claimed_total_inr=round(claimed_total, 2),
            total_deduction_inr=total_deduction,
        )

    # ------------------------------------------------------------------ #

    def _compute_limits(
        self,
        case: ClaimCase,
        facts: dict,
        evidence: list[EvidenceItem],
        findings: list[Finding],
    ) -> list[AppliedLimit]:
        """Apply every cap whose authorising clause was retrieved."""
        sum_insured = case.sum_insured_inr
        if not sum_insured:
            logger.info("No sum insured supplied for %s; monetary caps not computed", case.case_id)
            return []

        exp = case.expenses_inr
        days = int(facts.get("hospital_days") or 0)
        applied: list[AppliedLimit] = []

        def add(spec: L.LimitSpec, claimed: float, **kwargs) -> None:
            limit = L.apply_limit(
                spec,
                claimed=claimed,
                sum_insured=sum_insured,
                days=days,
                evidence=evidence,
                **kwargs,
            )
            if limit is not None:
                applied.append(limit)

        if facts.get("is_domiciliary"):
            # Domiciliary claims are governed by their own aggregate cap rather
            # than the per-category in-patient caps.
            domiciliary_total = (
                exp.room + exp.icu + exp.doctor_fees + exp.medicines_diagnostics
            )
            add(
                L.DOMICILIARY_AGGREGATE,
                domiciliary_total,
                basis_override=(
                    f"20% of Basic Sum Insured (INR {sum_insured:,.0f}) = "
                    f"INR {sum_insured * 0.20:,.0f} aggregate for domiciliary treatment"
                ),
            )
        else:
            if exp.room > 0:
                add(L.ROOM_RENT, exp.room)
            if exp.icu > 0:
                add(L.ICU_RENT, exp.icu)
            if exp.doctor_fees > 0:
                add(L.DOCTOR_FEES, exp.doctor_fees)
            if exp.medicines_diagnostics > 0:
                add(L.MEDICINES_DIAGNOSTICS, exp.medicines_diagnostics)

        if exp.ambulance > 0:
            add(L.AMBULANCE, exp.ambulance)

        # Aggregate sum-insured cap, computed on what survives the other caps.
        after_category = exp.total() - sum(l.deduction_inr for l in applied)
        if after_category > 0:
            add(
                L.SUM_INSURED_AGGREGATE,
                after_category,
                basis_override=(
                    f"Sum Insured in aggregate for the period of insurance = INR {sum_insured:,.0f}"
                ),
            )

        return applied

    # ------------------------------------------------------------------ #

    @staticmethod
    def _payable_estimate(
        case: ClaimCase,
        applied: list[AppliedLimit],
        findings: list[Finding],
        claimed_total: float,
    ) -> float | None:
        """Indicative payable amount after caps and excluded portions.

        Reported as an estimate, not an authorisation: it exists so a reviewer
        can see the financial effect of the limits the system identified.
        """
        if case.sum_insured_inr is None:
            return None

        blocking = any(
            f.severity == Severity.BLOCKING and f.status == FindingStatus.VIOLATED
            for f in findings
        )
        if blocking:
            return 0.0

        payable = claimed_total - sum(l.deduction_inr for l in applied)

        # Portions the policy excludes outright rather than caps.
        excluded_portion = 0.0
        for finding in findings:
            if finding.severity != Severity.LIMITING or finding.status != FindingStatus.VIOLATED:
                continue
            if finding.dimension == DecisionDimension.DOMICILIARY:
                excluded_portion += (
                    case.expenses_inr.pre_hospitalization
                    + case.expenses_inr.post_hospitalization
                )
            elif finding.dimension == DecisionDimension.PRE_POST_HOSPITALIZATION:
                excluded_portion += _out_of_window_amount(case)

        payable = max(0.0, payable - excluded_portion)
        return round(min(payable, case.sum_insured_inr), 2)


def _out_of_window_amount(case: ClaimCase) -> float:
    """Pre/post expenses falling outside the policy's 30/60-day windows."""
    timing = case.expense_timing
    if timing is None:
        return 0.0
    amount = 0.0
    if (timing.pre_hospitalization_days_before_admission or 0) > 30:
        amount += case.expenses_inr.pre_hospitalization
    if (timing.post_hospitalization_days_after_discharge or 0) > 60:
        amount += case.expenses_inr.post_hospitalization
    if timing.same_condition_confirmed is False:
        amount = case.expenses_inr.pre_hospitalization + case.expenses_inr.post_hospitalization
    return amount
