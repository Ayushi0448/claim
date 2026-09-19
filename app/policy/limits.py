"""Monetary limit calculators.

Each calculator is bound to the clause that authorises it. If the clause is not
present in the retrieved evidence, the limit is not applied — the system would
rather under-report a deduction than invent one.

All limits are expressed against the Basic Sum Insured, matching the wording in
"SCOPE OF COVER / WHAT WE COVER" on pages 7-8 of the supplied policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.models.schemas import AppliedLimit, ClaimCase, EvidenceItem
from app.policy.evidence_binding import Anchor, build_citation, resolve_anchors


@dataclass(frozen=True)
class LimitSpec:
    """Declarative description of one policy cap."""

    limit_id: str
    category: str
    description: str
    anchor: Anchor
    # Fraction of Basic Sum Insured; ``per_day`` caps multiply by hospital days.
    fraction_of_si: float | None = None
    absolute_cap_inr: float | None = None
    per_day: bool = False
    # When both a fraction and an absolute cap exist, the policy says
    # "whichever is less" for ambulance and daily allowance.
    take_lesser: bool = False


ROOM_RENT = LimitSpec(
    limit_id="limit_room_rent",
    category="room",
    description="Normal room expenses capped at 1.0% of Basic Sum Insured per day",
    anchor=Anchor(phrase="Normal Room expenses: 1.0% of Basic Sum Insured"),
    fraction_of_si=0.01,
    per_day=True,
)

ICU_RENT = LimitSpec(
    limit_id="limit_icu",
    category="icu",
    description="Intensive Care / Therapeutic Unit expenses capped at 2% of Basic Sum Insured per day",
    anchor=Anchor(phrase="Intensive Care/ Therapeutic Unit expenses: 2% of Basic Sum Insured"),
    fraction_of_si=0.02,
    per_day=True,
)

DOCTOR_FEES = LimitSpec(
    limit_id="limit_practitioner_fees",
    category="doctor_fees",
    description=(
        "Medical Practitioner / Anaesthetist / Consultant / Surgeon fees capped at "
        "25% of Sum Insured"
    ),
    anchor=Anchor(phrase="subject to a limit of 25% of Sum Assured"),
    fraction_of_si=0.25,
)

MEDICINES_DIAGNOSTICS = LimitSpec(
    limit_id="limit_medicines_diagnostics",
    category="medicines_diagnostics",
    description=(
        "Anaesthesia, blood, oxygen, operation theatre, surgical appliances, medicines, "
        "drugs, diagnostics, X-ray, dialysis, chemotherapy and similar expenses capped at "
        "40% of Sum Insured"
    ),
    anchor=Anchor(phrase="subject to a limit of 40% Sum Insured"),
    fraction_of_si=0.40,
)

AMBULANCE = LimitSpec(
    limit_id="limit_ambulance",
    category="ambulance",
    description=(
        "Ambulance charges capped at 1.0% of Basic Sum Insured or Rs 1,000, whichever is less"
    ),
    anchor=Anchor(phrase="Ambulance charges in connection with any admissible claim limited to"),
    fraction_of_si=0.01,
    absolute_cap_inr=1000.0,
    take_lesser=True,
)

DOMICILIARY_AGGREGATE = LimitSpec(
    limit_id="limit_domiciliary_aggregate",
    category="domiciliary_total",
    description="Domiciliary hospitalisation capped at an aggregate 20% of Basic Sum Insured",
    anchor=Anchor(phrase="maximum aggregate sub-limit of 20% of the Basic Sum Insured"),
    fraction_of_si=0.20,
)

SUM_INSURED_AGGREGATE = LimitSpec(
    limit_id="limit_sum_insured",
    category="aggregate",
    description="Total liability capped at the Sum Insured in aggregate for the period of insurance",
    anchor=Anchor(phrase="not exceeding the Sum Insured in aggregate"),
    fraction_of_si=1.0,
)


def hospital_days(case: ClaimCase) -> int:
    """Billable days, rounded up from admission hours.

    Room rent is charged "on per day (24 hours) basis" per the Room Rent
    definition on page 5, so a 96-hour stay is four billable days.
    """
    hours = case.treatment.admission_hours
    if hours is None or hours <= 0:
        return 0
    return max(1, math.ceil(hours / 24.0))


def _cap_amount(spec: LimitSpec, sum_insured: float, days: int) -> float | None:
    if spec.fraction_of_si is None and spec.absolute_cap_inr is None:
        return None
    fractional = (
        sum_insured * spec.fraction_of_si if spec.fraction_of_si is not None else None
    )
    if spec.per_day and fractional is not None:
        fractional = fractional * max(days, 0)

    if spec.take_lesser and fractional is not None and spec.absolute_cap_inr is not None:
        return min(fractional, spec.absolute_cap_inr)
    if fractional is not None:
        return fractional
    return spec.absolute_cap_inr


def apply_limit(
    spec: LimitSpec,
    *,
    claimed: float,
    sum_insured: float,
    days: int,
    evidence: list[EvidenceItem],
    basis_override: str | None = None,
) -> AppliedLimit | None:
    """Compute one cap, returning ``None`` if its clause was not retrieved.

    Returning ``None`` rather than silently applying the cap is deliberate: a
    deduction the system cannot cite is a deduction it must not make.
    """
    resolution = resolve_anchors(evidence, [spec.anchor])
    if not resolution.resolved or resolution.primary is None:
        return None

    cap = _cap_amount(spec, sum_insured, days)
    if cap is None:
        return None

    allowed = min(claimed, cap)
    deduction = max(0.0, claimed - cap)

    if basis_override:
        basis = basis_override
    elif spec.per_day and spec.fraction_of_si is not None:
        per_day_amount = sum_insured * spec.fraction_of_si
        basis = (
            f"{spec.fraction_of_si:.2%} of Sum Insured (INR {sum_insured:,.0f}) "
            f"= INR {per_day_amount:,.0f}/day x {days} day(s) = INR {cap:,.0f}"
        )
    elif spec.take_lesser and spec.absolute_cap_inr is not None and spec.fraction_of_si is not None:
        basis = (
            f"lesser of {spec.fraction_of_si:.2%} of Sum Insured "
            f"(INR {sum_insured * spec.fraction_of_si:,.0f}) and INR "
            f"{spec.absolute_cap_inr:,.0f} = INR {cap:,.0f}"
        )
    elif spec.fraction_of_si is not None:
        basis = (
            f"{spec.fraction_of_si:.0%} of Sum Insured (INR {sum_insured:,.0f}) "
            f"= INR {cap:,.0f}"
        )
    else:
        basis = f"INR {cap:,.0f}"

    claim_text = (
        f"{spec.description}. Claimed INR {claimed:,.0f}; allowed INR {allowed:,.0f}"
        + (f"; deduction INR {deduction:,.0f}." if deduction > 0 else ".")
    )

    return AppliedLimit(
        limit_id=spec.limit_id,
        category=spec.category,
        description=spec.description,
        basis=basis,
        limit_amount_inr=round(cap, 2),
        claimed_amount_inr=round(claimed, 2),
        allowed_amount_inr=round(allowed, 2),
        deduction_inr=round(deduction, 2),
        binding=deduction > 0.009,
        evidence_chunk_ids=resolution.chunk_ids,
        citations=[
            build_citation(
                claim_text,
                resolution.primary,
                rule_id=spec.limit_id,
                anchor_phrase=spec.anchor.phrase,
            )
        ],
    )
