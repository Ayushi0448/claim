"""Agent 1 — Case Analysis.

Responsibility: turn a raw claim into an *investigation*.

It extracts and derives facts, decides which decision dimensions this specific
claim actually turns on, notes what is missing, flags attributes irrelevant to
the policy decision (RULE 9), and emits one targeted retrieval query per
dimension (RULE 7/8) — never a single generic query.

Dimension selection is claim-driven: a domiciliary claim pulls in the
domiciliary dimension and its sub-limit; an 8-hour admission pulls in the
day-care dimension. This is what makes the retrieval plan specific rather than
a fixed checklist run against every claim.
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    CaseAnalysis,
    ClaimCase,
    DecisionDimension as D,
    InvestigationItem,
    RetrievalQuery,
)
from app.policy.limits import hospital_days
from app.services.llm import LLMClient, get_llm_client
from app.utils.dates import days_between, months_between, parse_date

logger = logging.getLogger(__name__)


# Attributes present in the dataset that carry no policy consequence. Naming
# them explicitly is better than silently ignoring them: the reviewer can see
# the system considered and discounted them.
_POLICY_IRRELEVANT = {
    "hospital.name": "The policy does not decide admissibility by hospital name.",
    "hospital.network_provider": (
        "Network status affects cashless settlement, not whether the claim is admissible."
    ),
    "patient.age": (
        "This policy applies no age-based limit to the benefits engaged by this claim."
    ),
    "task": "Reviewer instruction text, not a policy fact.",
    "policy_id": "Identifier only.",
}

# The query text for each dimension, plus expansions that bridge the gap
# between claim vocabulary and policy vocabulary.
_DIMENSION_QUERIES: dict[D, tuple[str, str, tuple[str, ...]]] = {
    D.SCOPE_OF_COVER: (
        "what hospitalization expenses does the policy cover reasonable and customary charges",
        "Establishes the positive limb of cover before limits and exclusions are applied.",
        ("room boarding nursing", "medical practitioner fees", "scope of cover"),
    ),
    # Claim documentation gets its own dimension rather than riding along with
    # the scope query. Folding it into SCOPE_OF_COVER diluted that query until
    # the scope clause itself dropped out of retrieval and five cases
    # over-abstained. One dimension, one information need.
    # See docs/failure-analysis.md F-04.
    D.CLAIM_DOCUMENTATION: (
        "claim documents required original attested photocopies of all bills receipts "
        "certificates information evidences from attending medical practitioner",
        "The policy conditions payment on the documentary evidence the claimant supplies.",
        ("reimbursement claims process", "claim form", "notice of claim within 7 days"),
    ),
    D.HOSPITALIZATION_DEFINITION: (
        "hospitalization means admission in a hospital for a minimum period of 24 consecutive hours",
        "The claim must meet the policy's definition of Hospitalization.",
        ("in-patient care", "minimum period of 24 hours", "specified procedures"),
    ),
    D.HOSPITAL_DEFINITION: (
        "hospital means institution registered local authorities minimum criteria "
        "qualified nursing staff in-patient beds operation theatre",
        "Treatment must be rendered in an institution meeting the policy's Hospital definition.",
        ("clinical establishments act", "network provider means", "10 in-patient beds"),
    ),
    D.DAY_CARE: (
        "day care treatment less than 24 hours technological advancement eye surgery "
        "dialysis chemotherapy 140 day care procedures",
        "A sub-24-hour admission is covered only through the day-care provisions.",
        ("minimum stay of 24 hours can be waived", "day care centre", "general or local anesthesia"),
    ),
    D.DOMICILIARY: (
        "domiciliary treatment at home non-availability of room patient cannot be removed "
        "domiciliary hospitalisation sub-limit 20% basic sum insured",
        "Domiciliary treatment has its own qualifying conditions and its own aggregate cap.",
        ("confined at home", "maximum aggregate sub-limit", "any expense under domiciliary"),
    ),
    D.WAITING_PERIOD_INITIAL: (
        "waiting period of 30 days will apply to all claims unless continuously insured",
        "A claim inside the initial waiting period is not payable regardless of the treatment.",
        ("previous policy year", "other Indian insurer individual health insurance policy"),
    ),
    D.WAITING_PERIOD_FIRST_YEAR: (
        "hospitalization expense incurred in the first year of operation cataract hernia "
        "piles arthritis waiting period of 1 year will not apply",
        "Named conditions carry a one-year waiting period with a continuity waiver.",
        ("database and claim history", "completed years of coverage"),
    ),
    D.PRE_EXISTING_DISEASE: (
        "pre-existing diseases will not be covered until 48 months of continuous coverage "
        "have elapsed since inception",
        "Pre-existing conditions carry a 48-month waiting period with a continuity reduction.",
        ("pre-existing diseases means", "reduced by the number of continuous preceding years"),
    ),
    D.PORTABILITY_CONTINUITY: (
        "continuous coverage with another Indian insurer reduces waiting period database "
        "and claim history received portability",
        "Prior continuous cover can reduce or remove a waiting period.",
        ("portability means", "credit gained for pre-existing conditions"),
    ),
    D.EXCLUSIONS: (
        "what we exclude cosmetic aesthetic treatment dental pregnancy HIV outpatient "
        "congenital naturopathy excluded diseases",
        "The claim must not fall within any exclusion.",
        ("plastic surgery except", "treatment of following diseases", "not approved by Indian Medical council"),
    ),
    D.EXPERIMENTAL_TREATMENT: (
        "unproven experimental treatment not based on established medical practice in India "
        "treatments not approved by Indian Medical council",
        "Experimental or unproven treatment is outside the cover.",
        ("medically necessary means", "professional standards widely accepted"),
    ),
    D.ROOM_RENT_LIMIT: (
        "normal room expenses 1.0% of basic sum insured per day intensive care 2% room rent "
        "charged on per day 24 hours basis",
        "Room rent is capped per day and frequently produces the largest deduction.",
        ("sub limits", "registration charges", "room rent means"),
    ),
    D.CATEGORY_SUBLIMITS: (
        "medical practitioner anesthetist consultant surgeon fees limit 25% of sum assured "
        "medicines drugs diagnostic materials 40% sum insured",
        "Category caps determine the payable amount independently of admissibility.",
        ("expenses on anesthesia blood oxygen operation theatre", "agreed package charges 75%"),
    ),
    D.AMBULANCE_LIMIT: (
        "ambulance charges in connection with any admissible claim limited to 1.0% of basic "
        "sum insured or rupees 1000 whichever is less",
        "Ambulance charges carry a small absolute cap that is easy to overlook.",
        ("additional benefits", "daily allowance"),
    ),
    D.PRE_POST_HOSPITALIZATION: (
        "pre-hospitalisation up to a maximum of 30 days immediately preceding and post "
        "hospitalisation 60 days immediately following",
        "Pre and post expenses are payable only inside fixed windows for the same condition.",
        ("pre- hospitalization medical expenses means", "post hospitalization medical expenses means"),
    ),
    D.MEDICAL_NECESSITY: (
        "medically necessary means required for medical management prescribed by a medical "
        "practitioner professional standards",
        "Medical necessity is a condition precedent to payment.",
        ("reasonable and customary charges", "medical expenses means"),
    ),
    D.SUM_INSURED_AGGREGATE: (
        "not exceeding the sum insured in aggregate in any one period of insurance",
        "Total liability is capped at the sum insured.",
        ("sum insured", "maximum limit of indemnity"),
    ),
}


class CaseAnalysisAgent:
    name = "case_analysis"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm_client()

    # ------------------------------------------------------------------ #

    def _derive_facts(self, case: ClaimCase) -> tuple[dict, dict, list[str]]:
        warnings: list[str] = []
        start = parse_date(case.policy_start_date)
        claim = parse_date(case.claim_date)

        if case.policy_start_date and start is None:
            warnings.append(f"Unparseable policy_start_date: {case.policy_start_date!r}")
        if case.claim_date and claim is None:
            warnings.append(f"Unparseable claim_date: {case.claim_date!r}")
        if start and claim and claim < start:
            warnings.append("claim_date precedes policy_start_date")

        treatment_type = (case.treatment.type or "").lower()
        elapsed_months = months_between(start, claim)
        stated_months = case.continuous_coverage_months

        # The dataset supplies continuous_coverage_months explicitly; prefer it
        # but reconcile against the dates so a contradiction is surfaced.
        months_covered = stated_months if stated_months is not None else elapsed_months
        if (
            stated_months is not None
            and elapsed_months is not None
            and abs(stated_months - elapsed_months) > 1.5
        ):
            warnings.append(
                f"continuous_coverage_months ({stated_months}) disagrees with the policy dates "
                f"({elapsed_months:.1f} months elapsed); the stated value is used."
            )

        prior_years = case.prior_insurer_continuous_years or 0
        prior_policy = case.prior_policy
        history_received = bool(
            prior_policy and prior_policy.database_and_claim_history_received is True
        )
        if prior_policy and prior_policy.continuous_years:
            prior_years = max(prior_years, prior_policy.continuous_years)

        # The policy credits prior cover only when the previous insurer's
        # database and claim history have been received (NB to exclusion 3).
        credited_prior_years = int(prior_years) if history_received else 0
        first_year_waiver = prior_years >= 1 and history_received

        facts = {
            "case_id": case.case_id,
            "policy_start_date": case.policy_start_date,
            "claim_date": case.claim_date,
            "sum_insured_inr": case.sum_insured_inr,
            "treatment_type": treatment_type or None,
            "diagnosis": case.treatment.diagnosis,
            "procedure": case.treatment.procedure,
            "admission_hours": case.treatment.admission_hours,
            "pre_existing": case.treatment.pre_existing,
            "experimental": case.treatment.experimental,
            "documents": list(case.documents),
            "claimed_total_inr": case.expenses_inr.total(),
        }

        derived = {
            "days_since_policy_start": days_between(start, claim),
            "elapsed_months": elapsed_months,
            "months_covered": months_covered,
            "prior_insurer_years": prior_years,
            "prior_history_received": history_received,
            "credited_prior_years": credited_prior_years,
            "first_year_waiver_available": first_year_waiver,
            "hospital_days": hospital_days(case),
            "is_inpatient": treatment_type in {"inpatient", "in-patient"},
            "is_day_care": treatment_type in {"day_care", "daycare"},
            "is_domiciliary": treatment_type in {"domiciliary", "home"},
            "is_outpatient": treatment_type in {"outpatient", "opd"},
            "evidence_context_keys": sorted(
                (case.evidence_context.model_dump(exclude_unset=True) or {}).keys()
            ),
        }
        return facts, derived, warnings

    # ------------------------------------------------------------------ #

    def _select_dimensions(self, case: ClaimCase, derived: dict) -> list[D]:
        """Choose the dimensions this specific claim actually turns on."""
        dims: list[D] = [
            D.SCOPE_OF_COVER,
            D.CLAIM_DOCUMENTATION,
            D.WAITING_PERIOD_INITIAL,
            D.EXCLUSIONS,
            D.SUM_INSURED_AGGREGATE,
        ]

        if derived["is_inpatient"] or derived["is_day_care"]:
            dims += [D.HOSPITALIZATION_DEFINITION, D.HOSPITAL_DEFINITION]
        if derived["is_day_care"] or (
            case.treatment.admission_hours is not None and case.treatment.admission_hours < 24
        ):
            dims.append(D.DAY_CARE)
        if derived["is_domiciliary"]:
            dims.append(D.DOMICILIARY)

        if case.treatment.pre_existing is True or case.treatment.pre_existing is None:
            dims.append(D.PRE_EXISTING_DISEASE)
        # The first-year list is checked whenever coverage is young enough to matter.
        months = derived.get("months_covered")
        if months is None or months < 24:
            dims.append(D.WAITING_PERIOD_FIRST_YEAR)
        if (derived.get("prior_insurer_years") or 0) > 0 or case.prior_policy is not None:
            dims.append(D.PORTABILITY_CONTINUITY)
        if case.treatment.experimental is True:
            dims.append(D.EXPERIMENTAL_TREATMENT)

        exp = case.expenses_inr
        if exp.room > 0 or exp.icu > 0:
            dims.append(D.ROOM_RENT_LIMIT)
        if exp.doctor_fees > 0 or exp.medicines_diagnostics > 0:
            dims.append(D.CATEGORY_SUBLIMITS)
        if exp.ambulance > 0:
            dims.append(D.AMBULANCE_LIMIT)
        if exp.pre_hospitalization > 0 or exp.post_hospitalization > 0:
            dims.append(D.PRE_POST_HOSPITALIZATION)

        if "medical_necessity_confirmed" in derived.get("evidence_context_keys", []):
            dims.append(D.MEDICAL_NECESSITY)

        # Preserve insertion order, drop duplicates.
        return list(dict.fromkeys(dims))

    # ------------------------------------------------------------------ #

    def _missing_fields(self, case: ClaimCase, derived: dict) -> list[str]:
        missing: list[str] = []
        if case.policy_start_date is None:
            missing.append("policy_start_date")
        if case.claim_date is None:
            missing.append("claim_date")
        if case.sum_insured_inr is None:
            missing.append("sum_insured_inr")
        if case.treatment.type is None:
            missing.append("treatment.type")
        if case.treatment.admission_hours is None and not derived["is_domiciliary"]:
            missing.append("treatment.admission_hours")
        if case.treatment.pre_existing is None:
            missing.append("treatment.pre_existing")
        if not case.documents:
            missing.append("documents")

        ec = case.evidence_context.model_dump(exclude_unset=True)
        for key, value in ec.items():
            if value is None:
                missing.append(f"evidence_context.{key}")
        return missing

    def _irrelevant(self, case: ClaimCase) -> list[str]:
        raw = case.model_dump()
        present: list[str] = []
        for path, reason in _POLICY_IRRELEVANT.items():
            head, _, tail = path.partition(".")
            value = raw.get(head)
            if tail and isinstance(value, dict):
                value = value.get(tail)
            if value not in (None, "", [], {}):
                present.append(f"{path} — {reason}")
        return present

    # ------------------------------------------------------------------ #

    def _build_queries(
        self, dims: list[D], case: ClaimCase, derived: dict
    ) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        condition = " ".join(
            p for p in (case.treatment.diagnosis, case.treatment.procedure) if p
        ).strip()

        # Pass 1: one broad topical query per decision dimension.
        for dim in dims:
            base, rationale, expansions = _DIMENSION_QUERIES[dim]
            exp = list(expansions)
            # Ground the exclusion and waiting-period searches in this claim's
            # own condition so the lexical arm can find the matching list entry.
            if condition and dim in {
                D.EXCLUSIONS,
                D.WAITING_PERIOD_FIRST_YEAR,
                D.EXPERIMENTAL_TREATMENT,
                D.SCOPE_OF_COVER,
            }:
                exp.append(condition)
            queries.append(
                RetrievalQuery(dimension=dim, query=base, rationale=rationale, expansions=exp)
            )

        # Pass 2: one precise query per clause the applicable rules depend on.
        # Broad queries find the right *area* of the policy; these find the
        # exact clause, which is what the evidence-binding step requires.
        from app.policy.rules import required_anchors_for_case

        facts = {**derived}
        seen_queries = {q.query for q in queries}
        for dimension, phrase in required_anchors_for_case(case, facts):
            if phrase in seen_queries:
                continue
            seen_queries.add(phrase)
            queries.append(
                RetrievalQuery(
                    dimension=dimension,
                    query=phrase,
                    rationale=(
                        "Targeted retrieval of a clause a policy rule for this case depends on."
                    ),
                    expansions=[],
                )
            )

        self._llm_expand(queries, case)
        return queries

    def _llm_expand(self, queries: list[RetrievalQuery], case: ClaimCase) -> None:
        """Optionally add LLM-proposed phrasings. Failure is a no-op."""
        if not self.llm.enabled or not queries:
            return
        try:
            payload = {
                "diagnosis": case.treatment.diagnosis,
                "procedure": case.treatment.procedure,
                "treatment_type": case.treatment.type,
                "dimensions": [q.dimension.value for q in queries],
            }
            data = self.llm.complete_json(
                system=(
                    "You expand search queries for an insurance policy retrieval system. "
                    "Return ONLY a JSON object mapping each dimension name to an array of at "
                    "most 2 short alternative phrasings that might appear in an Indian health "
                    "insurance policy wording. Do not invent policy rules."
                ),
                user=str(payload),
                max_tokens=400,
            )
            if isinstance(data, dict):
                by_dim = {q.dimension.value: q for q in queries}
                for key, values in data.items():
                    query = by_dim.get(key)
                    if query and isinstance(values, list):
                        query.expansions.extend(str(v)[:120] for v in values[:2])
        except Exception as exc:  # pragma: no cover - optional path
            logger.debug("LLM query expansion skipped: %s", exc)

    # ------------------------------------------------------------------ #

    def run(self, case: ClaimCase) -> tuple[CaseAnalysis, list[RetrievalQuery]]:
        facts, derived, warnings = self._derive_facts(case)
        dims = self._select_dimensions(case, derived)
        queries = self._build_queries(dims, case, derived)

        plan = [
            InvestigationItem(
                dimension=dim,
                question=_DIMENSION_QUERIES[dim][0],
                why_it_matters=_DIMENSION_QUERIES[dim][1],
                required_inputs=_REQUIRED_INPUTS.get(dim, []),
                blocking_if_unresolved=dim in _BLOCKING_DIMENSIONS,
            )
            for dim in dims
        ]

        analysis = CaseAnalysis(
            case_id=case.case_id,
            facts=facts,
            derived_facts=derived,
            decision_dimensions=dims,
            missing_fields=self._missing_fields(case, derived),
            irrelevant_attributes=self._irrelevant(case),
            investigation_plan=plan,
            input_warnings=warnings,
        )
        return analysis, queries


_REQUIRED_INPUTS: dict[D, list[str]] = {
    D.WAITING_PERIOD_INITIAL: ["policy_start_date", "claim_date", "continuous_coverage_months"],
    D.PRE_EXISTING_DISEASE: ["treatment.pre_existing", "continuous_coverage_months"],
    D.WAITING_PERIOD_FIRST_YEAR: ["treatment.diagnosis", "continuous_coverage_months"],
    D.HOSPITAL_DEFINITION: ["evidence_context.hospital_registered", "hospital.network_provider"],
    D.HOSPITALIZATION_DEFINITION: ["treatment.admission_hours"],
    D.DAY_CARE: ["treatment.procedure", "treatment.admission_hours"],
    D.DOMICILIARY: [
        "treatment.hospital_room_unavailable",
        "treatment.patient_cannot_be_moved",
    ],
    D.MEDICAL_NECESSITY: ["evidence_context.medical_necessity_confirmed"],
    D.ROOM_RENT_LIMIT: ["sum_insured_inr", "expenses_inr.room", "treatment.admission_hours"],
    D.CATEGORY_SUBLIMITS: ["sum_insured_inr", "expenses_inr"],
    D.PRE_POST_HOSPITALIZATION: ["expense_timing"],
}

_BLOCKING_DIMENSIONS = {
    D.HOSPITAL_DEFINITION,
    D.HOSPITALIZATION_DEFINITION,
    D.PRE_EXISTING_DISEASE,
    D.WAITING_PERIOD_INITIAL,
    D.DOMICILIARY,
    D.MEDICAL_NECESSITY,
}
