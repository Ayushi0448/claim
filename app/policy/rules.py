"""The policy rule engine.

Each rule below encodes one provision of the *supplied* policy wording and is
bound to the clause that authorises it (see ``evidence_binding``). A rule
cannot fire unless its anchor text was genuinely retrieved for the case at hand.

Why rules rather than an LLM verdict
------------------------------------
Admissibility here is a deterministic function of the policy: 27 months of
coverage is less than the 48 months a pre-existing condition requires, and no
amount of language modelling changes that. Encoding it as rules gives exact
arithmetic, reproducible evaluation, and — most importantly — an auditable link
from every statement to the clause behind it. The LLM layer (optional) writes
prose and offers a second opinion during validation; it never decides.

Where a listed condition is involved (the first-year disease list, the
exclusion lists), the matched term must appear **in the retrieved clause text**
as well as in the claim, so the citation genuinely supports the statement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import (
    ClaimCase,
    DecisionDimension,
    EvidenceItem,
    Finding,
    FindingStatus,
    MissingEvidence,
    Severity,
)
from app.policy.evidence_binding import (
    Anchor,
    AnchorResolution,
    build_citation,
    resolve_anchors,
)
from app.utils.text import contains_phrase

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Condition vocabularies, taken verbatim from the supplied policy's own lists.
# Every term is re-verified against the retrieved clause before a rule uses it.
# --------------------------------------------------------------------------- #

FIRST_YEAR_CONDITIONS = (
    "cataract", "benign prostatic hypertrophy", "myomectomy", "hysterectomy",
    "hernia", "hydrocele", "fistula in anus", "piles", "arthritis", "gout",
    "rheumatism", "joint replacement", "sinusitis", "stone in the urinary",
    "biliary", "dilatation and curettage", "tumors", "cysts", "nodules",
    "polyps", "breast lumps", "adenoids", "hemorrhoids", "dialysis",
    "tonsils", "gastric", "duodenal ulcers",
)

COSMETIC_TERMS = (
    "cosmetic", "aesthetic", "plastic surgery", "circumcision", "vaccination",
    "inoculation",
)

DENTAL_TERMS = ("dental",)

GENERAL_EXCLUSION_TERMS = (
    "convalescence", "general debility", "run down", "rest cure", "congenital",
    "sterility", "venereal", "self injury", "intoxicating",
)

HIV_TERMS = ("hiv", "aids")

MATERNITY_TERMS = (
    "pregnancy", "childbirth", "miscarriage", "abortion", "caesarean",
    "infertility", "sub fertility", "assisted conception",
)

UNAPPROVED_TREATMENT_TERMS = (
    "naturopathy", "non-allopathic", "adventurous sports",
)

# Exclusion 20's disease list. NOTE: in the supplied PDF this list is rendered
# as a flat numbered item, but the surrounding wording ("Any expense under
# Domiciliary Hospitalisation for") indicates it may be scoped to domiciliary
# treatment. That ambiguity is handled explicitly in the rule below rather than
# resolved by assumption — see docs/failure-analysis.md, finding F-02.
LISTED_DISEASE_TERMS = (
    "asthma", "bronchitis", "chronic nephritis", "nephritic syndrome",
    "diarrhoea", "dysenteries", "gastro-enteritis", "diabetes mellitus",
    "epilepsy", "hypertension", "influenza", "cough and cold", "psychiatric",
    "psychosomatic", "pyrexia of unknown origin", "tonsillitis",
    "upper respiratory tract", "laryngitis", "pharingitis",
)

# NB4's explicitly named sub-24-hour procedures.
NB4_PROCEDURES = (
    "dialysis", "chemotherapy", "radiotherapy", "eye surgery", "lithotripsy",
    "tonsillectomy", "d&c",
)


# --------------------------------------------------------------------------- #
# Rule infrastructure
# --------------------------------------------------------------------------- #


@dataclass
class RuleContext:
    case: ClaimCase
    facts: dict[str, Any]
    evidence: list[EvidenceItem]
    missing_evidence: list[MissingEvidence] = field(default_factory=list)

    @property
    def condition_text(self) -> str:
        parts = [
            self.case.treatment.diagnosis or "",
            self.case.treatment.procedure or "",
        ]
        return " ".join(parts).lower()

    def add_missing(
        self,
        item: str,
        dimension: DecisionDimension,
        why: str,
        *,
        blocking: bool = False,
        document: str | None = None,
    ) -> None:
        if any(m.item == item for m in self.missing_evidence):
            return
        self.missing_evidence.append(
            MissingEvidence(
                item=item,
                dimension=dimension,
                why_required=why,
                blocking=blocking,
                suggested_document=document,
            )
        )


@dataclass
class PolicyRule:
    rule_id: str
    dimension: DecisionDimension
    anchors: list[Anchor]
    applies: Callable[[RuleContext], bool]
    evaluate: Callable[[RuleContext, AnchorResolution], Finding | None]
    require_all_anchors: bool = False
    unresolved_severity: Severity = Severity.ABSTAIN
    unresolved_statement: str = "Required policy clause was not retrieved for this dimension."


REGISTRY: list[PolicyRule] = []


def register(rule: PolicyRule) -> PolicyRule:
    REGISTRY.append(rule)
    return rule


def _finding(
    rule: PolicyRule,
    status: FindingStatus,
    severity: Severity,
    statement: str,
    resolution: AnchorResolution,
    *,
    detail: str | None = None,
    anchor_phrase: str | None = None,
) -> Finding:
    citations = [
        build_citation(statement, item, rule_id=rule.rule_id, anchor_phrase=anchor_phrase)
        for item in resolution.items
    ]
    return Finding(
        rule_id=rule.rule_id,
        dimension=rule.dimension,
        status=status,
        severity=severity,
        statement=statement,
        detail=detail,
        evidence_chunk_ids=resolution.chunk_ids,
        citations=citations,
        evidence_supported=bool(citations),
    )


def _matched_listed_term(
    ctx: RuleContext, resolution: AnchorResolution, vocabulary: tuple[str, ...]
) -> str | None:
    """Return a term present in BOTH the claim and the retrieved clause text.

    Requiring the term to appear in the clause is what keeps the citation
    honest: the system may only say "cataract is a first-year condition" when
    the retrieved chunk actually lists cataract.
    """
    claim_text = ctx.condition_text
    clause_text = " ".join(item.text for item in resolution.items)
    for term in vocabulary:
        if contains_phrase(claim_text, term) and contains_phrase(clause_text, term):
            return term
    return None


# --------------------------------------------------------------------------- #
# 1. Initial 30-day waiting period
# --------------------------------------------------------------------------- #


def _eval_initial_waiting(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_INITIAL_WAITING
    days = ctx.facts.get("days_since_policy_start")
    if days is None:
        ctx.add_missing(
            "policy_start_date or claim_date",
            rule.dimension,
            "The 30-day initial waiting period cannot be evaluated without both dates.",
            blocking=True,
        )
        return _finding(
            rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
            "Initial waiting period could not be evaluated: policy start or claim date is missing.",
            res,
        )

    continuity_prev_year = (ctx.facts.get("months_covered") or 0) >= 12
    prior_year_cover = (ctx.facts.get("prior_insurer_years") or 0) >= 1

    if days < 30 and not (continuity_prev_year or prior_year_cover):
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            f"Claim falls within the 30-day initial waiting period: only {days} day(s) "
            f"elapsed between policy inception and the claim date, and no qualifying "
            f"continuous prior coverage is evidenced.",
            res,
            detail=(
                "The policy applies a 30-day waiting period to all claims unless the insured "
                "was continuously covered in the previous policy year or for at least one year "
                "under another Indian insurer's individual health policy."
            ),
            anchor_phrase="waiting period of 30 days will apply to all claims",
        )

    if days < 30:
        basis = "continuous cover in the previous policy year" if continuity_prev_year else (
            "at least one year of continuous cover with another Indian insurer"
        )
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            f"The 30-day initial waiting period does not bar this claim ({days} day(s) elapsed) "
            f"because the insured has {basis}.",
            res, anchor_phrase="waiting period of 30 days will apply to all claims",
        )

    return _finding(
        rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
        f"The 30-day initial waiting period is satisfied: {days} day(s) elapsed since policy inception.",
        res, anchor_phrase="waiting period of 30 days will apply to all claims",
    )


RULE_INITIAL_WAITING = register(
    PolicyRule(
        rule_id="waiting_period_initial_30d",
        dimension=DecisionDimension.WAITING_PERIOD_INITIAL,
        anchors=[Anchor(phrase="waiting period of 30 days will apply to all claims")],
        applies=lambda ctx: True,
        evaluate=_eval_initial_waiting,
        unresolved_statement="The 30-day waiting period clause was not retrieved; admissibility cannot be confirmed.",
    )
)


# --------------------------------------------------------------------------- #
# 2. Pre-existing disease — 48 months
# --------------------------------------------------------------------------- #


def _eval_ped(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_PED
    months = ctx.facts.get("months_covered")
    if months is None:
        return _finding(
            rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
            "Pre-existing disease waiting period could not be evaluated: continuous coverage is unknown.",
            res,
        )

    credited_years = ctx.facts.get("credited_prior_years") or 0
    effective_months = months + credited_years * 12

    if effective_months < 48:
        shortfall = 48 - effective_months
        detail = (
            f"Continuous coverage under this policy is {months:.0f} month(s)."
            + (
                f" Prior continuous coverage of {credited_years} year(s) is credited, giving "
                f"{effective_months:.0f} month(s) effective."
                if credited_years
                else ""
            )
            + f" The policy requires 48 months; the claim is short by {shortfall:.0f} month(s)."
        )
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            "The condition is declared pre-existing and the 48-month pre-existing disease "
            f"waiting period has not elapsed ({effective_months:.0f} of 48 months completed).",
            res, detail=detail,
            anchor_phrase="Pre-existing diseases will not be covered until 48 months",
        )

    return _finding(
        rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
        f"The 48-month pre-existing disease waiting period is satisfied "
        f"({effective_months:.0f} month(s) of continuous coverage).",
        res, anchor_phrase="Pre-existing diseases will not be covered until 48 months",
    )


RULE_PED = register(
    PolicyRule(
        rule_id="pre_existing_disease_48m",
        dimension=DecisionDimension.PRE_EXISTING_DISEASE,
        anchors=[Anchor(phrase="Pre-existing diseases will not be covered until 48 months")],
        applies=lambda ctx: ctx.case.treatment.pre_existing is True,
        evaluate=_eval_ped,
        unresolved_statement=(
            "The pre-existing disease clause was not retrieved; a pre-existing condition "
            "claim cannot be safely decided."
        ),
    )
)


def _eval_ped_unknown(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_PED_UNKNOWN
    ctx.add_missing(
        "pre_existing status of the diagnosed condition",
        rule.dimension,
        "The policy applies a 48-month waiting period to pre-existing conditions, so this "
        "status must be established before the claim can be decided.",
        blocking=True,
        document="prior_medical_records",
    )
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        "Whether the condition is pre-existing has not been established, and the policy "
        "applies a 48-month waiting period to pre-existing conditions.",
        res, anchor_phrase="Pre-existing diseases will not be covered until 48 months",
    )


RULE_PED_UNKNOWN = register(
    PolicyRule(
        rule_id="pre_existing_disease_unknown",
        dimension=DecisionDimension.PRE_EXISTING_DISEASE,
        anchors=[Anchor(phrase="Pre-existing diseases will not be covered until 48 months")],
        applies=lambda ctx: ctx.case.treatment.pre_existing is None,
        evaluate=_eval_ped_unknown,
    )
)


# --------------------------------------------------------------------------- #
# 3. First-year disease waiting period
# --------------------------------------------------------------------------- #


def _eval_first_year(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_FIRST_YEAR
    term = _matched_listed_term(ctx, res, FIRST_YEAR_CONDITIONS)
    if term is None:
        return None  # condition is not on the policy's first-year list

    months = ctx.facts.get("months_covered")
    if months is None:
        return _finding(
            rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
            f"'{term}' is subject to a first-year waiting period but continuous coverage is unknown.",
            res,
        )

    waiver = ctx.facts.get("first_year_waiver_available") is True
    if months < 12 and not waiver:
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            f"Treatment for '{term}' is excluded in the first year of the cover and only "
            f"{months:.0f} month(s) of coverage have elapsed.",
            res,
            detail=(
                "The one-year waiting period is waived only where the insured was continuously "
                "insured for at least one year under this or another Indian insurer's individual "
                "health policy and the previous insurer's database and claim history have been received."
            ),
            anchor_phrase="first year of operation of the insurance cover",
        )

    if months < 12 and waiver:
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            f"The first-year waiting period for '{term}' is waived: the insured held "
            f"{ctx.facts.get('prior_insurer_years')} year(s) of continuous cover with another "
            "Indian insurer and the previous insurer's database and claim history were received.",
            res, anchor_phrase="a waiting period of 1 year will not apply",
        )

    return _finding(
        rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
        f"The first-year waiting period for '{term}' is satisfied ({months:.0f} months of coverage).",
        res, anchor_phrase="first year of operation of the insurance cover",
    )


RULE_FIRST_YEAR = register(
    PolicyRule(
        rule_id="waiting_period_first_year",
        dimension=DecisionDimension.WAITING_PERIOD_FIRST_YEAR,
        anchors=[
            Anchor(phrase="first year of operation of the insurance cover"),
            Anchor(phrase="a waiting period of 1 year will not apply"),
        ],
        # Only relevant if the claim names a condition on the policy's own list.
        applies=lambda ctx: _claim_mentions(ctx, FIRST_YEAR_CONDITIONS),
        evaluate=_eval_first_year,
        unresolved_severity=Severity.ABSTAIN,
        unresolved_statement=(
            "The claim names a condition subject to a first-year waiting period but that "
            "clause was not retrieved; the claim cannot be safely decided."
        ),
    )
)


# --------------------------------------------------------------------------- #
# 4. Exclusions driven by the claim's own description
# --------------------------------------------------------------------------- #


def _claim_mentions(ctx: RuleContext, vocabulary: tuple[str, ...]) -> bool:
    """Is any term from this exclusion's vocabulary present in the claim?

    Gating on this keeps the finding list meaningful. Running the dental
    exclusion against an appendectomy produced an UNRESOLVED finding whose real
    meaning was "the dental clause was not retrieved, because nothing asked for
    it" — noise that buried the findings that mattered. A rule that cannot
    possibly bite is simply not applicable.

    This is a *pre-filter on the claim only*. The matched term must still be
    verified against the retrieved clause text before the rule fires, so the
    evidence-binding guarantee is unchanged.
    """
    text = ctx.condition_text
    return any(contains_phrase(text, term) for term in vocabulary)


def _make_exclusion_rule(
    rule_id: str,
    anchor: Anchor,
    vocabulary: tuple[str, ...],
    label: str,
    *,
    dimension: DecisionDimension = DecisionDimension.EXCLUSIONS,
) -> PolicyRule:
    def _evaluate(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
        rule = next(r for r in REGISTRY if r.rule_id == rule_id)
        term = _matched_listed_term(ctx, res, vocabulary)
        if term is None:
            return None
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            f"The treatment is excluded: the claim describes '{term}', which falls within the "
            f"policy's {label} exclusion.",
            res, anchor_phrase=anchor.phrase,
        )

    return register(
        PolicyRule(
            rule_id=rule_id,
            dimension=dimension,
            anchors=[anchor],
            applies=lambda ctx: _claim_mentions(ctx, vocabulary),
            evaluate=_evaluate,
            # If the claim *does* mention an excluded term but the clause was
            # not retrieved, that is a genuine evidence gap and must abstain.
            unresolved_severity=Severity.ABSTAIN,
            unresolved_statement=(
                f"The claim engages the policy's {label} exclusion but that clause was not "
                "retrieved; the claim cannot be safely decided."
            ),
        )
    )


RULE_COSMETIC = _make_exclusion_rule(
    "exclusion_cosmetic",
    Anchor(phrase="cosmetic or aesthetic treatment of any description"),
    COSMETIC_TERMS,
    "cosmetic and aesthetic treatment",
)
RULE_DENTAL = _make_exclusion_rule(
    "exclusion_dental",
    Anchor(phrase="Dental treatment or surgery of any kind"),
    DENTAL_TERMS,
    "dental treatment",
)
RULE_GENERAL = _make_exclusion_rule(
    "exclusion_general_conditions",
    Anchor(phrase="Convalescence, general debility"),
    GENERAL_EXCLUSION_TERMS,
    "convalescence / congenital / self-inflicted conditions",
)
RULE_HIV = _make_exclusion_rule(
    "exclusion_hiv",
    Anchor(phrase="treatment related to HIV, AIDS"),
    HIV_TERMS,
    "HIV / AIDS",
)
RULE_MATERNITY = _make_exclusion_rule(
    "exclusion_maternity",
    Anchor(phrase="traceable to pregnancy, childbirth, miscarriage, abortion"),
    MATERNITY_TERMS,
    "maternity and infertility",
)
RULE_UNAPPROVED = _make_exclusion_rule(
    "exclusion_unapproved_treatment",
    Anchor(phrase="treatments not approved by Indian Medical council"),
    UNAPPROVED_TREATMENT_TERMS,
    "non-allopathic / unapproved treatment",
)


# --------------------------------------------------------------------------- #
# 5. Experimental / unproven treatment
# --------------------------------------------------------------------------- #


def _eval_experimental(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_EXPERIMENTAL
    return _finding(
        rule, FindingStatus.VIOLATED, Severity.BLOCKING,
        "The treatment is recorded as experimental. The policy defines "
        "Unproven/Experimental Treatment as treatment not based on established medical "
        "practice in India, and excludes treatments not approved by the Indian Medical "
        "Council; such treatment also fails the policy's Medically Necessary test.",
        res,
        detail=(
            "Two independent provisions support this: the Unproven/Experimental Treatment "
            "definition and the exclusion of treatments not approved by the Indian Medical Council."
        ),
        anchor_phrase="Unproven/Experimental Treatment means",
    )


RULE_EXPERIMENTAL = register(
    PolicyRule(
        rule_id="exclusion_experimental_treatment",
        dimension=DecisionDimension.EXPERIMENTAL_TREATMENT,
        anchors=[
            Anchor(phrase="Unproven/Experimental Treatment means"),
            Anchor(phrase="treatments not approved by Indian Medical council"),
        ],
        applies=lambda ctx: ctx.case.treatment.experimental is True,
        evaluate=_eval_experimental,
        unresolved_statement=(
            "The experimental-treatment provisions were not retrieved; an experimental "
            "treatment claim cannot be safely decided."
        ),
    )
)


# --------------------------------------------------------------------------- #
# 6. Exclusion 20 disease list — scope ambiguity handled explicitly
# --------------------------------------------------------------------------- #


def _eval_listed_disease(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_LISTED_DISEASE
    term = _matched_listed_term(ctx, res, LISTED_DISEASE_TERMS)
    if term is None:
        return None

    if ctx.facts.get("is_domiciliary"):
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            f"Domiciliary treatment for '{term}' falls within the policy's list of excluded "
            "diseases under domiciliary hospitalisation.",
            res, anchor_phrase="Treatment of following diseases",
        )

    # For in-patient treatment the clause's scope cannot be established from the
    # supplied document, so the system abstains rather than rejecting.
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        f"The claim describes '{term}', which appears in the policy's excluded-disease list, "
        "but in the supplied wording that list is rendered under the domiciliary "
        "hospitalisation heading. Its applicability to in-patient treatment cannot be "
        "established from the document and requires underwriter review.",
        res,
        detail=(
            "Structural ambiguity in the source document: numbered items 17-20 appear as a "
            "flat list but read as sub-items of 'Any expense under Domiciliary Hospitalisation for'."
        ),
        anchor_phrase="Treatment of following diseases",
    )


RULE_LISTED_DISEASE = register(
    PolicyRule(
        rule_id="exclusion_listed_diseases",
        dimension=DecisionDimension.EXCLUSIONS,
        anchors=[Anchor(phrase="Treatment of following diseases")],
        applies=lambda ctx: _claim_mentions(ctx, LISTED_DISEASE_TERMS),
        evaluate=_eval_listed_disease,
        unresolved_severity=Severity.ABSTAIN,
        unresolved_statement=(
            "The claim names a condition on the policy's excluded-disease list but that "
            "clause was not retrieved; the claim cannot be safely decided."
        ),
    )
)


# --------------------------------------------------------------------------- #
# 7. Out-patient treatment
# --------------------------------------------------------------------------- #


def _eval_outpatient(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_OUTPATIENT
    return _finding(
        rule, FindingStatus.VIOLATED, Severity.BLOCKING,
        "The claim is for out-patient treatment, which the policy excludes.",
        res, anchor_phrase="treatment as an outpatient in a Hospital",
    )


RULE_OUTPATIENT = register(
    PolicyRule(
        rule_id="exclusion_outpatient",
        dimension=DecisionDimension.SCOPE_OF_COVER,
        anchors=[Anchor(phrase="treatment as an outpatient in a Hospital")],
        applies=lambda ctx: (ctx.case.treatment.type or "").lower() in {"outpatient", "opd"},
        evaluate=_eval_outpatient,
    )
)


# --------------------------------------------------------------------------- #
# 8. Hospital definition
# --------------------------------------------------------------------------- #


def _eval_hospital_definition(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_HOSPITAL_DEF
    ec = ctx.case.evidence_context
    registered = ec.hospital_registered
    criteria = ec.hospital_minimum_criteria_documented
    network = ctx.case.hospital.network_provider

    if registered is True or criteria is True:
        basis = (
            "the facility is evidenced as registered with the local authorities"
            if registered is True
            else "the facility is documented as meeting the policy's minimum criteria"
        )
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            f"The facility meets the policy definition of a Hospital: {basis}.",
            res, anchor_phrase="Hospital means any institution established for in-patient care",
        )

    if network is True:
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            "The facility is a Network Provider. The policy defines Network Provider as "
            "hospitals or health care providers enlisted by the insurer or TPA, which "
            "evidences the facility's standing as a Hospital under the policy.",
            res,
            detail=(
                "Registration evidence was not supplied directly; network enlistment is relied "
                "upon. Underwriters may still require the registration certificate."
            ),
            anchor_phrase="Network Provider means",
        )

    explicit_unknown = "hospital_registered" in ctx.facts.get("evidence_context_keys", [])
    ctx.add_missing(
        "hospital registration or minimum-criteria documentation",
        rule.dimension,
        "The policy pays only for treatment in an institution meeting its definition of a "
        "Hospital: registration under the Clinical Establishments Act, or documented "
        "compliance with the minimum criteria for nursing staff, in-patient beds, medical "
        "practitioners, operating theatre and daily patient records.",
        blocking=True,
        document="hospital_registration_certificate",
    )
    detail = (
        "The facility is not a network provider and neither registration nor minimum-criteria "
        "compliance has been evidenced."
    )
    if explicit_unknown:
        detail += " The case explicitly records this status as unestablished."
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        "It cannot be established that the facility meets the policy definition of a Hospital, "
        "which is a condition precedent to any hospitalisation benefit.",
        res, detail=detail,
        anchor_phrase="Hospital means any institution established for in-patient care",
    )


RULE_HOSPITAL_DEF = register(
    PolicyRule(
        rule_id="hospital_definition",
        dimension=DecisionDimension.HOSPITAL_DEFINITION,
        anchors=[
            Anchor(phrase="Hospital means any institution established for in-patient care"),
            Anchor(phrase="Network Provider means"),
        ],
        applies=lambda ctx: (ctx.case.treatment.type or "").lower()
        in {"inpatient", "in-patient", "day_care", "daycare"},
        evaluate=_eval_hospital_definition,
        unresolved_statement=(
            "The Hospital definition was not retrieved; the facility's eligibility cannot be confirmed."
        ),
    )
)


# --------------------------------------------------------------------------- #
# 9. Minimum 24-hour hospitalisation / day-care qualification
# --------------------------------------------------------------------------- #


def _eval_min_duration(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_MIN_DURATION
    hours = ctx.case.treatment.admission_hours
    if hours is None:
        ctx.add_missing(
            "admission duration in hours",
            rule.dimension,
            "The policy requires a minimum 24-hour admission unless a day-care exception applies.",
            blocking=True,
            document="discharge_summary",
        )
        return _finding(
            rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
            "Admission duration is unknown, so the policy's minimum 24-hour hospitalisation "
            "requirement cannot be evaluated.",
            res,
        )

    if hours >= 24:
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            f"The admission of {hours:.0f} hours satisfies the policy's minimum 24-hour "
            "hospitalisation requirement.",
            res, anchor_phrase="admission in a Hospital for a minimum period of 24",
        )

    # Under 24 hours: look for a day-care route.
    nb4 = next(
        (i for i in ctx.evidence if contains_phrase(i.text, "The minimum stay of 24 hours can be waived")
         or contains_phrase(i.text, "140 Day Care Procedures")),
        None,
    )
    if nb4 is not None:
        listed = next(
            (p for p in NB4_PROCEDURES
             if contains_phrase(ctx.condition_text, p) and contains_phrase(nb4.text, p)),
            None,
        )
        if listed:
            enriched = AnchorResolution(
                resolved=True, items=[nb4, *res.items][:2], missing_anchors=[]
            )
            return _finding(
                rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
                f"Although the admission lasted {hours:.0f} hours, the policy expressly covers "
                f"'{listed}' as a sub-24-hour day-care procedure.",
                enriched, anchor_phrase="140 Day Care Procedures",
            )
        if ctx.case.evidence_context.technological_advancement_certified is True:
            enriched = AnchorResolution(resolved=True, items=[nb4], missing_anchors=[])
            return _finding(
                rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
                f"The admission of {hours:.0f} hours qualifies: the policy waives the 24-hour "
                "minimum where technological advances have reduced the required stay, and this "
                "has been certified.",
                enriched, anchor_phrase="The minimum stay of 24 hours can be waived",
            )

    ctx.add_missing(
        "day-care qualification evidence",
        rule.dimension,
        "Admission was under 24 hours. The policy covers such treatment only where the "
        "procedure is among the listed day-care procedures, requires specialised hospital "
        "infrastructure, or has been shortened by technological advances.",
        blocking=True,
        document="doctor_certificate",
    )
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        f"The admission lasted {hours:.0f} hours, below the policy's 24-hour minimum, and it "
        "has not been established that a day-care exception applies.",
        res, anchor_phrase="admission in a Hospital for a minimum period of 24",
    )


RULE_MIN_DURATION = register(
    PolicyRule(
        rule_id="hospitalization_minimum_duration",
        dimension=DecisionDimension.HOSPITALIZATION_DEFINITION,
        anchors=[Anchor(phrase="admission in a Hospital for a minimum period of 24")],
        applies=lambda ctx: (ctx.case.treatment.type or "").lower()
        in {"inpatient", "in-patient", "day_care", "daycare"},
        evaluate=_eval_min_duration,
    )
)


# --------------------------------------------------------------------------- #
# 10. Domiciliary treatment conditions
# --------------------------------------------------------------------------- #


def _eval_domiciliary(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_DOMICILIARY
    t = ctx.case.treatment
    cannot_move = t.patient_cannot_be_moved
    no_room = t.hospital_room_unavailable

    if cannot_move is True or no_room is True:
        basis = (
            "the patient's condition does not permit removal to a hospital"
            if cannot_move is True
            else "no hospital room was available"
        )
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            f"The domiciliary treatment satisfies the policy's conditions: {basis}.",
            res, anchor_phrase="Domiciliary Treatment means medical treatment",
        )

    ctx.add_missing(
        "domiciliary qualifying circumstance",
        rule.dimension,
        "Domiciliary treatment is covered only where the patient cannot be moved to a "
        "hospital or no hospital room was available.",
        blocking=True,
        document="doctor_certificate",
    )
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        "Neither of the policy's two qualifying circumstances for domiciliary treatment has "
        "been established.",
        res, anchor_phrase="Domiciliary Treatment means medical treatment",
    )


RULE_DOMICILIARY = register(
    PolicyRule(
        rule_id="domiciliary_conditions",
        dimension=DecisionDimension.DOMICILIARY,
        anchors=[Anchor(phrase="Domiciliary Treatment means medical treatment")],
        applies=lambda ctx: (ctx.case.treatment.type or "").lower()
        in {"domiciliary", "domiciliary_hospitalization", "home"},
        evaluate=_eval_domiciliary,
    )
)


# --------------------------------------------------------------------------- #
# 11. Medical necessity
# --------------------------------------------------------------------------- #


def _eval_medical_necessity(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_MEDICAL_NECESSITY
    confirmed = ctx.case.evidence_context.medical_necessity_confirmed
    if confirmed is True:
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            "Medical necessity is confirmed, satisfying the policy's Medically Necessary requirement.",
            res, anchor_phrase="Medically Necessary means any treatment",
        )
    if confirmed is False:
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.BLOCKING,
            "Medical necessity has been assessed and not confirmed; the policy pays only for "
            "medically necessary treatment.",
            res, anchor_phrase="Medically Necessary means any treatment",
        )

    ctx.add_missing(
        "confirmation of medical necessity",
        rule.dimension,
        "The policy covers expenses only where the treatment is Medically Necessary: required "
        "for medical management, not exceeding the necessary level of care, prescribed by a "
        "Medical Practitioner and conforming to accepted professional standards.",
        blocking=True,
        document="treating_doctor_certificate",
    )
    return _finding(
        rule, FindingStatus.UNRESOLVED, Severity.ABSTAIN,
        "Medical necessity has been raised as an open question on this case and is not "
        "confirmed by the supplied evidence.",
        res, anchor_phrase="Medically Necessary means any treatment",
    )


RULE_MEDICAL_NECESSITY = register(
    PolicyRule(
        rule_id="medical_necessity",
        dimension=DecisionDimension.MEDICAL_NECESSITY,
        anchors=[Anchor(phrase="Medically Necessary means any treatment")],
        # Only evaluated when the case itself raises the question — an absent
        # field means "not in issue", an explicit null means "could not establish".
        applies=lambda ctx: "medical_necessity_confirmed" in ctx.facts.get("evidence_context_keys", []),
        evaluate=_eval_medical_necessity,
    )
)


# --------------------------------------------------------------------------- #
# 12. Pre- and post-hospitalisation windows
# --------------------------------------------------------------------------- #


def _eval_pre_post(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_PRE_POST
    timing = ctx.case.expense_timing
    pre = ctx.case.expenses_inr.pre_hospitalization
    post = ctx.case.expenses_inr.post_hospitalization

    if timing is None:
        ctx.add_missing(
            "dates of pre- and post-hospitalisation expenses",
            rule.dimension,
            "Pre-hospitalisation expenses are reimbursed for a maximum of 30 days before "
            "admission and post-hospitalisation expenses for a maximum of 60 days after "
            "discharge, and must relate to the same condition.",
            blocking=False,
            document="pre_post_expense_records",
        )
        return _finding(
            rule, FindingStatus.LIMIT_APPLIES, Severity.LIMITING,
            f"Pre-hospitalisation (INR {pre:,.0f}) and post-hospitalisation (INR {post:,.0f}) "
            "expenses are payable only within 30 days before admission and 60 days after "
            "discharge; the supplied case does not evidence these dates.",
            res, anchor_phrase="Pre-Hospitalisation up to a maximum of 30 days",
        )

    breaches: list[str] = []
    pre_days = timing.pre_hospitalization_days_before_admission
    post_days = timing.post_hospitalization_days_after_discharge
    if pre_days is not None and pre_days > 30:
        breaches.append(f"pre-hospitalisation expenses span {pre_days} days (limit 30)")
    if post_days is not None and post_days > 60:
        breaches.append(f"post-hospitalisation expenses span {post_days} days (limit 60)")

    if timing.same_condition_confirmed is False:
        breaches.append("the expenses are not confirmed to relate to the hospitalised condition")

    if breaches:
        return _finding(
            rule, FindingStatus.VIOLATED, Severity.LIMITING,
            "Part of the claim falls outside the policy's pre/post-hospitalisation windows: "
            + "; ".join(breaches)
            + ".",
            res, anchor_phrase="Pre-Hospitalisation up to a maximum of 30 days",
        )

    return _finding(
        rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
        f"Pre-hospitalisation expenses ({pre_days} days before admission) and "
        f"post-hospitalisation expenses ({post_days} days after discharge) fall within the "
        "policy's 30-day and 60-day windows for the same condition.",
        res, anchor_phrase="Pre-Hospitalisation up to a maximum of 30 days",
    )


RULE_PRE_POST = register(
    PolicyRule(
        rule_id="pre_post_hospitalization_window",
        dimension=DecisionDimension.PRE_POST_HOSPITALIZATION,
        anchors=[Anchor(phrase="Pre-Hospitalisation up to a maximum of 30 days")],
        applies=lambda ctx: (
            ctx.case.expenses_inr.pre_hospitalization > 0
            or ctx.case.expenses_inr.post_hospitalization > 0
        ),
        evaluate=_eval_pre_post,
        unresolved_severity=Severity.INFORMATIONAL,
    )
)


def _eval_domiciliary_pre_post(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_DOM_PRE_POST
    return _finding(
        rule, FindingStatus.VIOLATED, Severity.LIMITING,
        "Pre- and post-hospitalisation expenses are not payable under domiciliary "
        "hospitalisation, so that portion of the claim is excluded.",
        res, anchor_phrase="Any expense under Domiciliary Hospitalisation",
    )


RULE_DOM_PRE_POST = register(
    PolicyRule(
        rule_id="domiciliary_pre_post_excluded",
        dimension=DecisionDimension.DOMICILIARY,
        anchors=[Anchor(phrase="Any expense under Domiciliary Hospitalisation")],
        applies=lambda ctx: bool(ctx.facts.get("is_domiciliary"))
        and (
            ctx.case.expenses_inr.pre_hospitalization > 0
            or ctx.case.expenses_inr.post_hospitalization > 0
        ),
        evaluate=_eval_domiciliary_pre_post,
        unresolved_severity=Severity.INFORMATIONAL,
    )
)


# --------------------------------------------------------------------------- #
# 13. Claim documentation
# --------------------------------------------------------------------------- #

_CORE_DOCUMENTS = {
    "claim_form": "the completed claim form",
    "itemized_bill": "the itemised hospital bill",
}
_TREATMENT_RECORD_DOCS = {
    "discharge_summary", "procedure_record", "medical_records", "doctor_certificate",
}


def _eval_documentation(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_DOCUMENTATION
    supplied = {d.lower() for d in ctx.case.documents}
    missing_labels: list[str] = []

    for doc, label in _CORE_DOCUMENTS.items():
        if doc not in supplied:
            missing_labels.append(label)
            ctx.add_missing(
                label,
                rule.dimension,
                "The policy requires original or attested bills, receipts, certificates and "
                "evidence from the treating practitioner, hospital, chemist or laboratory.",
                blocking=False,
                document=doc,
            )

    if not (supplied & _TREATMENT_RECORD_DOCS):
        missing_labels.append("a discharge summary or equivalent treatment record")
        ctx.add_missing(
            "discharge summary or treatment record",
            rule.dimension,
            "Evidence of the treatment actually rendered is required to substantiate the claim.",
            blocking=False,
            document="discharge_summary",
        )

    if not missing_labels:
        return _finding(
            rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
            "The core claim documentation required by the policy has been supplied.",
            res, anchor_phrase="original/attested photocopies of all bills",
        )

    return _finding(
        rule, FindingStatus.LIMIT_APPLIES, Severity.LIMITING,
        "Claim documentation is incomplete: " + ", ".join(missing_labels) + " not supplied.",
        res, anchor_phrase="original/attested photocopies of all bills",
    )


RULE_DOCUMENTATION = register(
    PolicyRule(
        rule_id="claim_documentation",
        dimension=DecisionDimension.CLAIM_DOCUMENTATION,
        anchors=[Anchor(phrase="original/attested photocopies of all bills")],
        applies=lambda ctx: True,
        evaluate=_eval_documentation,
        unresolved_severity=Severity.INFORMATIONAL,
    )
)


# --------------------------------------------------------------------------- #
# 14. Scope of cover — the positive limb
# --------------------------------------------------------------------------- #


def _eval_scope(ctx: RuleContext, res: AnchorResolution) -> Finding | None:
    rule = RULE_SCOPE
    diagnosis = ctx.case.treatment.diagnosis or "the diagnosed condition"
    return _finding(
        rule, FindingStatus.SATISFIED, Severity.INFORMATIONAL,
        f"Hospitalisation expenses reasonably and necessarily incurred for {diagnosis} on the "
        "advice of a Medical Practitioner fall within the policy's scope of cover, subject to "
        "its limits and exclusions.",
        res, anchor_phrase="We will pay Reasonable and Customary charges",
    )


RULE_SCOPE = register(
    PolicyRule(
        rule_id="scope_of_cover",
        dimension=DecisionDimension.SCOPE_OF_COVER,
        anchors=[Anchor(phrase="We will pay Reasonable and Customary charges")],
        applies=lambda ctx: (ctx.case.treatment.type or "").lower()
        not in {"outpatient", "opd"},
        evaluate=_eval_scope,
        unresolved_statement=(
            "The scope-of-cover clause was not retrieved; coverage cannot be affirmed."
        ),
    )
)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def required_anchors_for_case(
    case, facts: dict[str, Any]
) -> list[tuple[DecisionDimension, str]]:
    """Anchor phrases the rules that will run on this case depend upon.

    This closes the loop between the rule engine and retrieval: rather than
    hoping a broad topical sweep happens to surface every governing clause, the
    rules declare the evidence they need and the Case Analysis Agent turns each
    declaration into its own targeted query.

    It exists because a broad query cannot rank a short, precise clause. The
    exclusions sweep covers ten different exclusion types, so "7. Dental
    treatment or surgery of any kind." matched only a fraction of the query
    terms and ranked fifth — below the per-dimension reservation — even though
    it disposed of the claim outright. A query built from the anchor itself
    matches it almost exactly. See docs/failure-analysis.md F-06.

    Applicability is probed against an empty evidence list, which is safe
    because ``applies`` predicates read only the case and its derived facts.
    """
    probe = RuleContext(case=case, facts=facts, evidence=[])
    pairs: list[tuple[DecisionDimension, str]] = []
    seen: set[str] = set()

    for rule in REGISTRY:
        try:
            if not rule.applies(probe):
                continue
        except Exception:  # pragma: no cover - defensive
            continue
        for anchor in rule.anchors:
            if anchor.phrase not in seen:
                seen.add(anchor.phrase)
                pairs.append((rule.dimension, anchor.phrase))
    return pairs


def evaluate_rules(ctx: RuleContext) -> list[Finding]:
    """Run every applicable rule, binding each to retrieved evidence."""
    findings: list[Finding] = []

    for rule in REGISTRY:
        try:
            if not rule.applies(ctx):
                continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rule %s applicability check failed: %s", rule.rule_id, exc)
            continue

        resolution = resolve_anchors(
            ctx.evidence, rule.anchors, require_all=rule.require_all_anchors
        )

        if not resolution.resolved:
            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    dimension=rule.dimension,
                    status=FindingStatus.UNRESOLVED,
                    severity=rule.unresolved_severity,
                    statement=rule.unresolved_statement,
                    detail=(
                        "Anchor text not present in retrieved evidence: "
                        + "; ".join(resolution.missing_anchors)
                    ),
                    evidence_chunk_ids=[],
                    citations=[],
                    evidence_supported=False,
                )
            )
            continue

        try:
            finding = rule.evaluate(ctx, resolution)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Rule %s failed to evaluate", rule.rule_id)
            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    dimension=rule.dimension,
                    status=FindingStatus.UNRESOLVED,
                    severity=Severity.ABSTAIN,
                    statement=f"Rule '{rule.rule_id}' could not be evaluated ({type(exc).__name__}).",
                    evidence_chunk_ids=resolution.chunk_ids,
                    citations=[],
                    evidence_supported=False,
                )
            )
            continue

        if finding is not None:
            findings.append(finding)

    return findings
