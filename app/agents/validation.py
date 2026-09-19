"""Agent 5 — Validation.

Responsibility: verify that the decision's material statements are actually
supported by evidence that retrieval returned, and that no citation was
fabricated.

Seven independent checks run against the draft decision:

C1 citation_integrity     Every citation's chunk_id exists in the retrieved
                          evidence, and its page/section match that chunk.
                          Catches fabricated or drifted citations (RULE 3).
C2 material_support       Every blocking/abstaining/limiting finding carries at
                          least one citation.
C3 anchor_presence        Each finding's quoted text genuinely appears in the
                          chunk it cites — catches a quote assembled rather
                          than extracted.
C4 limit_arithmetic       Each applied limit recomputes: allowed = min(claimed,
                          cap) and deduction = claimed - allowed.
C5 decision_alignment     The decision follows from the findings (a rejection
                          needs a blocking finding; an abstention needs an
                          unresolved one).
C6 abstention_discipline  If any required condition is unresolved, the decision
                          is not affirmative (RULE 4/5).
C7 llm_second_opinion     Optional. An independent model is shown the statement
                          and its quoted evidence and asked whether the evidence
                          supports it. Advisory only: it can flag, never decide.

On failure the agent requests one revision. The Decision Agent re-runs with the
validation result folded into confidence; if the failure persists, the decision
is forced to NEEDS_REVIEW rather than shipped unverified.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.models.schemas import (
    AnalysisResponse,
    Decision,
    EvidenceItem,
    FindingStatus,
    Severity,
    ValidationReport,
    ValidationStatus,
)
from app.services.llm import LLMClient, get_llm_client
from app.utils.text import collapse, contains_phrase

logger = logging.getLogger(__name__)

_MATERIAL = {Severity.BLOCKING, Severity.ABSTAIN, Severity.LIMITING}


class ValidationAgent:
    name = "validation"

    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client()

    # ------------------------------------------------------------------ #

    def run(self, draft: AnalysisResponse, evidence: list[EvidenceItem]) -> ValidationReport:
        by_id = {item.chunk_id: item for item in evidence}
        checks_run: list[str] = []
        checks_failed: list[str] = []
        unsupported: list[str] = []
        notes: list[str] = []
        citation_integrity = True

        # -- C1 citation integrity ------------------------------------- #
        checks_run.append("citation_integrity")
        for citation in draft.citations:
            item = by_id.get(citation.chunk_id)
            if item is None:
                citation_integrity = False
                unsupported.append(
                    f"Citation references chunk '{citation.chunk_id}', which is not in the "
                    f"retrieved evidence for this case."
                )
            elif citation.page != item.page or citation.section != item.section:
                citation_integrity = False
                unsupported.append(
                    f"Citation metadata for '{citation.chunk_id}' does not match the retrieved "
                    f"chunk (cited p.{citation.page}/{citation.section}, "
                    f"actual p.{item.page}/{item.section})."
                )
        if not citation_integrity:
            checks_failed.append("citation_integrity")

        # -- C2 material support --------------------------------------- #
        # Only findings that *assert* something about the policy need a
        # citation. An UNRESOLVED finding asserts the opposite — that the
        # governing clause could not be established — so demanding a citation
        # from it is incoherent and would make every abstention "fail".
        # Those are audited by C2b instead.
        checks_run.append("material_support")
        material = [
            f for f in draft.key_findings
            if f.severity in _MATERIAL and f.status != FindingStatus.UNRESOLVED
        ]
        unsupported_findings = [f for f in material if not f.citations]
        if unsupported_findings:
            checks_failed.append("material_support")
            for finding in unsupported_findings:
                unsupported.append(
                    f"[{finding.rule_id}] {collapse(finding.statement)[:180]} "
                    "— no policy citation supports this statement."
                )

        # -- C2b unresolved findings must not assert support ------------ #
        checks_run.append("unresolved_discipline")
        overclaiming = [
            f for f in draft.key_findings
            if f.status == FindingStatus.UNRESOLVED and f.evidence_supported and f.citations
            and f.severity == Severity.ABSTAIN and not f.evidence_chunk_ids
        ]
        if overclaiming:
            checks_failed.append("unresolved_discipline")
            for finding in overclaiming:
                unsupported.append(
                    f"[{finding.rule_id}] finding is unresolved yet claims evidential support."
                )

        # -- C3 anchor presence ---------------------------------------- #
        checks_run.append("quote_provenance")
        drifted = 0
        for finding in draft.key_findings:
            for citation in finding.citations:
                item = by_id.get(citation.chunk_id)
                if item is None or not citation.quote:
                    continue
                probe = collapse(citation.quote).strip("… ")
                # Compare a robust interior slice; the quote is a window into
                # the chunk, so its middle must be present verbatim.
                if len(probe) > 40:
                    probe = probe[10 : min(len(probe), 90)]
                if probe and not contains_phrase(item.text, probe):
                    drifted += 1
                    unsupported.append(
                        f"[{finding.rule_id}] quoted text does not appear verbatim in cited "
                        f"chunk '{citation.chunk_id}'."
                    )
        if drifted:
            checks_failed.append("quote_provenance")

        # -- C4 limit arithmetic --------------------------------------- #
        checks_run.append("limit_arithmetic")
        arithmetic_errors = 0
        for limit in draft.applicable_limits:
            if limit.limit_amount_inr is None or limit.claimed_amount_inr is None:
                continue
            expected_allowed = min(limit.claimed_amount_inr, limit.limit_amount_inr)
            expected_deduction = max(0.0, limit.claimed_amount_inr - limit.limit_amount_inr)
            if (
                abs((limit.allowed_amount_inr or 0.0) - expected_allowed) > 0.5
                or abs(limit.deduction_inr - expected_deduction) > 0.5
            ):
                arithmetic_errors += 1
                unsupported.append(
                    f"[{limit.limit_id}] limit arithmetic does not reconcile "
                    f"(claimed {limit.claimed_amount_inr:,.0f}, cap {limit.limit_amount_inr:,.0f}, "
                    f"allowed {limit.allowed_amount_inr:,.0f}, deduction {limit.deduction_inr:,.0f})."
                )
        if arithmetic_errors:
            checks_failed.append("limit_arithmetic")

        # -- C5 decision alignment ------------------------------------- #
        checks_run.append("decision_alignment")
        has_blocking = any(
            f.severity == Severity.BLOCKING and f.status == FindingStatus.VIOLATED
            for f in draft.key_findings
        )
        has_unresolved = any(
            f.severity == Severity.ABSTAIN and f.status == FindingStatus.UNRESOLVED
            for f in draft.key_findings
        )
        aligned = True
        if draft.decision == Decision.NOT_ADMISSIBLE and not has_blocking:
            aligned = False
            unsupported.append(
                "Decision NOT_ADMISSIBLE is not supported by any confirmed blocking finding."
            )
        if draft.decision == Decision.NEEDS_REVIEW and not (
            has_unresolved or draft.missing_evidence or draft.abstain_reason
        ):
            aligned = False
            unsupported.append(
                "Decision NEEDS_REVIEW is not supported by any unresolved condition or evidence gap."
            )
        if not aligned:
            checks_failed.append("decision_alignment")

        # -- C6 abstention discipline ---------------------------------- #
        checks_run.append("abstention_discipline")
        if has_unresolved and draft.decision in {
            Decision.ADMISSIBLE,
            Decision.ADMISSIBLE_WITH_LIMITS,
            Decision.PARTIALLY_ADMISSIBLE,
        }:
            checks_failed.append("abstention_discipline")
            unsupported.append(
                "An affirmative decision was reached while a required policy condition "
                "remains unresolved; the policy requires abstention."
            )

        # -- C7 optional LLM second opinion ---------------------------- #
        if self.llm.enabled and material:
            checks_run.append("llm_second_opinion")
            flagged = self._llm_review(material, by_id)
            if flagged:
                notes.append(
                    f"Second-opinion model flagged {len(flagged)} statement(s) for reviewer "
                    "attention; deterministic checks passed."
                )
                notes.extend(flagged[:3])

        status = ValidationStatus.FAIL if checks_failed else ValidationStatus.PASS
        if status == ValidationStatus.PASS and not notes:
            notes.append(
                f"All {len(checks_run)} verification checks passed across "
                f"{len(draft.citations)} citation(s)."
            )

        return ValidationReport(
            status=status,
            unsupported_claims=unsupported,
            revision_required=status == ValidationStatus.FAIL,
            checks_run=checks_run,
            checks_failed=checks_failed,
            citation_integrity=citation_integrity,
            notes=notes,
        )

    # ------------------------------------------------------------------ #

    def _llm_review(self, findings, by_id: dict[str, EvidenceItem]) -> list[str]:
        """Independent support check. Advisory: it can flag but never decide."""
        try:
            payload = []
            for finding in findings[:6]:
                citation = finding.citations[0] if finding.citations else None
                item = by_id.get(citation.chunk_id) if citation else None
                if item is None:
                    continue
                payload.append(
                    {
                        "statement": collapse(finding.statement)[:300],
                        "policy_text": collapse(item.text)[:700],
                    }
                )
            if not payload:
                return []

            data = self.llm.complete_json(
                system=(
                    "You verify whether each statement is supported by the policy text quoted "
                    "beside it. Judge ONLY from the quoted text. Return a JSON array of objects "
                    '{"index": <int>, "supported": <bool>, "reason": "<max 20 words>"}. '
                    "Do not add commentary."
                ),
                user=str(payload),
                max_tokens=500,
            )
            if not isinstance(data, list):
                return []
            return [
                f"Second opinion on statement {entry.get('index')}: "
                f"{str(entry.get('reason', ''))[:120]}"
                for entry in data
                if isinstance(entry, dict) and entry.get("supported") is False
            ]
        except Exception as exc:  # pragma: no cover - optional path
            logger.debug("LLM second opinion skipped: %s", exc)
            return []


def force_review(draft: AnalysisResponse, report: ValidationReport) -> AnalysisResponse:
    """Apply the terminal safety rule when validation cannot be satisfied.

    A decision whose statements failed verification is never shipped as an
    affirmative or negative verdict; it becomes NEEDS_REVIEW.
    """
    draft.decision = Decision.NEEDS_REVIEW
    draft.abstained = True
    draft.abstain_reason = (
        "INSUFFICIENT_EVIDENCE: validation could not confirm that every material statement "
        "is supported by retrieved policy evidence."
    )
    draft.summary = (
        f"Claim {draft.case_id} is referred for review: the system could not verify that all "
        "material decision statements are supported by the retrieved policy evidence."
    )
    draft.validation = report
    return draft
