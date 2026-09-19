"""Evidence binding — the mechanism that makes citations trustworthy.

This is the load-bearing idea of the whole system.

Every policy rule declares the *anchor text* it depends on: a verbatim fragment
of the clause that gives the rule its authority. A rule is only allowed to fire
if that anchor is found inside a chunk that retrieval actually returned for this
case. Consequences:

* RULE 2 — a rule can never assert a clause the policy does not contain, because
  the anchor would not resolve.
* RULE 3 — citations are emitted *from* the resolved chunk, so page, section and
  chunk_id are copied from real retrieved evidence. There is no code path that
  fabricates a page number.
* RULE 4/5 — if retrieval misses the governing clause, the rule reports
  ``UNRESOLVED`` instead of guessing, which propagates to abstention.

A retrieval regression therefore degrades the system into abstention rather than
into confident wrong answers. That is the correct failure mode for claims
adjudication.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import Citation, EvidenceItem
from app.utils.text import collapse, contains_phrase


@dataclass(frozen=True)
class Anchor:
    """A verbatim fragment of the policy that a rule depends upon."""

    phrase: str
    # Optional narrowing: the anchor must come from this section / heading.
    section: str | None = None
    heading_contains: str | None = None

    def matches(self, item: EvidenceItem) -> bool:
        if self.section and item.section != self.section:
            return False
        if self.heading_contains and not contains_phrase(item.heading, self.heading_contains):
            return False
        return contains_phrase(item.text, self.phrase)


@dataclass
class AnchorResolution:
    """The outcome of trying to ground a rule in retrieved evidence."""

    resolved: bool
    items: list[EvidenceItem]
    missing_anchors: list[str]

    @property
    def chunk_ids(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            if item.chunk_id not in seen:
                seen.append(item.chunk_id)
        return seen

    @property
    def primary(self) -> EvidenceItem | None:
        return self.items[0] if self.items else None


def resolve_anchors(
    evidence: list[EvidenceItem],
    anchors: list[Anchor],
    *,
    require_all: bool = False,
) -> AnchorResolution:
    """Find the retrieved chunks that carry each anchor.

    ``require_all=True`` is used by rules that genuinely need to combine two
    clauses (for example a waiting period *and* its continuity waiver), which is
    the multi-section reasoning case from assignment §10.
    """
    found: list[EvidenceItem] = []
    missing: list[str] = []

    for anchor in anchors:
        hit = next((item for item in evidence if anchor.matches(item)), None)
        if hit is None:
            missing.append(anchor.phrase)
        elif hit not in found:
            found.append(hit)

    resolved = (not missing) if require_all else bool(found)
    return AnchorResolution(resolved=resolved, items=found, missing_anchors=missing)


def quote_for(item: EvidenceItem, anchor_phrase: str, *, window: int = 260) -> str:
    """Extract a short verbatim quote around the anchor for reviewer display.

    The quote is sliced out of the retrieved chunk text itself — never
    reconstructed — so what a reviewer reads is what the policy says.
    """
    text = collapse(item.text)
    needle = collapse(anchor_phrase).lower()
    pos = text.lower().find(needle)
    if pos < 0:
        return text[:window] + ("…" if len(text) > window else "")
    start = max(0, pos - window // 3)
    end = min(len(text), pos + len(needle) + (2 * window) // 3)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def build_citation(
    claim: str,
    item: EvidenceItem,
    *,
    rule_id: str | None = None,
    anchor_phrase: str | None = None,
) -> Citation:
    """Construct a citation from a genuinely retrieved evidence item."""
    return Citation(
        claim=claim,
        source=item.source,
        page=item.page,
        section=item.section,
        heading=item.heading,
        chunk_id=item.chunk_id,
        quote=quote_for(item, anchor_phrase) if anchor_phrase else item.snippet(260),
        rule_id=rule_id,
    )
