"""Structure-aware chunking of the policy wording.

Why not fixed-size windows
--------------------------
This policy is a legal instrument whose meaning lives in *atomic units*: a
single definition ("Hospital means ..."), a single numbered exclusion, a single
sub-limit note. A 512-character sliding window slices those units in half, so a
retriever returns the top of exclusion 5 and the bottom of exclusion 4 and the
reasoning layer has to guess which clause it is looking at.

The chunker below therefore segments by document structure first and only
applies size control *within* a unit:

1. Lines are tagged with the page they came from, so provenance survives.
2. Top-level sections are detected from the wording's own headings.
3. Each section is segmented by the unit type it actually uses --
   definitions, numbered exclusions, numbered benefits and NB notes,
   lettered claims-procedure blocks, numbered standard conditions.
4. Units longer than the size budget are split on sentence boundaries with
   overlap; units are never merged across a clause boundary, because merging
   two exclusions into one chunk would make a citation ambiguous.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import ChunkingSettings, get_settings
from app.models.schemas import PolicyChunk
from app.retrieval.ingestion import PolicyPage
from app.utils.text import collapse, estimate_tokens, normalise

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #

SECTION_PREAMBLE = "Preamble"
SECTION_DEFINITIONS = "Definitions"
SECTION_CRITICAL_ILLNESS = "Critical Illness Definitions"
SECTION_SCOPE = "Scope of Cover"
SECTION_EXCLUSIONS = "What We Exclude"
SECTION_EXTENSIONS = "Extensions"
SECTION_CLAIMS = "Claims Procedure"
SECTION_TERMS = "Standard Terms and Conditions"
SECTION_GRIEVANCE = "Grievance and Ombudsman"

# Ordered: the first pattern that matches a line wins.
_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*DEFINITIONS\s*$", re.I), SECTION_DEFINITIONS),
    (re.compile(r"^\s*SCOPE\s+OF\s+COVER\s*$", re.I), SECTION_SCOPE),
    (re.compile(r"^\s*WHAT\s+WE\s+COVER\s*$", re.I), SECTION_SCOPE),
    (re.compile(r"^\s*WHAT\s+WE\s+EXCLUDE\s*$", re.I), SECTION_EXCLUSIONS),
    (re.compile(r"^\s*EXTENSIONS\s*$", re.I), SECTION_EXTENSIONS),
    (re.compile(r"^\s*CLAIMS\s+PROCEDURE\s*$", re.I), SECTION_CLAIMS),
    (re.compile(r"^\s*STANDARD\s+TERMS\s+AND\s+CONDITIONS\s*:?\s*$", re.I), SECTION_TERMS),
    (re.compile(r"^\s*Critical\s+Illness\s*$"), SECTION_CRITICAL_ILLNESS),
    (re.compile(r"^\s*\d+\.\s*Grievances\s*$", re.I), SECTION_GRIEVANCE),
]

# "Term means ..." / "Term refers to ..." — the shape of every definition here.
_DEFINITION_RE = re.compile(
    r"^(?P<term>[A-Z][A-Za-z0-9''\-/&\. ]{1,64}?)\s+"
    r"(?P<verb>means\b|refers to\b|is essentially\b|occurs\b|is one in which\b|shall mean\b|is a person\b|is the process\b)"
)
# A bare title line that introduces a definition on the following line.
_DEF_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9''\-/& ]{2,60}$")

_NUMBERED_RE = re.compile(r"^(?P<num>\d{1,2})\.\s+(?P<rest>\S.*)$")
_NB_RE = re.compile(r"^(?P<tag>NB\s?\d?)\s*[:.]\s*(?P<rest>.*)$", re.I)
_LETTER_BLOCK_RE = re.compile(r"^\((?P<letter>[A-Z])\)\s*(?P<rest>.*)$")
_CI_ITEM_RE = re.compile(r"^(?P<num>\d{1,2})\.\s+(?P<rest>[A-Z].*)$")
_SUBLIMIT_RE = re.compile(r"^Sub\s*limits?\s*$", re.I)
_NOTE_RE = re.compile(r"^Note\s*$", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class _Line:
    text: str
    page: int


@dataclass
class _Unit:
    """A semantically atomic block of policy text."""

    section: str
    heading: str
    clause_ref: str | None
    lines: list[_Line]

    @property
    def text(self) -> str:
        return normalise("\n".join(l.text for l in self.lines))

    @property
    def page_start(self) -> int:
        return min(l.page for l in self.lines) if self.lines else 1

    @property
    def page_end(self) -> int:
        return max(l.page for l in self.lines) if self.lines else 1


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def _flatten(pages: list[PolicyPage]) -> list[_Line]:
    lines: list[_Line] = []
    for page in pages:
        for raw in page.text.splitlines():
            if raw.strip():
                lines.append(_Line(text=raw.rstrip(), page=page.page))
    return lines


def _detect_section(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    for pattern, name in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def _segment_definitions(lines: list[_Line], section: str) -> list[_Unit]:
    """One unit per defined term."""
    units: list[_Unit] = []
    current: _Unit | None = None

    for idx, line in enumerate(lines):
        stripped = line.text.strip()
        match = _DEFINITION_RE.match(stripped)

        heading: str | None = None
        if match:
            heading = match.group("term").strip(" .")
        elif _DEF_HEADING_RE.match(stripped) and not stripped.endswith((".", ",", ":")):
            # A standalone title line counts only if the next line reads like a body.
            nxt = lines[idx + 1].text.strip() if idx + 1 < len(lines) else ""
            first_word = stripped.split()[0].lower() if stripped.split() else ""
            if nxt and (nxt.lower().startswith(first_word) or _DEFINITION_RE.match(nxt)
                        or nxt.lower().startswith(("it means", "day care treatment refers"))):
                heading = stripped

        if heading and len(heading) >= 3:
            if current is not None:
                units.append(current)
            current = _Unit(section=section, heading=heading, clause_ref=None, lines=[line])
            continue

        if current is None:
            current = _Unit(section=section, heading="Preamble", clause_ref=None, lines=[])
        current.lines.append(line)

    if current is not None:
        units.append(current)
    return [u for u in units if u.lines]


def _segment_numbered(
    lines: list[_Line],
    section: str,
    *,
    prefix: str,
    heading_from_text: bool = True,
    extra_starts: tuple[re.Pattern[str], ...] = (),
) -> list[_Unit]:
    """One unit per numbered clause, plus any auxiliary block starters."""
    units: list[_Unit] = []
    current: _Unit | None = None

    for line in lines:
        stripped = line.text.strip()
        started = False

        m = _NUMBERED_RE.match(stripped)
        if m:
            num = m.group("num")
            rest = m.group("rest").strip()
            label = collapse(rest)
            if heading_from_text:
                label = label[:72].rstrip(" ,;:")
            heading = f"{prefix} {num}: {label}" if label else f"{prefix} {num}"
            if current is not None:
                units.append(current)
            current = _Unit(
                section=section,
                heading=heading,
                clause_ref=f"{prefix} {num}",
                lines=[line],
            )
            started = True

        if not started:
            for pattern in extra_starts:
                em = pattern.match(stripped)
                if em:
                    tag = collapse(stripped)[:72]
                    if current is not None:
                        units.append(current)
                    current = _Unit(
                        section=section, heading=tag, clause_ref=tag.split(":")[0].strip(),
                        lines=[line],
                    )
                    started = True
                    break

        if started:
            continue

        if current is None:
            current = _Unit(section=section, heading=f"{section} (introduction)",
                            clause_ref=None, lines=[])
        current.lines.append(line)

    if current is not None:
        units.append(current)
    return [u for u in units if u.lines]


def _segment_generic(lines: list[_Line], section: str) -> list[_Unit]:
    """Blank-line-delimited paragraphs, used where there is no clause numbering."""
    units: list[_Unit] = []
    buf: list[_Line] = []
    for line in lines:
        buf.append(line)
    if buf:
        units.append(_Unit(section=section, heading=section, clause_ref=None, lines=buf))
    return units


def _segment_section(section: str, lines: list[_Line]) -> list[_Unit]:
    if not lines:
        return []
    if section == SECTION_DEFINITIONS:
        return _segment_definitions(lines, section)
    if section == SECTION_CRITICAL_ILLNESS:
        return _segment_numbered(lines, section, prefix="Critical Illness")
    if section == SECTION_SCOPE:
        return _segment_numbered(
            lines,
            section,
            prefix="Cover",
            extra_starts=(_NB_RE, _SUBLIMIT_RE, _NOTE_RE),
        )
    if section == SECTION_EXCLUSIONS:
        return _segment_numbered(lines, section, prefix="Exclusion")
    if section == SECTION_TERMS:
        return _segment_numbered(lines, section, prefix="Condition")
    if section == SECTION_CLAIMS:
        return _segment_numbered(
            lines, section, prefix="Claims", extra_starts=(_LETTER_BLOCK_RE,)
        )
    if section == SECTION_EXTENSIONS:
        return _segment_numbered(lines, section, prefix="Extension")
    return _segment_generic(lines, section)


# --------------------------------------------------------------------------- #
# Size control
# --------------------------------------------------------------------------- #


def _split_oversized(unit: _Unit, cfg: ChunkingSettings) -> list[tuple[str, int, int, int]]:
    """Split one unit into (text, page_start, page_end, part_index) pieces."""
    text = unit.text
    if len(text) <= cfg.max_chunk_chars:
        return [(text, unit.page_start, unit.page_end, 0)]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    parts: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) > cfg.max_chunk_chars and buf:
            parts.append(buf)
            # Carry an overlap tail so a clause split mid-condition stays readable.
            tail = buf[-cfg.overlap_chars :]
            buf = f"{tail} {sentence}".strip()
        else:
            buf = candidate
    if buf:
        parts.append(buf)

    # A pathological unit with no sentence boundaries still has to be cut.
    if len(parts) == 1 and len(parts[0]) > cfg.max_chunk_chars:
        raw = parts[0]
        step = cfg.max_chunk_chars - cfg.overlap_chars
        parts = [raw[i : i + cfg.max_chunk_chars] for i in range(0, len(raw), step)]

    return [(p, unit.page_start, unit.page_end, i) for i, p in enumerate(parts)]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def chunk_policy(
    pages: list[PolicyPage],
    *,
    source_name: str,
    settings: ChunkingSettings | None = None,
) -> list[PolicyChunk]:
    """Turn ingested pages into retrievable, fully-attributed policy chunks."""
    cfg = settings or get_settings().chunking
    lines = _flatten(pages)

    # Pass 1: assign every line to a section.
    buckets: list[tuple[str, list[_Line]]] = []
    current_section = SECTION_PREAMBLE
    current_lines: list[_Line] = []
    for line in lines:
        detected = _detect_section(line.text)
        if detected:
            if current_lines:
                buckets.append((current_section, current_lines))
            current_section = detected
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        buckets.append((current_section, current_lines))

    # Pass 2: segment each section by its own unit type.
    units: list[_Unit] = []
    for section, section_lines in buckets:
        units.extend(_segment_section(section, section_lines))

    # Pass 3: size control and id assignment.
    chunks: list[PolicyChunk] = []
    ordinal = 0
    for unit in units:
        if len(collapse(unit.text)) < 40:  # strip stray fragments
            continue
        for text, page_start, page_end, part in _split_oversized(unit, cfg):
            clean = collapse(text)
            if len(clean) < 40:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", unit.heading.lower()).strip("-")[:48] or "block"
            suffix = f"-p{part}" if part else ""
            chunk_id = f"p{page_start:02d}-{slug}-{ordinal:03d}{suffix}"
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    text=clean,
                    page=page_start,
                    page_end=page_end if page_end != page_start else None,
                    section=unit.section,
                    heading=unit.heading,
                    clause_ref=unit.clause_ref,
                    source=source_name,
                    char_count=len(clean),
                    token_estimate=estimate_tokens(clean),
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    logger.info(
        "Chunked policy into %d chunks across %d sections",
        len(chunks),
        len({c.section for c in chunks}),
    )
    return chunks


def embedding_text(chunk: PolicyChunk) -> str:
    """Contextualised text used for indexing only.

    Prefixing the section and heading lets a chunk carry its own context, which
    measurably helps both arms retrieve short definition chunks.
    The stored ``text`` stays pristine so citations quote the policy verbatim.
    """
    head = f"{chunk.section} — {chunk.heading}"
    if chunk.clause_ref and chunk.clause_ref not in chunk.heading:
        head = f"{head} ({chunk.clause_ref})"
    return f"{head}\n{chunk.text}"
