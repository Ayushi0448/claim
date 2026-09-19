"""Policy PDF ingestion.

Produces one cleaned text block per page while preserving the 1-based page
number, because every citation the system emits must be traceable back to a
physical page of the supplied policy (assignment §4.1 / §6).

Running headers and footers are detected empirically rather than hard-coded by
line index: a line that repeats on most pages is boilerplate, not content.
Leaving them in would pollute BM25 with the insurer's name on every chunk.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.utils.text import normalise

logger = logging.getLogger(__name__)

# A line appearing on at least this fraction of pages is treated as boilerplate.
_BOILERPLATE_PAGE_RATIO = 0.6
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")


@dataclass(frozen=True)
class PolicyPage:
    page: int
    text: str
    raw_text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class PolicyIngestionError(RuntimeError):
    """Raised when the policy document cannot be read at all."""


def _extract_pages_pdfplumber(pdf_path: Path) -> list[str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # layout=True preserves the indentation that signals list nesting in
            # the exclusions section, which the chunker relies on.
            try:
                text = page.extract_text(layout=True) or ""
            except Exception:  # pragma: no cover - per-page resilience
                text = page.extract_text() or ""
            pages.append(text)
    return pages


def _extract_pages_pypdf(pdf_path: Path) -> list[str]:  # pragma: no cover - fallback
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [(p.extract_text() or "") for p in reader.pages]


def _strip_boilerplate(page_texts: list[str]) -> list[str]:
    """Remove running headers/footers and bare page numbers."""
    n_pages = len(page_texts)
    if n_pages == 0:
        return []

    line_pages: Counter[str] = Counter()
    for text in page_texts:
        seen: set[str] = set()
        for line in text.splitlines():
            key = " ".join(line.split()).lower()
            if len(key) < 8:
                continue
            if key not in seen:
                seen.add(key)
                line_pages[key] += 1

    threshold = max(2, int(n_pages * _BOILERPLATE_PAGE_RATIO))
    boilerplate = {k for k, c in line_pages.items() if c >= threshold}
    if boilerplate:
        logger.debug("Detected %d boilerplate lines", len(boilerplate))

    cleaned: list[str] = []
    for text in page_texts:
        kept: list[str] = []
        for line in text.splitlines():
            key = " ".join(line.split()).lower()
            if key in boilerplate:
                continue
            if _PAGE_NUM_RE.match(line):
                continue
            # The footer carries the page number glued to the wording title.
            if "policy wording" in key and "unihlip" in key:
                continue
            kept.append(line.rstrip())
        cleaned.append("\n".join(kept))
    return cleaned


def load_policy_pages(pdf_path: str | Path) -> list[PolicyPage]:
    """Read the policy PDF into cleaned, page-numbered blocks."""
    path = Path(pdf_path)
    if not path.exists():
        raise PolicyIngestionError(
            f"Policy PDF not found at {path}. Set POLICY_PDF_PATH or place the "
            "supplied policy under data/policy/."
        )

    try:
        raw_pages = _extract_pages_pdfplumber(path)
        engine = "pdfplumber"
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("pdfplumber failed (%s); falling back to pypdf", exc)
        try:
            raw_pages = _extract_pages_pypdf(path)
            engine = "pypdf"
        except Exception as exc2:
            raise PolicyIngestionError(f"Unable to extract text from {path}: {exc2}") from exc2

    if not any(t.strip() for t in raw_pages):
        raise PolicyIngestionError(
            f"{path} produced no extractable text. A scanned policy would need OCR."
        )

    cleaned = _strip_boilerplate(raw_pages)
    logger.info("Ingested %d pages from %s using %s", len(cleaned), path.name, engine)

    return [
        PolicyPage(page=i + 1, text=normalise(cleaned[i]), raw_text=raw_pages[i])
        for i in range(len(cleaned))
    ]


def policy_fingerprint(pdf_path: str | Path) -> str:
    """Content hash used to invalidate a stale index."""
    path = Path(pdf_path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]
