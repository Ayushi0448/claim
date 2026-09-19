"""Text normalisation and tokenisation tuned for insurance policy wording."""

from __future__ import annotations

import re
import unicodedata

# The PDF was produced by Acrobat's Paper Capture plug-in and carries smart
# quotes, private-use bullets and ligatures. Normalising them once at ingestion
# keeps every downstream component (BM25, embeddings, anchor matching) honest.
_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
    "": "* ",
    "": "* ",
    "•": "* ",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "„": '"',
}

_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "ours", "shall", "she", "so", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "under", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your", "yours",
}

# Domain synonyms drive query expansion: the claim says "day care" but the
# policy says "less than 24 hrs"; without expansion the lexical arm misses it.
DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "hospitalization": ("hospitalisation", "in-patient care", "admission"),
    "hospitalisation": ("hospitalization", "in-patient care", "admission"),
    "preexisting": ("pre-existing", "pre existing"),
    "pre-existing": ("preexisting", "48 months"),
    "daycare": ("day care", "day care treatment", "less than 24 hrs"),
    "day_care": ("day care treatment", "less than 24 hours", "technological advancement"),
    "domiciliary": ("domiciliary hospitalisation", "treatment at home", "confined at home"),
    "sublimit": ("sub limit", "sub-limit", "limits"),
    "room_rent": ("room boarding and nursing", "normal room expenses", "room rent"),
    "waiting_period": ("waiting period", "30 days", "first year", "48 months"),
    "exclusion": ("what we exclude", "we exclude", "not be covered"),
    "experimental": ("unproven", "experimental treatment", "not approved"),
    "ambulance": ("ambulance charges", "additional benefits"),
    "cosmetic": ("aesthetic treatment", "plastic surgery"),
    "portability": ("continuous coverage", "previous insurer", "portability benefit"),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|\d+(?:\.\d+)?%?")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    """Canonicalise unicode oddities without destroying layout."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def collapse(text: str) -> str:
    """Single-line form used for display and quoting."""
    return " ".join(text.split())


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenisation preserving hyphenation and percentages.

    Percentages and clause numbers matter enormously here ("1.0%", "48 months",
    "24 hours"), so unlike a generic tokenizer we keep digits and ``%``.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whitespace- and case-insensitive phrase containment.

    Used for evidence anchoring, where the anchor must genuinely appear in the
    retrieved chunk before a policy rule is allowed to fire.
    """
    h = " ".join(haystack.lower().split())
    p = " ".join(phrase.lower().split())
    return p in h


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — avoids a tokenizer dependency."""
    return max(1, len(text) // 4)
