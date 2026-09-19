"""Sparse lexical retrieval (Okapi BM25).

Implemented directly rather than pulled from ``rank_bm25`` for two reasons that
matter in this domain:

1. The tokenizer must keep percentages, clause numbers and hyphenation
   ("1.0%", "48", "months", "pre-existing", "24"). Those tokens *are* the
   decisive evidence in this policy; a generic tokenizer discards or mangles
   them.
2. We need per-term score attribution to explain *why* a chunk matched, which
   the library does not expose.

The scoring function is textbook Okapi BM25 with the standard k1/b defaults.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.utils.text import tokenize


@dataclass
class BM25Result:
    index: int
    score: float
    matched_terms: dict[str, float] = field(default_factory=dict)


class BM25Index:
    """Okapi BM25 over a fixed corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._avg_len: float = 0.0
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    # ------------------------------------------------------------------ #

    def fit(self, corpus: list[str]) -> "BM25Index":
        self._docs = [tokenize(doc) for doc in corpus]
        self._doc_freqs = [Counter(tokens) for tokens in self._docs]
        self._doc_len = [len(tokens) for tokens in self._docs]
        self._n_docs = len(self._docs)
        self._avg_len = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

        df: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            df.update(freqs.keys())

        # Robertson/Sparck-Jones IDF with the +1 guard that keeps common terms
        # from going negative on small corpora (this policy is only ~140 chunks).
        self._idf = {
            term: math.log(1.0 + (self._n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        return self

    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        if self._n_docs == 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        results: list[BM25Result] = []
        for idx in range(self._n_docs):
            freqs = self._doc_freqs[idx]
            dl = self._doc_len[idx]
            score = 0.0
            matched: dict[str, float] = {}
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self._avg_len or 1.0)))
                contribution = idf * (tf * (self.k1 + 1)) / denom
                score += contribution
                matched[term] = matched.get(term, 0.0) + contribution
            if score > 0:
                results.append(BM25Result(index=idx, score=score, matched_terms=matched))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------ #

    def idf(self, term: str) -> float:
        """Exposed so the reranker can weight rare terms more heavily."""
        return self._idf.get(term, 0.0)

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def state_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "docs": self._docs,
            "doc_len": self._doc_len,
            "avg_len": self._avg_len,
            "idf": self._idf,
            "n_docs": self._n_docs,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BM25Index":
        index = cls(k1=state["k1"], b=state["b"])
        index._docs = state["docs"]
        index._doc_freqs = [Counter(tokens) for tokens in state["docs"]]
        index._doc_len = state["doc_len"]
        index._avg_len = state["avg_len"]
        index._idf = state["idf"]
        index._n_docs = state["n_docs"]
        return index
