"""Dense vector index.

At ~140 policy chunks an exact cosine scan over a contiguous ``float32`` matrix
is a single BLAS call — faster than any ANN structure once index build time is
counted, and exact by construction. FAISS is used when installed (so the same
code scales if the corpus grows), otherwise numpy serves as the backend.

This is a deliberate "do not add unnecessary frameworks" choice, documented as
a trade-off in the README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DenseResult:
    index: int
    score: float


class DenseIndex:
    """Cosine-similarity search over L2-normalised embeddings."""

    def __init__(self, use_faiss: bool | None = None) -> None:
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._use_faiss = use_faiss

    @staticmethod
    def _faiss_available() -> bool:
        try:
            import faiss  # noqa: F401

            return True
        except Exception:
            return False

    def build(self, embeddings: np.ndarray) -> "DenseIndex":
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2-D embedding matrix, got shape {embeddings.shape}")
        self._matrix = np.ascontiguousarray(embeddings.astype(np.float32))

        want_faiss = self._use_faiss if self._use_faiss is not None else self._faiss_available()
        if want_faiss and self._faiss_available():
            import faiss

            # Inner product on normalised vectors == cosine similarity.
            index = faiss.IndexFlatIP(self._matrix.shape[1])
            index.add(self._matrix)
            self._faiss_index = index
            logger.info("Dense index: FAISS IndexFlatIP with %d vectors", self._matrix.shape[0])
        else:
            self._faiss_index = None
            logger.info("Dense index: numpy exact search with %d vectors", self._matrix.shape[0])
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[DenseResult]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = np.ascontiguousarray(query_vector.astype(np.float32).reshape(1, -1))
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q /= norm

        k = min(top_k, self._matrix.shape[0])
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k)
            return [
                DenseResult(index=int(i), score=float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]

        sims = self._matrix @ q[0]
        top = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
        top = top[np.argsort(-sims[top])]
        return [DenseResult(index=int(i), score=float(sims[i])) for i in top]

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"
