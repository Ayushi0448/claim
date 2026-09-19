"""Pluggable dense embedding backends.

Design note (trade-off, documented in README §8 and docs/architecture.md)
------------------------------------------------------------------------
The preferred backend is a BGE sentence-transformer. It is *not* always
available: free-tier hosts and locked-down CI runners frequently cannot reach
the model hub, and a 400 MB download is a poor fit for a 512 MB dyno.

Rather than let the whole system fail in that situation, the dense arm is an
interface with two implementations:

* ``SentenceTransformerBackend`` — BGE / MiniLM via sentence-transformers.
* ``LsaBackend`` — TF-IDF over word *and* character n-grams reduced with
  truncated SVD (latent semantic analysis).

Both emit L2-normalised dense vectors scored by cosine similarity, so the rest
of the pipeline is unchanged. LSA is genuinely dense and genuinely semantic —
SVD places "domiciliary" near "confined at home" through co-occurrence — but it
is corpus-fitted rather than pre-trained, so it generalises less well to
paraphrases the policy never uses. That weakness is precisely what the lexical
BM25 arm and the reranker compensate for, which is the point of hybrid search.

``EMBEDDING_BACKEND=auto`` (default) probes for the transformer and falls back
silently. Set it explicitly to pin a backend in production.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import ModelSettings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def fit(self, corpus: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> bool: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LsaBackend:
    """Corpus-fitted latent semantic embeddings. No network, no model download."""

    name = "lsa"

    def __init__(self, dim: int = 192) -> None:
        self.dim = dim
        self._word_vec = None
        self._char_vec = None
        self._svd = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not corpus:
            raise ValueError("Cannot fit LSA backend on an empty corpus")

        # Word n-grams capture clause phrasing; char n-grams absorb the
        # British/American spelling split that runs through this document
        # ("hospitalisation" vs "hospitalization") and OCR noise.
        self._word_vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.92,
            strip_accents="unicode",
        )
        self._char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )
        word = self._word_vec.fit_transform(corpus)
        char = self._char_vec.fit_transform(corpus)

        from scipy.sparse import hstack

        combined = hstack([word, char]).tocsr()

        n_components = int(min(self.dim, max(2, min(combined.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(combined)
        self.dim = n_components
        self._fitted = True
        explained = float(self._svd.explained_variance_ratio_.sum())
        logger.info(
            "LSA backend fitted: %d docs -> %d dims (%.1f%% variance retained)",
            len(corpus), n_components, explained * 100,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LsaBackend.encode() called before fit()/load()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        from scipy.sparse import hstack

        combined = hstack(
            [self._word_vec.transform(texts), self._char_vec.transform(texts)]
        ).tocsr()
        return _l2_normalise(self._svd.transform(combined).astype(np.float32))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "word_vec": self._word_vec,
                    "char_vec": self._char_vec,
                    "svd": self._svd,
                    "dim": self.dim,
                },
                fh,
            )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with path.open("rb") as fh:
                state = pickle.load(fh)
            self._word_vec = state["word_vec"]
            self._char_vec = state["char_vec"]
            self._svd = state["svd"]
            self.dim = state["dim"]
            self._fitted = True
            return True
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("Could not load LSA backend from %s: %s", path, exc)
            return False


class SentenceTransformerBackend:
    """BGE / MiniLM embeddings when the weights are actually reachable."""

    name = "sentence_transformer"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: list[str]) -> None:  # pre-trained: nothing to fit
        self._ensure()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def save(self, path: Path) -> None:  # weights live in the HF cache
        return None

    def load(self, path: Path) -> bool:
        try:
            self._ensure()
            return True
        except Exception:
            return False


def _transformer_available(model_name: str) -> bool:
    """Probe without raising: import, then attempt an actual load."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        logger.info("sentence-transformers not installed; using LSA dense backend")
        return False
    try:
        backend = SentenceTransformerBackend(model_name)
        backend._ensure()
        return True
    except Exception as exc:
        logger.info(
            "Embedding model %r unavailable (%s); using LSA dense backend",
            model_name, type(exc).__name__,
        )
        return False


def get_embedding_backend(settings: ModelSettings | None = None) -> EmbeddingBackend:
    cfg = settings or get_settings().models
    choice = cfg.embedding_backend

    if choice in {"sentence_transformer", "st", "transformer"}:
        # Probe even when pinned. Returning an unloadable backend does not fail
        # loudly: retrieval raises per-claim, every claim degrades to
        # NEEDS_REVIEW at zero confidence, and the operator sees no cause. A
        # logged fallback keeps the service answering with the offline backend.
        if _transformer_available(cfg.embedding_model):
            return SentenceTransformerBackend(cfg.embedding_model)
        logger.error(
            "EMBEDDING_BACKEND=%s was pinned but %r could not be loaded; "
            "falling back to LSA. Install sentence-transformers or set "
            "EMBEDDING_BACKEND=lsa to silence this.",
            choice, cfg.embedding_model,
        )
        return LsaBackend(dim=cfg.embedding_dim)
    if choice == "lsa":
        return LsaBackend(dim=cfg.embedding_dim)

    # auto
    if _transformer_available(cfg.embedding_model):
        logger.info("Dense backend: sentence-transformer %s", cfg.embedding_model)
        return SentenceTransformerBackend(cfg.embedding_model)
    return LsaBackend(dim=cfg.embedding_dim)
