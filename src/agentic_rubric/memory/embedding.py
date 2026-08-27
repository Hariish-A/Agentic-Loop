"""Text embedding, with a fallback that is a feature rather than an apology.

The default embedder is `fastembed` running BAAI/bge-small-en-v1.5 through ONNX
Runtime: ~130 MB of model, 384 dimensions, CPU-only, **no PyTorch**. Choosing it
over `sentence-transformers` keeps the Docker image around 400 MB instead of
2.5 GB for embeddings of comparable quality at this scale.

Everything here is designed around one fact: **the embedder can be unavailable**
-- not installed, model download blocked, no disk. When that happens the memory
layer degrades to SQLite FTS5 keyword recall and the loop keeps running. That
path is exercised by tests, not hoped for, because it doubles as the Milestone 3
"memory read failure" fallback.

Loading is lazy. A run configured for keyword-only recall must not pay a
model-load cost it will never use.
"""

from __future__ import annotations

import os
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

#: Known output dimensions, so the vector table can be created before the model
#: is loaded. Anything not listed is probed on first use.
KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class EmbeddingUnavailable(RuntimeError):
    """The configured embedder cannot be used. Never fatal: callers degrade."""


class Embedder(ABC):
    """Turns text into vectors."""

    name: str = "unknown"

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether :meth:`embed` can be expected to work."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width, or 0 when this embedder produces nothing."""

    @abstractmethod
    def embed(self, texts: t.Sequence[str]) -> list[list[float]]:
        """Embed a batch. Raises :class:`EmbeddingUnavailable` on failure."""

    def embed_one(self, text: str) -> list[float] | None:
        """Convenience wrapper that swallows unavailability."""
        if not self.available:
            return None
        try:
            vectors = self.embed([text])
        except EmbeddingUnavailable:
            return None
        return vectors[0] if vectors else None

    def describe(self) -> str:
        return f"{self.name} ({self.dimension}d)" if self.available else f"{self.name} (disabled)"


class NullEmbedder(Embedder):
    """Produces nothing. Selects keyword-only recall."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return 0

    def embed(self, texts: t.Sequence[str]) -> list[list[float]]:
        raise EmbeddingUnavailable("no embedder is configured")


class FastEmbedEmbedder(Embedder):
    """ONNX embeddings via `fastembed`, loaded on first use."""

    name = "fastembed"

    def __init__(self, model_name: str, *, cache_dir: str | Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model: t.Any = None
        self._dimension = KNOWN_DIMENSIONS.get(model_name, 0)
        self._failure: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def _load(self) -> t.Any:
        if self._model is not None:
            return self._model
        if self._failure is not None:
            raise EmbeddingUnavailable(self._failure)
        # Silence Hugging Face's download chrome before the import that
        # triggers it. Warnings on stderr corrupt the console transcript, which
        # is the thing a reviewer actually watches. Progress bars are left
        # alone: fastembed re-enables them itself and warns if we forbid it,
        # which trades one line of noise for another.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the install
            self._failure = f"fastembed is not installed ({exc})"
            raise EmbeddingUnavailable(self._failure) from exc
        try:
            # First construction downloads the model; a pinned cache_dir keeps
            # that out of a temp directory so Docker can bake it into the image.
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        except Exception as exc:  # noqa: BLE001 - offline, disk full, bad name
            self._failure = f"could not load {self.model_name}: {type(exc).__name__}: {exc}"
            raise EmbeddingUnavailable(self._failure) from exc
        return self._model

    @property
    def available(self) -> bool:
        return self._failure is None

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: t.Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            vectors = [[float(x) for x in vector] for vector in model.embed(list(texts))]
        except Exception as exc:  # noqa: BLE001 - runtime inference failure
            self._failure = f"embedding failed: {type(exc).__name__}: {exc}"
            raise EmbeddingUnavailable(self._failure) from exc
        if vectors and not self._dimension:
            self._dimension = len(vectors[0])
        return vectors

    def warm_up(self) -> str | None:
        """Force the model to load now. Returns a failure reason, or ``None``.

        Called by the factory so an unavailable embedder is discovered at
        startup and reported once, instead of surfacing mid-run as a per-record
        save failure.
        """
        try:
            self.embed(["warm up"])
        except EmbeddingUnavailable as exc:
            return str(exc)
        return None


def build_embedder(
    kind: str, model_name: str, *, cache_dir: str | Path | None = None, warm: bool = True
) -> tuple[Embedder, str]:
    """Construct the configured embedder. Returns ``(embedder, note)``.

    Never raises. An unusable embedder comes back as :class:`NullEmbedder` with
    a human-readable reason, which the caller logs and carries into the run as a
    degradation note.
    """
    normalised = (kind or "none").strip().lower()
    if normalised in ("none", "null", "off", "disabled", ""):
        return NullEmbedder(), "semantic recall disabled by config; using keyword recall"

    if normalised != "fastembed":
        return NullEmbedder(), f"unknown embedder {kind!r}; falling back to keyword recall"

    embedder = FastEmbedEmbedder(model_name, cache_dir=cache_dir)
    if warm:
        failure = embedder.warm_up()
        if failure:
            return NullEmbedder(), f"{failure}; falling back to keyword recall"
    return embedder, f"semantic recall via {embedder.describe()}"


__all__ = [
    "KNOWN_DIMENSIONS",
    "Embedder",
    "EmbeddingUnavailable",
    "FastEmbedEmbedder",
    "NullEmbedder",
    "build_embedder",
]
