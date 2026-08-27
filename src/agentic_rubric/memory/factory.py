"""Build the memory stack from configuration.

`AGENTIC_MEMORY__BACKEND=sqlite_fts` is a complete swap to keyword-only recall;
`--no-memory` is a complete swap to no memory at all. Neither requires touching
the loop, which asks for a :class:`~.base.MemoryStore` and gets one.

Nothing here raises. A misconfigured or unusable backend degrades to a working
store plus a note, because a run that cannot remember is far better than a run
that cannot start.
"""

from __future__ import annotations

import typing as t

from ..config import AppConfig
from .base import MemoryStore, NullMemory
from .embedding import build_embedder
from .manager import MemoryManager
from .sqlite_store import SQLiteMemory

#: Where the ONNX embedding model is cached. Project-local (and gitignored) so
#: a container image can bake it in rather than downloading on first run.
MODEL_CACHE_DIRNAME = "models"

BACKENDS = ("sqlite_vec", "sqlite_fts", "null")


def build_memory(
    config: AppConfig, *, enabled: bool | None = None, warm_embedder: bool = True
) -> tuple[MemoryStore, list[str]]:
    """Construct the configured memory stack. Returns ``(store, notes)``."""
    notes: list[str] = []
    active = config.memory.enabled if enabled is None else enabled

    if not active:
        return NullMemory(), ["memory disabled"]

    backend = (config.memory.backend or "null").strip().lower()
    if backend not in BACKENDS:
        notes.append(f"unknown memory backend {backend!r}; falling back to sqlite_fts")
        backend = "sqlite_fts"

    if backend == "null":
        return NullMemory(), ["memory backend is 'null'; nothing will be remembered"]

    want_vectors = backend == "sqlite_vec"
    if want_vectors:
        embedder, note = build_embedder(
            config.memory.embedder,
            config.memory.embed_model,
            cache_dir=config.path(MODEL_CACHE_DIRNAME),
            warm=warm_embedder,
        )
    else:
        embedder, note = build_embedder("none", "", warm=False)
        note = "backend sqlite_fts: keyword recall only"
    notes.append(note)

    try:
        store: MemoryStore = SQLiteMemory(
            config.path(config.memory.db_path),
            embedder=embedder,
            enable_vector=want_vectors,
            vector_weight=config.memory.vector_weight,
        )
    except Exception as exc:  # noqa: BLE001 - unwritable path, locked file, ...
        notes.append(
            f"could not open the memory database ({type(exc).__name__}: {exc}); "
            "running without memory"
        )
        return NullMemory(), notes

    notes.extend(getattr(store, "notes", []))
    return MemoryManager(store, config.memory), notes


def memory_stats(store: MemoryStore) -> dict[str, t.Any]:
    """Best-effort stats for the CLI, safe on any store."""
    try:
        return store.stats()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = ["BACKENDS", "MODEL_CACHE_DIRNAME", "build_memory", "memory_stats"]
