"""Persistent memory: episodic events, cross-session lessons, and profiles.

The loop depends on the :class:`~.base.MemoryStore` ABC and on nothing else
here, so the backend is a config value. :func:`~.factory.build_memory` decides
which concrete stack to assemble and reports why in its notes.
"""

from .base import MemoryKind, MemoryRecord, MemoryStore, NullMemory
from .embedding import Embedder, EmbeddingUnavailable, FastEmbedEmbedder, NullEmbedder
from .factory import build_memory, memory_stats
from .manager import MemoryManager, wrap
from .sqlite_store import SQLiteMemory

__all__ = [
    "Embedder",
    "EmbeddingUnavailable",
    "FastEmbedEmbedder",
    "MemoryKind",
    "MemoryManager",
    "MemoryRecord",
    "MemoryStore",
    "NullEmbedder",
    "NullMemory",
    "SQLiteMemory",
    "build_memory",
    "memory_stats",
    "wrap",
]
