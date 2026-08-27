"""Memory interface and a no-op implementation.

Defined in Milestone 1 so the loop can call ``recall`` and ``save`` from the
start, with :class:`NullMemory` standing in. Milestone 2 supplies a real
SQLite-backed store behind the same three operations and nothing in the loop
changes -- which is also what makes the "memory read failure" fallback in
Milestone 3 trivial: swap the store for a NullMemory and keep running.

Three record kinds, with different lifetimes:

``episodic``
    What happened in one iteration of one session. Scoped to that session.
``lesson``
    A transferable finding produced by Reflect. Recalled across sessions --
    this is the Reflexion memory.
``profile``
    Durable constraints for a rubric or user (tone, target, banned edits).
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.state import MemoryHit, utc_now

MemoryKind = t.Literal["episodic", "lesson", "profile"]


@dataclass
class MemoryRecord:
    """One thing worth remembering."""

    kind: MemoryKind
    content: str
    session_id: str = ""
    iteration: int = 0
    rubric_id: str = ""
    criterion_id: str = ""
    score_delta: float | None = None
    metadata: dict[str, t.Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.content = self.content.strip()


class MemoryStore(ABC):
    """The three operations the challenge requires, plus two for operability."""

    @abstractmethod
    def save(self, record: MemoryRecord) -> str:
        """Persist a record and return its id."""

    @abstractmethod
    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        kinds: t.Sequence[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        """Return the most relevant records for ``query``, best first."""

    @abstractmethod
    def clear_session(self, session_id: str) -> int:
        """Delete every record for one session. Returns the number removed."""

    def list_sessions(self) -> list[str]:
        return []

    def stats(self) -> dict[str, t.Any]:
        return {}

    def close(self) -> None:  # noqa: B027 - optional hook; NullMemory holds nothing
        """Release resources. Safe to call more than once."""

    @property
    def describe(self) -> str:
        return type(self).__name__


class NullMemory(MemoryStore):
    """Remembers nothing, fails at nothing.

    Used as the Milestone 1 default, by ``--no-memory`` for A/B demonstrations,
    and as the circuit-breaker target when the real store becomes unreadable.
    """

    def save(self, record: MemoryRecord) -> str:
        return ""

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        kinds: t.Sequence[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        return []

    def clear_session(self, session_id: str) -> int:
        return 0

    @property
    def describe(self) -> str:
        return "NullMemory (disabled)"


__all__ = ["MemoryKind", "MemoryRecord", "MemoryStore", "NullMemory"]
