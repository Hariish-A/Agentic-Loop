"""Memory policy: what gets recalled, from where, and what happens when it breaks.

The store answers "what matches this query". The manager answers the harder
question: **what should this agent be reminded of right now**. Those are
different, and keeping them apart means the retrieval maths can change without
touching the policy, and vice versa.

Three tiers, three lifetimes
----------------------------

===========  ===================  ==========================================
tier         scope                why
===========  ===================  ==========================================
``episodic`` this session only    "I already tried adding examples and it
                                  moved evidence by 0.2" is about *this* run
``lesson``   this rubric, any     the Reflexion payload -- durable findings
             session              that transfer to a different text
``profile``  this rubric          standing constraints (tone, target, banned
                                  edits)
===========  ===================  ==========================================

Lessons are **rubric-scoped, not global**. A finding about attributing
statistics in essays is not evidence about bug reports, and recalling it there
is worse than recalling nothing -- it spends prompt budget arguing for an
irrelevant edit.

Why lessons are not relevance-gated
-----------------------------------

Episodic records are voluminous and noisy, so they must clear
``recall_min_score``. Lessons are the opposite: a handful of curated findings
per rubric, each one already filtered by Reflect deciding it was worth keeping.
Gating them on cosine similarity throws away the whole point of Reflexion the
first time a query happens to be phrased differently. They are ranked, capped by
``max_lessons_per_recall``, and always offered. That is a deliberate
precision/recall trade made in favour of recall, and it is the single most
consequential policy choice in this module.
"""

from __future__ import annotations

import typing as t

from ..config import MemoryConfig
from ..core.state import MemoryHit
from .base import MemoryRecord, MemoryStore, NullMemory
from .sqlite_store import SQLiteMemory

#: Consecutive failures before the manager stops calling the store at all.
CIRCUIT_BREAKER_THRESHOLD = 3


class MemoryManager(MemoryStore):
    """Applies scope policy over a store, and never lets it break the run.

    Also the circuit breaker required by Milestone 3's "memory read failure"
    fallback: after :data:`CIRCUIT_BREAKER_THRESHOLD` consecutive failures the
    manager stops calling the store, records the reason, and behaves like
    :class:`~.base.NullMemory` for the rest of the process. Retrying a store
    that has failed three times in a row costs latency on every iteration and
    almost never succeeds.
    """

    def __init__(self, store: MemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config
        self.failures = 0
        self.degraded = False
        self.degraded_reason = ""
        self.notes: list[str] = list(getattr(store, "notes", []))

    # -- failure containment ------------------------------------------------

    def _guard(self, operation: str, fn: t.Callable[[], t.Any], default: t.Any) -> t.Any:
        if self.degraded:
            return default
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - the loop must survive this
            self.failures += 1
            reason = f"{operation} failed ({type(exc).__name__}: {exc})"
            self.notes.append(reason)
            if self.failures >= CIRCUIT_BREAKER_THRESHOLD:
                self.degraded = True
                self.degraded_reason = (
                    f"memory disabled after {self.failures} consecutive failures; last: {reason}"
                )
                self.notes.append(self.degraded_reason)
            return default
        self.failures = 0
        return result

    # -- write --------------------------------------------------------------

    def save(self, record: MemoryRecord) -> str:
        if not self.config.enabled:
            return ""
        return t.cast(str, self._guard("memory save", lambda: self.store.save(record), ""))

    # -- read ---------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        rubric_id: str | None = None,
        kinds: t.Sequence[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        """Recall under the tier policy described in the module docstring."""
        if not self.config.enabled:
            return []
        return t.cast(
            "list[MemoryHit]",
            self._guard(
                "memory recall",
                lambda: self._recall(query, session_id, rubric_id, kinds, limit, min_score),
                [],
            ),
        )

    def _recall(
        self,
        query: str,
        session_id: str | None,
        rubric_id: str | None,
        kinds: t.Sequence[str] | None,
        limit: int,
        min_score: float,
    ) -> list[MemoryHit]:
        wanted = set(kinds) if kinds else {"lesson", "profile", "episodic"}
        hits: list[MemoryHit] = []

        # 1. Profile first: standing constraints outrank situational advice.
        if "profile" in wanted:
            hits.extend(
                self._search("profile", query, rubric_id=rubric_id, limit=2, gate=0.0)
            )

        # 2. Lessons: cross-session, rubric-scoped, ranked but ungated.
        if "lesson" in wanted:
            lesson_gate = min_score if self.config.gate_lessons else 0.0
            hits.extend(
                self._search(
                    "lesson",
                    query,
                    session_id=None if self.config.lesson_scope != "session" else session_id,
                    rubric_id=rubric_id if self.config.lesson_scope == "rubric" else None,
                    limit=self.config.max_lessons_per_recall,
                    gate=lesson_gate,
                )
            )

        # 3. Episodic fills whatever budget is left, session-scoped and gated.
        if "episodic" in wanted:
            remaining = max(0, limit - len(hits))
            if remaining:
                hits.extend(
                    self._search(
                        "episodic",
                        query,
                        session_id=session_id if self.config.episodic_scope == "session" else None,
                        limit=remaining,
                        gate=min_score,
                    )
                )

        return self._dedupe(hits)[:limit]

    def _search(
        self,
        kind: str,
        query: str,
        *,
        session_id: str | None = None,
        rubric_id: str | None = None,
        limit: int,
        gate: float,
    ) -> list[MemoryHit]:
        if limit <= 0:
            return []
        if isinstance(self.store, SQLiteMemory):
            results = self.store.search(
                query, kinds=[kind], session_id=session_id, rubric_id=rubric_id, limit=limit
            )
            # Keyword-only mode produces ordinal, uncalibrated scores; gating on
            # them would silently return nothing. See sqlite_store's docstring.
            effective_gate = gate if self.store.vector_enabled else 0.0
            return [
                SQLiteMemory._to_hit(item) for item in results if item.score >= effective_gate
            ]
        return self.store.recall(
            query,
            session_id=session_id,
            rubric_id=rubric_id,
            kinds=[kind],
            limit=limit,
            min_score=gate,
        )

    @staticmethod
    def _dedupe(hits: t.Sequence[MemoryHit]) -> list[MemoryHit]:
        seen: set[str] = set()
        unique: list[MemoryHit] = []
        for hit in hits:
            key = " ".join(hit.content.lower().split())
            if key in seen:
                continue
            seen.add(key)
            unique.append(hit)
        return unique

    # -- maintenance --------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        return t.cast(
            int,
            self._guard("memory clear_session", lambda: self.store.clear_session(session_id), 0),
        )

    def list_sessions(self) -> list[str]:
        return t.cast(
            "list[str]", self._guard("memory list_sessions", self.store.list_sessions, [])
        )

    def stats(self) -> dict[str, t.Any]:
        base = t.cast("dict[str, t.Any]", self._guard("memory stats", self.store.stats, {}))
        base.update(
            {
                "degraded": self.degraded,
                "degraded_reason": self.degraded_reason,
                "consecutive_failures": self.failures,
                "policy": {
                    "lesson_scope": self.config.lesson_scope,
                    "episodic_scope": self.config.episodic_scope,
                    "gate_lessons": self.config.gate_lessons,
                    "recall_top_k": self.config.recall_top_k,
                    "recall_min_score": self.config.recall_min_score,
                },
            }
        )
        return base

    def close(self) -> None:
        self.store.close()

    @property
    def describe(self) -> str:
        if self.degraded:
            return f"{self.store.describe} [DEGRADED: {self.degraded_reason}]"
        return self.store.describe


def wrap(store: MemoryStore | None, config: MemoryConfig) -> MemoryStore:
    """Wrap a store in the policy layer, or return a no-op when disabled."""
    if store is None or not config.enabled:
        return NullMemory()
    return MemoryManager(store, config)


__all__ = ["CIRCUIT_BREAKER_THRESHOLD", "MemoryManager", "wrap"]
