"""Detect a loop that is running but no longer getting anywhere.

The failure mode this defends against is the expensive one, because it does not
look like a failure. Milestone 1 produced a live example: a registry bug made
*every* tool call fail, and the loop ran all six iterations, failed identically
each time, and returned cleanly. Nothing raised. The error containment that kept
the run alive is exactly what hid the problem.

Three independent signals, because the same pathology shows up in three
different places depending on where it starts:

``repeated_action``
    The same ``(action, arguments)`` signature ``repeat_action_threshold`` times
    **consecutively**. Consecutive matters: an agent that alternates
    ``score -> revise -> score -> revise`` calls ``score_against_rubric()`` with
    identical (empty) arguments many times in a healthy run, and a total count
    would flag it. A run of three identical calls back to back is the agent
    failing to notice that nothing changed.

``score_plateau``
    The last ``stuck_score_window`` scorecards span less than
    ``stuck_score_epsilon``. This normally fires *after* the Reflect plateau
    rule has already stopped the run -- it is the backstop for the case where
    scoring keeps happening but the number has frozen and Reflect's own
    ``min_improvement`` threshold is configured looser than epsilon.

``draft_cycle``
    The draft returns to a state it already held. Not "the draft did not
    change" -- that is normal on every scoring turn -- but A -> B -> A, which
    means two revisions are undoing each other. Compared on a normalised hash,
    so whitespace and case churn do not disguise it.

Detection produces ``RunStatus.STUCK`` and the run returns its best draft. Being
stuck is not an error: the agent did real work and the harness is declining to
pay for more of it.
"""

from __future__ import annotations

import hashlib
import typing as t
from dataclasses import dataclass, field

from ..config import GuardrailsConfig
from ..core.state import IterationRecord, LoopState


def draft_fingerprint(text: str) -> str:
    """Whitespace- and case-insensitive identity for a draft."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StuckVerdict:
    """Why the detector believes the loop is spinning."""

    signal: str
    detail: str

    def render(self) -> str:
        return f"stuck ({self.signal}): {self.detail}"


@dataclass
class StuckDetector:
    """Stateful across one run. Fed one :class:`IterationRecord` at a time."""

    repeat_threshold: int = 3
    score_epsilon: float = 0.5
    score_window: int = 3

    _last_signature: str = field(default="", init=False)
    _repeat_run: int = field(default=0, init=False)
    #: Distinct draft states in the order they were first seen.
    _draft_states: list[str] = field(default_factory=list, init=False)

    @classmethod
    def from_config(cls, config: GuardrailsConfig) -> StuckDetector:
        return cls(
            repeat_threshold=config.repeat_action_threshold,
            score_epsilon=config.stuck_score_epsilon,
            score_window=config.stuck_score_window,
        )

    def observe(self, state: LoopState, record: IterationRecord) -> StuckVerdict | None:
        """Fold in one iteration and report the first signal that fires."""
        return (
            self._repeated_action(record)
            or self._draft_cycle(state)
            or self._score_plateau(state)
        )

    # -- signals ------------------------------------------------------------

    def _repeated_action(self, record: IterationRecord) -> StuckVerdict | None:
        signature = record.decision.signature()
        self._repeat_run = self._repeat_run + 1 if signature == self._last_signature else 1
        self._last_signature = signature

        if self._repeat_run < self.repeat_threshold:
            return None
        return StuckVerdict(
            signal="repeated_action",
            detail=(
                f"{signature} was chosen {self._repeat_run} times in a row; "
                "the agent is not reacting to its own results"
            ),
        )

    def _draft_cycle(self, state: LoopState) -> StuckVerdict | None:
        current = draft_fingerprint(state.workspace.draft)
        if not self._draft_states:
            self._draft_states.append(current)
            return None
        if current == self._draft_states[-1]:
            return None  # unchanged is normal: scoring and analysis edit nothing

        if current in self._draft_states:
            position = self._draft_states.index(current)
            self._draft_states.append(current)
            return StuckVerdict(
                signal="draft_cycle",
                detail=(
                    f"the draft returned to the state it held at revision {position}; "
                    "successive edits are undoing each other"
                ),
            )
        self._draft_states.append(current)
        return None

    def _score_plateau(self, state: LoopState) -> StuckVerdict | None:
        history: t.Sequence[float] = state.score_history
        if len(history) < self.score_window:
            return None
        window = list(history[-self.score_window :])
        spread = max(window) - min(window)
        if spread >= self.score_epsilon:
            return None
        return StuckVerdict(
            signal="score_plateau",
            detail=(
                f"the last {self.score_window} scores span {spread:.2f} points "
                f"(< {self.score_epsilon}); further iterations are not buying anything"
            ),
        )


__all__ = ["StuckDetector", "StuckVerdict", "draft_fingerprint"]
