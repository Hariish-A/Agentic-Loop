"""Limits enforced outside the model, on the things the model can spend.

Each guardrail exists because of a specific way an unsupervised agent burns
money or time:

``max_iterations``
    An agent that can decide it is finished can also decide it is not. The cap
    lives in :class:`~..core.loop.AgenticLoop` itself, in Python -- never
    expressed as a request in a prompt. This class re-checks it so that a
    caller who invokes the loop with a different cap still cannot exceed the
    configured one.

``token_budget``
    A revision loop over a long document is the archetypal runaway cost: every
    iteration re-reads the draft and writes a new one. Checked before each
    iteration *and* after it, because one iteration can spend several thousand
    tokens across Reason, the judge, the reviser and Reflect. A warning fires at
    ``token_warn_ratio`` so an operator sees the trend before the stop.

``wall_clock_timeout_s``
    Tokens do not measure a hung socket, a provider taking 90 seconds per call,
    or an embedder that decided to download a model. Time does.

``max_document_chars``
    Bounds what the process ingests at all, so a pasted 40 MB file cannot make
    metrics, diffing and hashing crawl. Distinct from
    ``guardrails.max_input_chars``, which caps the *prompt view* of the draft
    and is applied by Perceive, where the prompt is built.

Stuck detection is delegated to :mod:`.loop_detect` and surfaced through the
same interface, because from the loop's point of view "you are spinning" and
"you are out of budget" are the same kind of instruction: stop now, keep the
best draft.

Every stop is graceful. The run ends with a status, a reason, the best draft
seen and a complete trace -- never a raised exception. An agent that crashes on
its budget has thrown away the work it already paid for.
"""

from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass

from ..config import AppConfig
from ..core.loop import StopSignal
from ..core.state import IterationRecord, LoopState, RunStatus
from .loop_detect import StuckDetector

Emit = t.Callable[..., None]
Clock = t.Callable[[], float]


def _noop(*_args: t.Any, **_kwargs: t.Any) -> None:
    return None


@dataclass
class InputCheck:
    """The result of applying the hard ingestion cap."""

    text: str
    truncated: bool
    note: str = ""


class Guardrails:
    """The :class:`~..core.loop.LoopController` the harness installs.

    Holds no opinion about *what* the agent should do -- only about when it must
    stop. That narrowness is deliberate: a guardrail that could redirect the
    agent would be a fifth cognitive step hiding in the harness.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        emit: Emit | None = None,
        clock: Clock = time.monotonic,
        detector: StuckDetector | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.limits = config.guardrails
        self.max_iterations = (
            max_iterations if max_iterations is not None else config.loop.max_iterations
        )
        self._emit = emit or _noop
        self._clock = clock
        self._detector = detector or StuckDetector.from_config(config.guardrails)
        self._started = clock()
        self._warned = False
        #: What actually fired, for the summary and the tests.
        self.triggered: list[str] = []

    # -- ingestion ----------------------------------------------------------

    def check_input(self, text: str) -> InputCheck:
        """Apply the hard document cap before the run starts."""
        limit = self.limits.max_document_chars
        if limit <= 0 or len(text) <= limit:
            return InputCheck(text=text, truncated=False)
        note = (
            f"input is {len(text):,} characters, above the {limit:,}-character ingestion "
            "cap; the tail has been dropped before the run"
        )
        self._fire("max_document_chars", note)
        return InputCheck(text=text[:limit], truncated=True, note=note)

    # -- the controller interface ------------------------------------------

    def before_iteration(self, state: LoopState) -> StopSignal | None:
        return self._budget(state) or self._deadline(state) or self._cap(state)

    def after_iteration(self, state: LoopState, record: IterationRecord) -> StopSignal | None:
        stop = self._budget(state) or self._deadline(state)
        if stop is not None:
            return stop

        verdict = self._detector.observe(state, record)
        if verdict is None:
            return None
        self._fire(verdict.signal, verdict.detail)
        return StopSignal(RunStatus.STUCK, verdict.render())

    # -- individual limits --------------------------------------------------

    def _cap(self, state: LoopState) -> StopSignal | None:
        if state.iteration < self.max_iterations:
            return None
        self._fire("max_iterations", f"the {self.max_iterations}-iteration cap was reached")
        return StopSignal(
            RunStatus.MAX_ITERATIONS_REACHED,
            f"stopped at the {self.max_iterations}-iteration cap; "
            f"returning the best draft seen",
        )

    def _budget(self, state: LoopState) -> StopSignal | None:
        budget = self.limits.token_budget
        if budget <= 0:
            return None
        used = state.usage.total_tokens

        if used >= budget:
            self._fire("token_budget", f"{used:,} of {budget:,} tokens spent")
            return StopSignal(
                RunStatus.BUDGET_EXHAUSTED,
                f"token budget exhausted ({used:,}/{budget:,}); finalising on the best "
                f"draft seen rather than starting an iteration it cannot pay for",
            )

        threshold = budget * self.limits.token_warn_ratio
        if not self._warned and used >= threshold:
            self._warned = True
            message = (
                f"token budget {used / budget:.0%} spent ({used:,}/{budget:,}); "
                "roughly one more iteration remains"
            )
            state.note(message)
            self._emit("budget_warning", used=used, budget=budget, reason=message)
        return None

    def _deadline(self, state: LoopState) -> StopSignal | None:
        limit = self.limits.wall_clock_timeout_s
        if limit <= 0:
            return None
        elapsed = self._clock() - self._started
        if elapsed < limit:
            return None
        self._fire("wall_clock", f"{elapsed:.1f}s elapsed against a {limit:.0f}s limit")
        return StopSignal(
            RunStatus.TIMEOUT,
            f"wall-clock limit reached after {elapsed:.1f}s; returning the best draft seen",
        )

    # -- reporting ----------------------------------------------------------

    def _fire(self, name: str, detail: str) -> None:
        self.triggered.append(name)
        self._emit("guardrail_trip", guardrail=name, detail=detail)

    def snapshot(self, *, tokens_used: int = 0, iterations: int = 0) -> dict[str, t.Any]:
        """What the summary records about limits, spent and remaining.

        Takes the two numbers rather than the ``LoopState``, because the runner
        writes the summary after the loop has returned and the state is gone.
        """
        budget = self.limits.token_budget
        used = tokens_used
        return {
            "iterations": {"used": iterations, "cap": self.max_iterations},
            "tokens": {
                "used": used,
                "budget": budget,
                "fraction": round(used / budget, 4) if budget else None,
            },
            "wall_clock_s": {
                "used": round(self._clock() - self._started, 2),
                "limit": self.limits.wall_clock_timeout_s,
            },
            "triggered": list(self.triggered),
        }


__all__ = ["Clock", "Guardrails", "InputCheck"]
