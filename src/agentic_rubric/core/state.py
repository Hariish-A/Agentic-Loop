"""The value objects that flow between Perceive, Reason, Act and Reflect.

The loop is a pipeline of typed hand-offs:

    Observation ──▶ Decision ──▶ ActionResult ──▶ Reflection ──┐
         ▲                                                     │
         └──────────────── fed into the next Perceive ─────────┘

Keeping each hand-off a distinct frozen type is what stops the four steps
collapsing into one another. If Reason could reach into loop state directly, or
Act could decide what to do next, the separation would be nominal.

Two things are deliberately mutable: :class:`Workspace` (tools genuinely edit
the working document) and :class:`LoopState` (it accumulates history). Both are
explicit about it rather than pretending otherwise.
"""

from __future__ import annotations

import time
import typing as t
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from ..llm.types import Usage
from .rubric import ProbeResult, Rubric, ScoreCard


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStatus(str, Enum):
    """How a run ended. Anything other than RUNNING is terminal."""

    RUNNING = "running"
    TARGET_REACHED = "target_reached"
    PLATEAU = "plateau"
    AGENT_FINALIZED = "agent_finalized"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    STUCK = "stuck"  # the harness loop detector saw a repeated cycle
    BUDGET_EXHAUSTED = "budget_exhausted"  # the token budget guardrail tripped
    TIMEOUT = "timeout"  # the wall-clock guardrail tripped
    ERROR = "error"

    @property
    def is_guardrail_stop(self) -> bool:
        """Stopped by the harness rather than by the agent's own judgement."""
        return self in {
            RunStatus.MAX_ITERATIONS_REACHED,
            RunStatus.STUCK,
            RunStatus.BUDGET_EXHAUSTED,
            RunStatus.TIMEOUT,
        }

    @property
    def is_success(self) -> bool:
        return self in {RunStatus.TARGET_REACHED, RunStatus.AGENT_FINALIZED}


# ---------------------------------------------------------------------------
# Perceive output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextMetrics:
    """Deterministic measurements of the current draft.

    Computed in Perceive without an LLM, so the agent always has at least one
    source of truth it cannot talk itself out of.
    """

    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_words: float = 0.0
    sentence_length_stdev: float = 0.0
    longest_sentence_words: int = 0
    flesch_reading_ease: float = 0.0
    hedge_count: int = 0
    filler_count: int = 0
    passive_hits: int = 0

    def render(self) -> str:
        return (
            f"words={self.word_count}, sentences={self.sentence_count}, "
            f"avg_sentence={self.avg_sentence_words:.1f}w "
            f"(sd={self.sentence_length_stdev:.1f}, longest={self.longest_sentence_words}w), "
            f"flesch={self.flesch_reading_ease:.1f}, hedges={self.hedge_count}, "
            f"filler={self.filler_count}, passive={self.passive_hits}"
        )

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_words": round(self.avg_sentence_words, 2),
            "sentence_length_stdev": round(self.sentence_length_stdev, 2),
            "longest_sentence_words": self.longest_sentence_words,
            "flesch_reading_ease": round(self.flesch_reading_ease, 1),
            "hedge_count": self.hedge_count,
            "filler_count": self.filler_count,
            "passive_hits": self.passive_hits,
        }


@dataclass(frozen=True)
class MemoryHit:
    """One recalled memory record. Populated from Milestone 2 onwards."""

    kind: str
    content: str
    score: float = 0.0
    session_id: str = ""
    iteration: int = 0

    def render(self) -> str:
        origin = f"session {self.session_id[:8]}, iter {self.iteration}" if self.session_id else "*"
        return f"[{self.kind} | {origin} | relevance {self.score:.2f}] {self.content}"


@dataclass(frozen=True)
class ReactStep:
    """One completed Thought/Action/Observation triple, for the ReAct scratchpad."""

    iteration: int
    thought: str
    action: str
    observation: str

    def render(self) -> str:
        return (
            f"Iteration {self.iteration}\n"
            f"  Thought: {self.thought}\n"
            f"  Action: {self.action}\n"
            f"  Observation: {self.observation}"
        )


@dataclass(frozen=True)
class Observation:
    """Everything the agent can see at the start of one iteration.

    Assembled entirely by Perceive. Reason receives this and nothing else, which
    is what makes a reasoning failure reproducible: re-run Reason on the same
    Observation and you get the same prompt.
    """

    iteration: int
    draft: str
    rubric: Rubric
    target_score: float
    metrics: TextMetrics
    max_iterations: int = 0
    latest_score: ScoreCard | None = None
    score_history: tuple[float, ...] = ()
    probe_results: tuple[ProbeResult, ...] = ()
    recalled: tuple[MemoryHit, ...] = ()
    scratchpad: tuple[ReactStep, ...] = ()
    last_reflection: Reflection | None = None
    best_score: float | None = None
    notes: tuple[str, ...] = ()  # degradation warnings, truncation notices

    @property
    def current_percent(self) -> float | None:
        return self.latest_score.weighted_percent() if self.latest_score else None

    @property
    def has_been_scored(self) -> bool:
        return self.latest_score is not None

    @property
    def iterations_remaining(self) -> int:
        """Shown to the model so it can budget its turns.

        An agent that does not know it is nearly out of iterations will spend
        the last one on analysis instead of a final revision.
        """
        return max(0, self.max_iterations - self.iteration)


# ---------------------------------------------------------------------------
# Reason output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """The single next action the model chose, with the reasoning behind it.

    ``thought`` is carried as a required argument on every tool schema rather
    than scraped from free text, because several providers return empty content
    alongside a tool call. Structuring it guarantees the ReAct Thought is
    always captured instead of being present only when the provider cooperates.
    """

    action: str
    arguments: dict[str, t.Any] = field(default_factory=dict)
    thought: str = ""
    call_id: str = ""
    raw_text: str = ""
    degraded: bool = False  # produced by a fallback rather than a clean choice
    usage: Usage = field(default_factory=Usage)

    def signature(self) -> str:
        """Order-independent identity, used by the Milestone 3 stuck detector."""
        import json

        return f"{self.action}({json.dumps(self.arguments, sort_keys=True, default=str)})"


# ---------------------------------------------------------------------------
# Act output
# ---------------------------------------------------------------------------


class ErrorKind(str, Enum):
    """Why a tool failed, expressed as *what the harness should do about it*.

    The registry contains every exception, which means the exception type is
    lost by the time the result reaches the harness. Classifying at the point of
    containment keeps the recovery decision type-driven instead of leaving the
    harness to pattern-match on error strings.
    """

    NONE = ""
    #: Arguments did not satisfy the schema. Retrying verbatim cannot help;
    #: sanitising the arguments can.
    VALIDATION = "validation"
    #: The tool does not exist. Route to an alternative.
    UNKNOWN_TOOL = "unknown_tool"
    #: Transient: a rate limit or 5xx from an LLM-backed tool. Back off, retry.
    TRANSIENT = "transient"
    #: The handler declined for a reason the agent should read and react to.
    RECOVERABLE = "recoverable"
    #: A bug or a permanent refusal. Report it and move on.
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ActionResult:
    """What a tool actually did.

    A failed tool is a normal, typed outcome -- ``ok=False`` with an ``error``
    string that goes back into the next Observation. The agent gets to react to
    its own broken tool call instead of the run dying.
    """

    action: str
    arguments: dict[str, t.Any] = field(default_factory=dict)
    ok: bool = True
    output: dict[str, t.Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    #: Set by the registry when it contains an exception; read by the harness.
    error_kind: ErrorKind = ErrorKind.NONE
    #: How many extra attempts the harness spent before this result stuck.
    retry_count: int = 0
    #: True when the harness turned a failure into this success.
    recovered: bool = False

    @classmethod
    def failure(
        cls,
        action: str,
        arguments: dict[str, t.Any],
        error: str,
        *,
        kind: ErrorKind = ErrorKind.TERMINAL,
    ) -> ActionResult:
        return cls(
            action=action,
            arguments=arguments,
            ok=False,
            error=error,
            summary=f"TOOL FAILED: {error}",
            error_kind=kind,
        )


# ---------------------------------------------------------------------------
# Reflect output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reflection:
    """The verdict on one iteration, and the instruction for the next.

    ``task_complete`` is decided by deterministic rules, not by the model. The
    model contributes ``critique``, ``lesson`` and ``next_focus``; its own
    opinion on completion arrives as ``model_votes_done`` and is only ever one
    input to the plateau rule. An agent that can declare itself finished will.
    """

    task_complete: bool
    reason: str
    critique: str = ""
    lesson: str | None = None
    next_focus: str | None = None
    score_delta: float | None = None
    plateau: bool = False
    model_votes_done: bool = False
    status: RunStatus = RunStatus.RUNNING
    usage: Usage = field(default_factory=Usage)
    degraded: bool = False  # the critique came from a fallback, not the model

    def render(self) -> str:
        parts = [f"complete={self.task_complete} ({self.reason})"]
        if self.score_delta is not None:
            parts.append(f"delta={self.score_delta:+.1f}pts")
        if self.next_focus:
            parts.append(f"next_focus={self.next_focus}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Mutable working state
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """The document the tools operate on.

    Tools take a *reference* to the working draft rather than receiving the text
    as an argument. That is a deliberate tool-design choice: passing the draft
    through the model would force it to re-emit thousands of tokens on every
    call, and would let it silently corrupt the text in transit.
    """

    draft: str
    previous_draft: str | None = None
    scorecard: ScoreCard | None = None
    scorecard_history: list[ScoreCard] = field(default_factory=list)
    finalized: bool = False
    finalize_reason: str = ""

    def replace_draft(self, text: str) -> None:
        self.previous_draft = self.draft
        self.draft = text

    def record_score(self, card: ScoreCard) -> None:
        self.scorecard = card
        self.scorecard_history.append(card)

    @property
    def percent(self) -> float | None:
        return self.scorecard.weighted_percent() if self.scorecard else None


@dataclass
class IterationRecord:
    """One full cycle, retained for the trace and the final report."""

    iteration: int
    observation: Observation
    decision: Decision
    result: ActionResult
    reflection: Reflection
    started_at: str = field(default_factory=utc_now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "iteration": self.iteration,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "score_before": self.observation.current_percent,
            "thought": self.decision.thought,
            "action": self.decision.action,
            "arguments": self.decision.arguments,
            "ok": self.result.ok,
            "result": self.result.summary,
            "error": self.result.error,
            "error_kind": self.result.error_kind.value,
            "retry_count": self.result.retry_count,
            "recovered": self.result.recovered,
            "reflection": {
                "task_complete": self.reflection.task_complete,
                "reason": self.reflection.reason,
                "critique": self.reflection.critique,
                "lesson": self.reflection.lesson,
                "next_focus": self.reflection.next_focus,
                "score_delta": self.reflection.score_delta,
                "plateau": self.reflection.plateau,
            },
        }


@dataclass
class LoopState:
    """The running record of one execution.

    ``last_reflection`` is the feedback edge: Reflect writes it, and the next
    Perceive reads it. That single field is what makes this a loop rather than a
    chain of four calls.
    """

    rubric: Rubric
    workspace: Workspace
    target_score: float
    session_id: str = field(default_factory=lambda: new_id("sess"))
    run_id: str = field(default_factory=lambda: new_id("run"))
    iteration: int = 0
    status: RunStatus = RunStatus.RUNNING
    records: list[IterationRecord] = field(default_factory=list)
    scratchpad: list[ReactStep] = field(default_factory=list)
    last_reflection: Reflection | None = None
    plateau_streak: int = 0
    initial_draft: str = ""
    best_draft: str = ""
    best_score: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Running token total across every LLM call this run has made, from any
    #: step or tool. The token-budget guardrail reads this, so it has to be
    #: updated as the run proceeds rather than summed at the end.
    usage: Usage = field(default_factory=Usage)
    started_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.initial_draft:
            self.initial_draft = self.workspace.draft
        if not self.best_draft:
            self.best_draft = self.workspace.draft

    # -- accounting ---------------------------------------------------------

    def record_usage(self, usage: Usage) -> None:
        """Fold one call's tokens into the running total the guardrail watches."""
        self.usage = self.usage + usage

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_monotonic

    # -- the feedback edge --------------------------------------------------

    def advance(self, record: IterationRecord) -> None:
        """Fold one completed iteration back into the state Perceive will read."""
        self.records.append(record)
        self.last_reflection = record.reflection
        self.scratchpad.append(
            ReactStep(
                iteration=record.iteration,
                thought=record.decision.thought,
                action=record.decision.action,
                observation=record.result.summary or (record.result.error or "no output"),
            )
        )
        if record.reflection.plateau:
            self.plateau_streak += 1
        else:
            self.plateau_streak = 0
        self._track_best()

    def _track_best(self) -> None:
        """Keep the highest-scoring draft seen.

        A revision can make things worse. Without this, hitting the iteration
        cap after a bad final edit would return text worse than what the agent
        already had -- which is the single most embarrassing way for an
        improvement agent to fail.
        """
        current = self.workspace.percent
        if current is None:
            return
        if self.best_score is None or current > self.best_score:
            self.best_score = current
            self.best_draft = self.workspace.draft

    @property
    def score_history(self) -> tuple[float, ...]:
        return tuple(card.weighted_percent() for card in self.workspace.scorecard_history)

    def note(self, message: str) -> None:
        self.notes.append(message)


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """What ``AgenticLoop.run`` returns."""

    run_id: str
    session_id: str
    status: RunStatus
    rubric_id: str
    iterations: int
    initial_draft: str
    final_draft: str
    best_draft: str
    initial_score: float | None
    final_score: float | None
    best_score: float | None
    target_score: float
    records: list[IterationRecord] = field(default_factory=list)
    scorecards: list[ScoreCard] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    #: Harness telemetry, filled in by the runner after the loop returns. The
    #: loop cannot know these -- it never sees a retry, by design.
    provider: str = ""
    retry_count: int = 0
    failover_count: int = 0
    repair_count: int = 0
    tool_recovery_count: int = 0
    degraded_memory: bool = False
    cost_est_usd: float = 0.0
    trace_path: str = ""

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def improvement(self) -> float | None:
        if self.initial_score is None or self.best_score is None:
            return None
        return self.best_score - self.initial_score

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "rubric_id": self.rubric_id,
            "iterations": self.iterations,
            "target_score": self.target_score,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "elapsed_s": round(self.elapsed_s, 2),
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "score_trajectory": [round(c.weighted_percent(), 1) for c in self.scorecards],
            "harness": {
                "provider": self.provider,
                "retries": self.retry_count,
                "failovers": self.failover_count,
                "repairs": self.repair_count,
                "tool_recoveries": self.tool_recovery_count,
                "degraded_memory": self.degraded_memory,
                "cost_est_usd": round(self.cost_est_usd, 6),
                "trace": self.trace_path,
            },
            "iterations_detail": [r.to_dict() for r in self.records],
            "notes": self.notes,
        }


__all__ = [
    "ActionResult",
    "Decision",
    "ErrorKind",
    "IterationRecord",
    "LoopState",
    "MemoryHit",
    "Observation",
    "ReactStep",
    "Reflection",
    "RunResult",
    "RunStatus",
    "TextMetrics",
    "Workspace",
    "new_id",
    "utc_now",
]
