"""The agentic loop: Perceive -> Reason -> Act -> Reflect, repeat.

::

        ┌──────────────────── Reflection ────────────────────┐
        │                                                    │
        ▼                                                    │
    PERCEIVE ──▶ REASON ──▶ ACT ──▶ REFLECT ──▶ complete? ───┘
    (no LLM)     (1 call)   (dispatch)  (rules + LLM)    │
                                                        yes
                                                         ▼
                                              best draft + trace

The feedback edge is real, not decorative: :meth:`LoopState.advance` stores the
Reflection, and the next :func:`~.perceive.perceive` reads it. Break that one
line and this becomes four functions called in sequence.

Termination has three sources, in priority order:

1. **Reflect** decides the work is done (target met, credible finalize, plateau).
2. **The loop** enforces ``max_iterations``. The cap is applied here, in Python,
   never delegated to the model.
3. A :class:`LoopController` -- the harness's guardrails and stuck detector --
   may stop the run at either iteration boundary.

The harness attaches through exactly two seams, ``controller`` and ``act_fn``,
and both default to "no harness". Retry, repair, failover and budget enforcement
all live in :mod:`..harness`, so none of them appears here and the loop stays
readable as the four steps it is meant to demonstrate. The single ``except`` in
this module guards a memory *write*, which is an enhancement the run is allowed
to lose -- not error handling the harness should have owned.

Whatever the reason, the run returns the **best-scoring draft ever seen**, not
the last one produced. A revision can make things worse, and an improvement
agent that hands back text worse than its input has failed at its one job.
"""

from __future__ import annotations

import time
import typing as t

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.types import Usage
from ..memory.base import MemoryRecord, MemoryStore, NullMemory
from ..tools.definitions import build_registry
from ..tools.registry import ToolContext, ToolRegistry
from .act import act
from .perceive import perceive
from .reason import reason
from .reflect import reflect
from .rubric import Rubric
from .state import (
    ActionResult,
    Decision,
    IterationRecord,
    LoopState,
    Observation,
    RunResult,
    RunStatus,
    Workspace,
    new_id,
)

#: Callback signature for observability. The JSONL tracer and the console
#: renderer both attach here, so what is watched and what is recorded cannot
#: drift apart.
EventHook = t.Callable[[str, dict[str, t.Any]], None]

#: The Act seam. Defaults to :func:`~.act.act`; the harness substitutes a
#: version that retries transient tool failures and sanitises bad arguments.
ActFn = t.Callable[[Decision, ToolRegistry, ToolContext], "tuple[ActionResult, Usage]"]


class StopSignal(t.NamedTuple):
    """A controller's instruction to end the run, with the status to report."""

    status: RunStatus
    reason: str


class LoopController(t.Protocol):
    """The guardrail seam. Consulted at both iteration boundaries.

    Deliberately narrow: a controller may stop the run and may annotate it, and
    can do nothing else. It cannot choose actions, edit the draft or change a
    score -- so no amount of harness code can quietly become a fifth cognitive
    step.
    """

    def before_iteration(self, state: LoopState) -> StopSignal | None:
        """Called before Perceive. Return a signal to stop instead."""

    def after_iteration(self, state: LoopState, record: IterationRecord) -> StopSignal | None:
        """Called after the Reflection has been folded in."""


class AgenticLoop:
    """One configured agent: a rubric, a provider, a tool set and a memory."""

    def __init__(
        self,
        *,
        config: AppConfig,
        provider: LLMProvider,
        rubric: Rubric,
        registry: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        on_event: EventHook | None = None,
        controller: LoopController | None = None,
        act_fn: ActFn = act,
    ) -> None:
        self.config = config
        self.provider = provider
        self.rubric = rubric
        # Built from the rubric so the tool schemas carry its real criterion ids.
        self.registry = registry or build_registry(rubric)
        self.memory = memory or NullMemory()
        self._on_event = on_event
        self._controller = controller
        self._act = act_fn

    # -- events -------------------------------------------------------------

    def _emit(self, event: str, **payload: t.Any) -> None:
        if self._on_event is not None:
            self._on_event(event, payload)

    # -- memory writes ------------------------------------------------------

    def _remember(self, state: LoopState, record: IterationRecord) -> None:
        """Write what this iteration taught, after Reflect.

        Two records with different lifetimes: an *episodic* one scoped to this
        session so later iterations can see what was already tried, and a
        *lesson* only when Reflect produced one -- lessons outlive the session,
        so writing a vacuous one pollutes every future run.
        """
        if not self.config.memory.enabled:
            return

        reflection = record.reflection
        try:
            self.memory.save(
                MemoryRecord(
                    kind="episodic",
                    content=(
                        f"iteration {record.iteration}: {record.decision.action} -> "
                        f"{record.result.summary or record.result.error}"
                    ),
                    session_id=state.session_id,
                    iteration=record.iteration,
                    rubric_id=self.rubric.id,
                    criterion_id=reflection.next_focus or "",
                    score_delta=reflection.score_delta,
                    metadata={
                        "action": record.decision.action,
                        "ok": record.result.ok,
                        "thought": record.decision.thought,
                    },
                )
            )
            if reflection.lesson:
                self.memory.save(
                    MemoryRecord(
                        kind="lesson",
                        content=reflection.lesson,
                        session_id=state.session_id,
                        iteration=record.iteration,
                        rubric_id=self.rubric.id,
                        criterion_id=reflection.next_focus or "",
                        score_delta=reflection.score_delta,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement
            state.note(f"memory write failed ({type(exc).__name__}: {exc}); continuing")

    # -- guardrail seam -----------------------------------------------------

    def _before(self, state: LoopState) -> StopSignal | None:
        return self._controller.before_iteration(state) if self._controller else None

    def _after(self, state: LoopState, record: IterationRecord) -> StopSignal | None:
        return self._controller.after_iteration(state, record) if self._controller else None

    def _halt(self, signal: StopSignal | None, state: LoopState) -> bool:
        """Apply a controller's stop signal. Returns True when the run must end."""
        if signal is None:
            return False
        state.status = signal.status
        state.note(signal.reason)
        self._emit(
            "guardrail",
            iteration=state.iteration,
            status=signal.status.value,
            reason=signal.reason,
        )
        return True

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        draft: str,
        *,
        session_id: str | None = None,
        target_score: float | None = None,
        max_iterations: int | None = None,
    ) -> RunResult:
        """Run the loop until the task is complete or the iteration cap is hit."""
        if not draft.strip():
            raise ValueError("cannot improve an empty draft")

        target = target_score if target_score is not None else (
            self.rubric.target_score
            if self.rubric.target_score is not None
            else self.config.loop.target_score
        )
        cap = max_iterations if max_iterations is not None else self.config.loop.max_iterations

        state = LoopState(
            rubric=self.rubric,
            workspace=Workspace(draft=draft.strip()),
            target_score=target,
            session_id=session_id or new_id("sess"),
        )
        ctx = ToolContext(
            config=self.config,
            rubric=self.rubric,
            workspace=state.workspace,
            llm=self.provider,
        )

        started = time.perf_counter()
        self._emit(
            "run_start",
            run_id=state.run_id,
            session_id=state.session_id,
            rubric_id=self.rubric.id,
            target_score=target,
            max_iterations=cap,
            provider=self.provider.describe(),
            memory=self.memory.describe,
        )

        while state.iteration < cap and state.status is RunStatus.RUNNING:
            if self._halt(self._before(state), state):
                break

            state.iteration += 1
            ctx.iteration = state.iteration
            iteration_started = time.perf_counter()
            self._emit("iteration_start", iteration=state.iteration, run_id=state.run_id)

            # 1. PERCEIVE - structure the world. No model involved.
            observation = perceive(state, self.config, self.memory)
            self._emit("perceive", iteration=state.iteration, **_observation_event(observation))

            # 2. REASON - one call, one decision.
            decision = reason(observation, self.provider, self.registry, self.config)
            state.record_usage(decision.usage)
            self._emit(
                "reason",
                iteration=state.iteration,
                action=decision.action,
                thought=decision.thought,
                arguments=decision.arguments,
                degraded=decision.degraded,
                tokens=decision.usage.total_tokens,
            )

            # 3. ACT - execute, never decide.
            result, tool_usage = self._act(decision, self.registry, ctx)
            state.record_usage(tool_usage)
            self._emit(
                "act",
                iteration=state.iteration,
                action=result.action,
                ok=result.ok,
                summary=result.summary,
                error=result.error,
                error_kind=result.error_kind.value,
                retry_count=result.retry_count,
                recovered=result.recovered,
                duration_ms=result.duration_ms,
                tokens=tool_usage.total_tokens,
            )

            # 4. REFLECT - measure, critique, decide whether to continue.
            reflection = reflect(
                observation,
                decision,
                result,
                state=state,
                config=self.config,
                provider=self.provider,
            )
            state.record_usage(reflection.usage)
            self._emit(
                "reflect",
                iteration=state.iteration,
                task_complete=reflection.task_complete,
                reason=reflection.reason,
                critique=reflection.critique,
                lesson=reflection.lesson,
                next_focus=reflection.next_focus,
                score_delta=reflection.score_delta,
                plateau=reflection.plateau,
                degraded=reflection.degraded,
                tokens=reflection.usage.total_tokens,
            )

            record = IterationRecord(
                iteration=state.iteration,
                observation=observation,
                decision=decision,
                result=result,
                reflection=reflection,
                duration_ms=(time.perf_counter() - iteration_started) * 1000.0,
            )
            # The feedback edge: this is what the next Perceive will read.
            state.advance(record)
            self._remember(state, record)

            if reflection.task_complete:
                state.status = reflection.status

            self._emit(
                "iteration_end",
                iteration=state.iteration,
                score=state.workspace.percent,
                best_score=state.best_score,
                tokens_used=state.usage.total_tokens,
                status=state.status.value,
            )

            # A guardrail may override the agent's own verdict, but only to stop
            # the run -- never to keep it going after Reflect said it was done.
            if state.status is RunStatus.RUNNING:
                self._halt(self._after(state, record), state)

        if state.status is RunStatus.RUNNING:
            state.status = RunStatus.MAX_ITERATIONS_REACHED
            state.note(
                f"stopped at the {cap}-iteration cap; returning the best draft seen "
                f"({state.best_score:.1f}%)" if state.best_score is not None
                else f"stopped at the {cap}-iteration cap before any score was recorded"
            )

        # Named apart from the per-iteration `result` above: one is an
        # ActionResult, this is the RunResult. Reusing the name compiled but
        # made the last twenty lines of the loop read as though a tool call
        # had somehow become the run.
        run_result = self._build_result(state, state.usage, time.perf_counter() - started)
        # The `harness` block is dropped: the loop never learns that a retry,
        # a repair or a failover happened, so reporting zeroes for them here
        # would be a confident lie. The runner emits `run_summary` with the
        # real numbers once it has folded them in.
        self._emit(
            "run_end", **{k: v for k, v in run_result.to_dict().items() if k != "harness"}
        )
        return run_result

    # -- output -------------------------------------------------------------

    def _build_result(self, state: LoopState, usage: Usage, elapsed: float) -> RunResult:
        history = state.workspace.scorecard_history
        return RunResult(
            run_id=state.run_id,
            session_id=state.session_id,
            status=state.status,
            rubric_id=self.rubric.id,
            iterations=state.iteration,
            initial_draft=state.initial_draft,
            final_draft=state.workspace.draft,
            best_draft=state.best_draft or state.workspace.draft,
            initial_score=history[0].weighted_percent() if history else None,
            final_score=state.workspace.percent,
            best_score=state.best_score,
            target_score=state.target_score,
            records=list(state.records),
            scorecards=list(history),
            notes=list(state.notes),
            elapsed_s=elapsed,
            total_input_tokens=usage.input_tokens,
            total_output_tokens=usage.output_tokens,
        )


def _observation_event(observation: Observation) -> dict[str, t.Any]:
    return {
        "score": observation.current_percent,
        "words": observation.metrics.word_count,
        "flesch": round(observation.metrics.flesch_reading_ease, 1),
        "failing_probes": [r.id for r in observation.probe_results if not r.passed],
        "recalled": len(observation.recalled),
        "notes": list(observation.notes),
    }


__all__ = ["ActFn", "AgenticLoop", "EventHook", "LoopController", "StopSignal"]
