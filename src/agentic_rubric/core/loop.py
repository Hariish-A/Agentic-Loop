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
3. Milestone 3 adds budget and stuck detection through the same seam.

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
    IterationRecord,
    LoopState,
    Observation,
    RunResult,
    RunStatus,
    Workspace,
    new_id,
)

#: Callback signature for observability. Milestone 3 attaches the JSONL tracer
#: here; Milestone 1 uses it for the console renderer.
EventHook = t.Callable[[str, dict[str, t.Any]], None]


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
    ) -> None:
        self.config = config
        self.provider = provider
        self.rubric = rubric
        # Built from the rubric so the tool schemas carry its real criterion ids.
        self.registry = registry or build_registry(rubric)
        self.memory = memory or NullMemory()
        self._on_event = on_event

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
        reason_usage = Usage()
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
            state.iteration += 1
            ctx.iteration = state.iteration
            iteration_started = time.perf_counter()
            self._emit("iteration_start", iteration=state.iteration, run_id=state.run_id)

            # 1. PERCEIVE - structure the world. No model involved.
            observation = perceive(state, self.config, self.memory)
            self._emit("perceive", iteration=state.iteration, **_observation_event(observation))

            # 2. REASON - one call, one decision.
            decision = reason(observation, self.provider, self.registry, self.config)
            reason_usage = reason_usage + decision.usage
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
            result, tool_usage = act(decision, self.registry, ctx)
            self._emit(
                "act",
                iteration=state.iteration,
                action=result.action,
                ok=result.ok,
                summary=result.summary,
                error=result.error,
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
                status=state.status.value,
            )

        if state.status is RunStatus.RUNNING:
            state.status = RunStatus.MAX_ITERATIONS_REACHED
            state.note(
                f"stopped at the {cap}-iteration cap; returning the best draft seen "
                f"({state.best_score:.1f}%)" if state.best_score is not None
                else f"stopped at the {cap}-iteration cap before any score was recorded"
            )

        result = self._build_result(state, reason_usage + ctx.usage, time.perf_counter() - started)
        self._emit("run_end", **result.to_dict())
        return result

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


__all__ = ["AgenticLoop", "EventHook"]
