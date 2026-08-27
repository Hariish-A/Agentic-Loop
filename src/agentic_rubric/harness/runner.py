"""Assemble retry, fallbacks, guardrails and tracing around the loop.

This is the only module that knows about all of them, and that is the point.
:class:`~..core.loop.AgenticLoop` contains no ``try/except`` at all: it reads as
Perceive, Reason, Act, Reflect, because everything that could go wrong is
handled by objects that were substituted for its collaborators before it
started.

Four substitutions, all through interfaces the loop already had:

===========================  ==================================================
loop collaborator            what the runner passes instead
===========================  ==================================================
``provider``                 :class:`~.fallbacks.ResilientProvider` -- retry,
                             local salvage, one repair call, sticky failover
``act_fn``                   :class:`~.fallbacks.ToolRecovery` -- the same
                             signature as ``act``, with a recovery ladder
``controller``               :class:`~.guardrails.Guardrails` -- budget, clock,
                             iteration cap, stuck detection
``on_event``                 tracer + console renderer, fanned out
===========================  ==================================================

The last thing the runner does is annotate the :class:`~..core.state.RunResult`
with what the harness actually had to do -- retries, repairs, failovers, tool
recoveries, whether memory degraded -- and write ``summary.json``. The loop
cannot report those itself: by design it never learns that any of it happened.
"""

from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from ..core.loop import AgenticLoop, EventHook
from ..core.rubric import Rubric
from ..core.state import RunResult
from ..llm.base import LLMProvider
from ..memory.base import MemoryStore, NullMemory
from ..observability.logger import get_logger
from ..observability.trace import RunTracer, estimate_cost, fanout
from ..tools.registry import ToolRegistry
from .fallbacks import ProviderChain, ResilientProvider, ToolRecovery
from .guardrails import Guardrails
from .retry import RetryPolicy


@dataclass
class RunnerReport:
    """What one supervised run produced, beyond the RunResult itself."""

    result: RunResult
    run_dir: Path | None
    trace_path: str
    guardrails: dict[str, t.Any]
    notes: list[str]


class Runner:
    """Runs the loop under supervision. One instance per run.

    Deliberately not reusable across runs: the retry counters, the wall-clock
    deadline and the trace file all belong to a single execution, and sharing
    them would make the second run's telemetry a lie.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        rubric: Rubric,
        provider: LLMProvider,
        memory: MemoryStore | None = None,
        registry: ToolRegistry | None = None,
        chain: ProviderChain | None = None,
        console: EventHook | None = None,
        trace: bool | None = None,
        sleep: t.Callable[[float], None] | None = None,
        clock: t.Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.rubric = rubric
        self.memory = memory or NullMemory()
        self.notes: list[str] = []
        self._log = get_logger("runner")

        enabled = config.logging.trace_enabled if trace is None else trace
        self.tracer = RunTracer(
            config.path(config.logging.trace_dir),
            config=config,
            enabled=enabled,
            provider_name=provider.name,
        )
        self.events: EventHook = fanout(console, self.tracer, self._to_log)

        self.provider = ResilientProvider(
            chain or ProviderChain.of(provider),
            policy=RetryPolicy.for_llm(config.retry),
            repair_attempts=config.retry.repair_attempts,
            emit=self._emit,
            sleep=sleep,
        )
        self.tools = ToolRecovery(
            policy=RetryPolicy.for_tool(config.retry),
            emit=self._emit,
            sleep=sleep or time.sleep,
        )
        self.guardrails = Guardrails(config, emit=self._emit, clock=clock)

        self.loop = AgenticLoop(
            config=config,
            provider=self.provider,
            rubric=rubric,
            registry=registry,
            memory=self.memory,
            on_event=self.events,
            controller=self.guardrails,
            act_fn=self.tools,
        )

    # -- event plumbing -----------------------------------------------------

    def _emit(self, event: str, **payload: t.Any) -> None:
        """Harness components emit through here, onto the same event stream."""
        self.events(event, payload)

    def _to_log(self, event: str, payload: dict[str, t.Any]) -> None:
        """Mirror the interesting events into the structured logger.

        Only the ones an operator would want in a log aggregator: everything is
        already in the trace file, and duplicating all of it into stderr makes
        the console unreadable during a demo.
        """
        level = {
            "retry": "warning",
            "repair": "warning",
            "failover": "error",
            "tool_recovery": "warning",
            "guardrail": "warning",
            "guardrail_trip": "warning",
            "budget_warning": "warning",
            # `run_summary`, not `run_end`: the latter carries every iteration's
            # full detail and would bury a live transcript in one line.
            "run_summary": "info",
        }.get(event)
        if level is None:
            return
        getattr(self._log, level)(
            event, extra={k: v for k, v in payload.items() if k not in ("notes", "guardrails")}
        )

    # -- the run ------------------------------------------------------------

    def run(
        self,
        draft: str,
        *,
        session_id: str | None = None,
        target_score: float | None = None,
        max_iterations: int | None = None,
    ) -> RunnerReport:
        """Run the loop with every guardrail installed, and report on it."""
        check = self.guardrails.check_input(draft)
        if check.truncated:
            self.notes.append(check.note)

        try:
            result = self.loop.run(
                check.text,
                session_id=session_id,
                target_score=target_score,
                max_iterations=max_iterations,
            )
        except BaseException:
            # The trace must survive whatever happened, KeyboardInterrupt included.
            self.tracer.close()
            raise

        self._annotate(result)
        summary = result.to_dict()
        summary["guardrails"] = self.guardrails.snapshot(
            tokens_used=result.total_tokens, iterations=result.iterations
        )
        summary["notes"] = [*result.notes, *self.notes]

        # The last event of the run: what the harness had to do to get here.
        # Emitted before the trace closes so it lands in trace.jsonl too.
        self._emit(
            "run_summary",
            run_id=result.run_id,
            session_id=result.session_id,
            status=result.status.value,
            harness=summary["harness"],
            guardrails=summary["guardrails"],
            notes=self.notes,
        )
        self.tracer.close()
        self.tracer.write_summary(summary)

        return RunnerReport(
            result=result,
            run_dir=self.tracer.run_dir,
            trace_path=self.tracer.trace_path,
            guardrails=summary["guardrails"],
            notes=list(self.notes),
        )

    def _annotate(self, result: RunResult) -> None:
        """Fold harness telemetry into the result the caller will render."""
        result.provider = self.provider.describe()
        result.retry_count = self.provider.retries
        result.failover_count = self.provider.failovers
        result.repair_count = self.provider.repairs
        result.tool_recovery_count = self.tools.recoveries
        result.degraded_memory = bool(getattr(self.memory, "degraded", False))
        result.trace_path = self.tracer.trace_path
        result.cost_est_usd = estimate_cost(
            self.config,
            self.provider.name,
            result.total_input_tokens,
            result.total_output_tokens,
        )
        if result.degraded_memory:
            reason = getattr(self.memory, "degraded_reason", "memory unavailable")
            self.notes.append(f"memory degraded during the run: {reason}")

    def close(self) -> None:
        self.provider.close()
        self.memory.close()
        self.tracer.close()

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["Runner", "RunnerReport"]
