"""Human-readable console rendering of loop events.

Consumes the same ``on_event`` stream the JSONL tracer attaches to. Two
subscribers, one event source: what a reviewer watches on screen and what gets
written to the trace file cannot drift apart, because they are fed from the
same call.

Harness events (retry, repair, failover, tool recovery, guardrail trips) render
inline with the step that provoked them, marked ``!``. Recovery that is invisible
is indistinguishable from a slow happy path, and a reviewer watching a demo has
no other way to tell that a failure was handled rather than absent.

Deliberately ASCII-only. Box-drawing characters and emoji render as mojibake in
a Windows console under a legacy code page, and a demo video is not the place to
discover that.
"""

from __future__ import annotations

import sys
import typing as t

BAR_WIDTH = 34

STEP_LABEL = {
    "perceive": "PERCEIVE",
    "reason": "REASON  ",
    "act": "ACT     ",
    "reflect": "REFLECT ",
    "harness": "HARNESS ",
}


def score_bar(percent: float | None, target: float, width: int = BAR_WIDTH) -> str:
    """A text progress bar with the target marked, e.g. ``[#####----|---]``."""
    if percent is None:
        return "[" + "?" * width + "]"
    filled = int(round(width * max(0.0, min(100.0, percent)) / 100.0))
    marker = int(round(width * max(0.0, min(100.0, target)) / 100.0))
    cells = ["#" if i < filled else "-" for i in range(width)]
    if 0 <= marker < width:
        cells[marker] = "|" if cells[marker] == "-" else "|"
    return "[" + "".join(cells) + "]"


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    if not text:
        return ""
    return "\n".join(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            replace_whitespace=True,
        ).splitlines()
    )


class ConsoleRenderer:
    """Renders loop events as a readable transcript.

    ``verbose`` adds the Perceive detail line and full reflection critiques;
    without it the output is one block per iteration, which is what fits on
    screen during a walkthrough.
    """

    def __init__(
        self,
        stream: t.TextIO | None = None,
        *,
        verbose: bool = False,
        width: int = 88,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.verbose = verbose
        self.width = width
        self._target = 0.0

    # -- plumbing -----------------------------------------------------------

    def _write(self, text: str = "") -> None:
        print(text, file=self.stream)

    def _rule(self, title: str = "") -> None:
        if title:
            self._write(f"{'=' * 4} {title} {'=' * max(0, self.width - len(title) - 6)}")
        else:
            self._write("=" * self.width)

    def __call__(self, event: str, payload: dict[str, t.Any]) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    # -- events -------------------------------------------------------------

    def _on_run_start(self, payload: dict[str, t.Any]) -> None:
        self._target = float(payload.get("target_score", 0.0))
        self._rule("RUN START")
        self._write(f"  run       : {payload.get('run_id')}")
        self._write(f"  session   : {payload.get('session_id')}")
        self._write(f"  rubric    : {payload.get('rubric_id')}")
        self._write(f"  provider  : {payload.get('provider')}")
        self._write(f"  memory    : {payload.get('memory')}")
        self._write(
            f"  target    : {self._target:.0f}%   "
            f"max iterations: {payload.get('max_iterations')}"
        )
        self._write()

    def _on_iteration_start(self, payload: dict[str, t.Any]) -> None:
        self._rule(f"ITERATION {payload.get('iteration')}")

    def _on_perceive(self, payload: dict[str, t.Any]) -> None:
        score = payload.get("score")
        current = f"{score:.1f}%" if isinstance(score, (int, float)) else "unscored"
        self._write(
            f"  {STEP_LABEL['perceive']}  {score_bar(score, self._target)}  {current}"
        )
        if self.verbose or payload.get("notes"):
            failing = payload.get("failing_probes") or []
            self._write(
                f"            words={payload.get('words')} "
                f"flesch={payload.get('flesch')} "
                f"failing_probes={len(failing)} "
                f"recalled={payload.get('recalled')}"
            )
        for note in payload.get("notes") or []:
            self._write(f"            ! {note}")

    def _on_reason(self, payload: dict[str, t.Any]) -> None:
        flag = "  [DEGRADED FALLBACK]" if payload.get("degraded") else ""
        self._write(f"  {STEP_LABEL['reason']}  -> {payload.get('action')}{flag}")
        thought = str(payload.get("thought") or "")
        if thought:
            self._write(_wrap(f'"{thought}"', self.width, " " * 12))

    def _on_act(self, payload: dict[str, t.Any]) -> None:
        status = "ok" if payload.get("ok") else "FAILED"
        detail = payload.get("summary") if payload.get("ok") else payload.get("error")
        self._write(
            f"  {STEP_LABEL['act']}  {payload.get('action')} [{status}] "
            f"({payload.get('duration_ms', 0.0):.0f} ms)"
        )
        self._write(_wrap(str(detail or ""), self.width, " " * 12))

    def _on_reflect(self, payload: dict[str, t.Any]) -> None:
        delta = payload.get("score_delta")
        movement = f"  delta={delta:+.1f}pts" if isinstance(delta, (int, float)) else ""
        plateau = "  [PLATEAU]" if payload.get("plateau") else ""
        degraded = "  [RULE-BASED]" if payload.get("degraded") else ""
        self._write(
            f"  {STEP_LABEL['reflect']}  complete={payload.get('task_complete')}"
            f"{movement}{plateau}{degraded}"
        )
        if self.verbose and payload.get("critique"):
            self._write(_wrap(str(payload["critique"]), self.width, " " * 12))
        if payload.get("lesson"):
            self._write(_wrap(f"LESSON: {payload['lesson']}", self.width, " " * 12))
        if payload.get("next_focus"):
            self._write(f"            next focus: {payload['next_focus']}")

    # -- harness events -----------------------------------------------------

    def _harness(self, text: str) -> None:
        self._write(f"  {STEP_LABEL['harness']}  ! {text}")

    def _on_retry(self, payload: dict[str, t.Any]) -> None:
        source = "Retry-After" if payload.get("honoured_retry_after") else "backoff"
        self._harness(
            f"retry {payload.get('attempt')} on {payload.get('provider')} after "
            f"{payload.get('delay_s')}s ({source}): {payload.get('error_type')}"
        )

    def _on_repair(self, payload: dict[str, t.Any]) -> None:
        method = payload.get("method")
        label = "salvaged the tool call locally" if method == "local_salvage" else (
            "sent one repair prompt"
        )
        self._harness(f"unusable reply from {payload.get('provider')}: {label}")

    def _on_failover(self, payload: dict[str, t.Any]) -> None:
        destination = payload.get("to") or "nothing left in the chain"
        self._harness(f"failover {payload.get('from')} -> {destination}")
        self._write(_wrap(str(payload.get("reason") or ""), self.width, " " * 14))

    def _on_tool_recovery(self, payload: dict[str, t.Any]) -> None:
        target = payload.get("retry_as")
        outcome = f"-> {target}" if target else "(handed back to the agent as an observation)"
        self._harness(f"{payload.get('action')} failed; {payload.get('method')} {outcome}")

    def _on_budget_warning(self, payload: dict[str, t.Any]) -> None:
        self._harness(str(payload.get("reason") or "token budget warning"))

    def _on_guardrail_trip(self, payload: dict[str, t.Any]) -> None:
        self._harness(f"[{payload.get('guardrail')}] {payload.get('detail')}")

    def _on_guardrail(self, payload: dict[str, t.Any]) -> None:
        self._write(f"  {STEP_LABEL['harness']}  STOP ({payload.get('status')})")
        self._write(_wrap(str(payload.get("reason") or ""), self.width, " " * 12))

    def _on_iteration_end(self, payload: dict[str, t.Any]) -> None:
        self._write()

    def _on_run_end(self, payload: dict[str, t.Any]) -> None:
        self._rule("RUN COMPLETE")
        trajectory = payload.get("score_trajectory") or []
        initial = payload.get("initial_score")
        best = payload.get("best_score")
        tokens = payload.get("tokens") or {}

        self._write(f"  status      : {payload.get('status')}")
        self._write(f"  iterations  : {payload.get('iterations')}")
        self._write(
            "  trajectory  : "
            + (" -> ".join(f"{value:.1f}%" for value in trajectory) or "never scored")
        )
        if isinstance(initial, (int, float)) and isinstance(best, (int, float)):
            self._write(
                f"  score       : {initial:.1f}% -> {best:.1f}%  "
                f"({best - initial:+.1f} points)"
            )
            self._write(
                f"  best        : {score_bar(best, self._target)}  "
                f"target {self._target:.0f}%"
            )
        self._write(
            f"  tokens      : {tokens.get('input', 0):,} in / "
            f"{tokens.get('output', 0):,} out"
        )
        self._write(f"  elapsed     : {payload.get('elapsed_s', 0.0):.2f}s")

        for note in payload.get("notes") or []:
            self._write(f"  ! {note}")

    def _on_run_summary(self, payload: dict[str, t.Any]) -> None:
        """What the harness had to do. Emitted by the runner, not the loop."""
        harness = payload.get("harness") or {}
        guards = payload.get("guardrails") or {}
        self._write(
            f"  harness     : provider={harness.get('provider') or 'n/a'} "
            f"retries={harness.get('retries', 0)} "
            f"repairs={harness.get('repairs', 0)} "
            f"failovers={harness.get('failovers', 0)} "
            f"tool_recoveries={harness.get('tool_recoveries', 0)}"
        )
        if harness.get("degraded_memory"):
            self._write("                memory DEGRADED (circuit breaker open)")
        tokens = guards.get("tokens") or {}
        if tokens.get("budget"):
            self._write(
                f"  budget      : {tokens.get('used', 0):,} / {tokens['budget']:,} tokens "
                f"({(tokens.get('fraction') or 0) * 100:.0f}%)"
            )
        if guards.get("triggered"):
            self._write(f"  guardrails  : {', '.join(guards['triggered'])}")
        cost = harness.get("cost_est_usd")
        if isinstance(cost, (int, float)) and cost > 0:
            self._write(f"  cost (est)  : ${cost:.4f}")
        if harness.get("trace"):
            self._write(f"  trace       : {harness['trace']}")
        for note in payload.get("notes") or []:
            self._write(f"  ! {note}")
        self._write()


__all__ = ["ConsoleRenderer", "score_bar"]
