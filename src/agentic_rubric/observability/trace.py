"""One JSONL event per step per iteration, plus a per-run summary.

Layout, one directory per run::

    runs/<run_id>/trace.jsonl     every event, in order, one JSON object a line
    runs/<run_id>/summary.json    the RunResult plus harness telemetry

Why JSONL rather than one JSON document: a trace is append-only and is most
useful when the run *did not* finish. A file that only becomes valid on the
final closing brace is useless in exactly the situation observability exists
for. Every line stands alone, so a killed run still leaves a readable trace, and
``jq`` and ``grep`` both work on it without a parser.

Every event carries the same envelope -- ``run_id``, ``iteration``, ``step``,
``duration_ms``, ``tokens``, ``cost_est``, ``error``, ``retry_count`` -- so a
column means the same thing on every row and questions like "where did the time
go" or "which iteration cost the most" are one ``jq`` away rather than a parsing
exercise. Step-specific detail lands under ``detail``.

The tracer is an ``EventHook``, the same interface the console renderer uses.
Two subscribers on one event source is what stops what a reviewer *sees* and
what the trace *records* from drifting apart.

Cost is estimated from ``cost_per_1k_input`` / ``cost_per_1k_output`` on the
provider's config entry, which default to zero. Every provider in the shipped
chain has a free path, and inventing a price would be worse than an honest zero.
"""

from __future__ import annotations

import json
import time
import typing as t
from pathlib import Path

from ..config import AppConfig
from .logger import get_logger, redact

EventHook = t.Callable[[str, dict[str, t.Any]], None]

TRACE_FILENAME = "trace.jsonl"
SUMMARY_FILENAME = "summary.json"

#: Which loop step each event belongs to, for the ``step`` column.
_STEP_OF = {
    "perceive": "perceive",
    "reason": "reason",
    "act": "act",
    "reflect": "reflect",
    "retry": "reason",
    "repair": "reason",
    "tool_recovery": "act",
}

#: Payload keys promoted into the envelope rather than nested under `detail`.
_ENVELOPE_KEYS = ("iteration", "duration_ms", "error", "retry_count", "action", "tokens")


def fanout(*hooks: EventHook | None) -> EventHook:
    """Combine event subscribers. A failing subscriber must not end the run.

    Observability is the one component that absolutely must not be able to kill
    the thing it observes -- a crash in a renderer would otherwise take down a
    run that was going perfectly well.
    """
    live = [hook for hook in hooks if hook is not None]

    def emit(event: str, payload: dict[str, t.Any]) -> None:
        for hook in live:
            try:
                hook(event, payload)
            except Exception as exc:  # noqa: BLE001 - see the docstring
                get_logger("trace").warning(
                    "event subscriber failed", extra={"event": event, "error": str(exc)}
                )

    return emit


class RunTracer:
    """Writes ``trace.jsonl`` and ``summary.json`` for one run.

    The directory is created when the ``run_start`` event arrives, because that
    is the event carrying the ``run_id`` -- the loop mints its own id, and
    having the tracer invent a second one would mean two names for one run.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        config: AppConfig | None = None,
        enabled: bool = True,
        provider_name: str = "",
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.config = config
        self.provider_name = provider_name
        self.run_id = ""
        self.session_id = ""
        self.run_dir: Path | None = None
        self.events_written = 0
        self._handle: t.TextIO | None = None
        self._log = get_logger("trace")
        self._redact_keys = config.logging.redact_keys if config else ()
        self._max_chars = config.logging.trace_max_field_chars if config else 2000
        self._iteration = 0
        self._step_started = time.perf_counter()

    # -- lifecycle ----------------------------------------------------------

    def open(self, run_id: str) -> Path:
        """Create the run directory and open the trace file."""
        self.run_id = run_id
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._handle = (run_dir / TRACE_FILENAME).open("a", encoding="utf-8")
        return run_dir

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> RunTracer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the hook -----------------------------------------------------------

    def __call__(self, event: str, payload: dict[str, t.Any]) -> None:
        if not self.enabled:
            return
        if event == "run_start":
            self.open(str(payload.get("run_id") or "run_unknown"))
            # Carried onto every later line: a trace whose rows cannot be tied
            # back to a memory session is missing half the story.
            self.session_id = str(payload.get("session_id") or "")
        if event == "iteration_start":
            self._iteration = int(payload.get("iteration") or 0)
        if event == "failover" and payload.get("to"):
            # Price the rest of the run against the provider actually serving it.
            self.provider_name = str(payload["to"])
        self.write(event, payload)

    def write(self, event: str, payload: dict[str, t.Any]) -> None:
        """Append one envelope. Never raises: tracing cannot fail a run."""
        if self._handle is None:
            return
        try:
            line = json.dumps(self._envelope(event, payload), default=str, ensure_ascii=False)
            self._handle.write(line + "\n")
            self._handle.flush()  # a trace that is buffered when a run is killed is no trace
            self.events_written += 1
        except Exception as exc:  # noqa: BLE001
            self._log.warning("could not write trace event", extra={"error": str(exc)})

    def _envelope(self, event: str, payload: dict[str, t.Any]) -> dict[str, t.Any]:
        tokens = payload.get("tokens")
        token_count = tokens if isinstance(tokens, int) else None
        detail = {k: v for k, v in payload.items() if k not in _ENVELOPE_KEYS}

        envelope: dict[str, t.Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "run_id": self.run_id,
            "session_id": payload.get("session_id") or self.session_id,
            "iteration": payload.get("iteration", self._iteration),
            "event": event,
            "step": _STEP_OF.get(event, event),
            "tool": payload.get("action") or payload.get("tool") or None,
            "duration_ms": _round(payload.get("duration_ms")),
            "tokens": token_count,
            "cost_est": self._cost(payload),
            "error": payload.get("error"),
            "retry_count": payload.get("retry_count", 0),
            "detail": detail,
        }
        return t.cast(
            "dict[str, t.Any]",
            redact(envelope, self._redact_keys, max_chars=self._max_chars),
        )

    # -- cost ---------------------------------------------------------------

    def _cost(self, payload: dict[str, t.Any]) -> float:
        """Estimated USD for this event's tokens. Zero unless prices are set."""
        tokens = payload.get("tokens")
        if not isinstance(tokens, int) or tokens <= 0 or self.config is None:
            return 0.0
        return round(estimate_cost(self.config, self.provider_name, tokens, 0), 6)

    # -- summary ------------------------------------------------------------

    def write_summary(self, summary: t.Mapping[str, t.Any]) -> Path | None:
        """Write ``summary.json``. Called by the runner, after annotation."""
        if not self.enabled or self.run_dir is None:
            return None
        path = self.run_dir / SUMMARY_FILENAME
        safe = redact(dict(summary), self._redact_keys, max_chars=self._max_chars)
        try:
            path.write_text(
                json.dumps(safe, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            self._log.warning("could not write summary", extra={"error": str(exc)})
            return None
        return path

    @property
    def trace_path(self) -> str:
        return str(self.run_dir / TRACE_FILENAME) if self.run_dir else ""


def estimate_cost(
    config: AppConfig, provider_name: str, input_tokens: int, output_tokens: int
) -> float:
    """Cost in USD from the provider's declared per-1k prices (default 0.0)."""
    try:
        settings = config.llm.provider(provider_name)
    except Exception:  # noqa: BLE001 - an unknown provider costs nothing to report
        return 0.0
    return (
        input_tokens / 1000.0 * settings.cost_per_1k_input
        + output_tokens / 1000.0 * settings.cost_per_1k_output
    )


def read_trace(path: Path | str) -> list[dict[str, t.Any]]:
    """Read a trace file back. Used by the tests and by the demo tooling."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _round(value: t.Any) -> float | None:
    return round(float(value), 2) if isinstance(value, (int, float)) else None


__all__ = [
    "SUMMARY_FILENAME",
    "TRACE_FILENAME",
    "EventHook",
    "RunTracer",
    "estimate_cost",
    "fanout",
    "read_trace",
]
