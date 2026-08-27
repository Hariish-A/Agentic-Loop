"""Structured JSON logging, with secret redaction applied on the way out.

Two decisions worth stating.

**Redaction happens in the formatter, not at the call sites.** A rule that
depends on every caller remembering it is not a rule. Everything that reaches a
log record or a trace line passes through :func:`redact` first, so the way to
leak a key is to bypass logging entirely rather than to forget a keyword. The
key names come from ``logging.redact_keys`` in config, and the *value* patterns
(``Bearer …``, ``sk-…``, ``gsk_…``) are matched too -- a key pasted into a URL
or an error message does not arrive under a helpfully-named field.

**JSON by default, console on request.** Machine-readable is the right default
for something meant to run unsupervised; ``logging.format: console`` exists for
the times a human is reading along live.

The logger is configured once, by the harness. Library modules call
``logging.getLogger(__name__)`` and never touch handlers, so importing this
package cannot reconfigure an application that embeds it.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import typing as t
from datetime import datetime, timezone

from ..config import LoggingConfig

REDACTED = "[REDACTED]"

LOGGER_NAME = "agentic_rubric"

#: Secrets that arrive inside a *value* rather than under a known key.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\b(?:sk|gsk|xai|api)[-_][A-Za-z0-9._\-]{12,}"),
)

#: Fields the logging module puts on every record; anything else is ours.
_STANDARD = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info", "taskName",
    "thread", "threadName",
})


def scrub_text(value: str) -> str:
    """Mask secrets that appear inside a free-text value."""
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def redact(value: t.Any, keys: t.Collection[str], *, max_chars: int = 0) -> t.Any:
    """Recursively mask secrets. Never raises, never mutates the input.

    ``max_chars`` truncates long strings -- drafts are thousands of characters
    and a trace nobody can open is not observability. Truncation is marked in
    the output rather than silent, so a reader can tell the difference between
    "short" and "shortened".
    """
    lowered = {k.lower() for k in keys}

    def walk(node: t.Any, depth: int = 0) -> t.Any:
        if depth > 12:  # cycles and pathological nesting
            return "[TRUNCATED: too deep]"
        if isinstance(node, dict):
            return {
                key: (
                    REDACTED
                    if str(key).lower() in lowered
                    else walk(value, depth + 1)
                )
                for key, value in node.items()
            }
        if isinstance(node, (list, tuple)):
            return [walk(item, depth + 1) for item in node]
        if isinstance(node, str):
            cleaned = scrub_text(node)
            if max_chars and len(cleaned) > max_chars:
                return cleaned[:max_chars] + f"... [+{len(cleaned) - max_chars} chars]"
            return cleaned
        if isinstance(node, (int, float, bool)) or node is None:
            return node
        return scrub_text(str(node))

    return walk(value)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with every extra field carried through."""

    def __init__(self, redact_keys: t.Collection[str] = (), max_chars: int = 0) -> None:
        super().__init__()
        self.redact_keys = tuple(redact_keys)
        self.max_chars = max_chars

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, t.Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in _STANDARD and k != "extra"}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        safe = redact(payload, self.redact_keys, max_chars=self.max_chars)
        return json.dumps(safe, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line output for when a person is watching."""

    def __init__(self, redact_keys: t.Collection[str] = ()) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        self.redact_keys = tuple(redact_keys)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD}
        if extras:
            safe = redact(extras, self.redact_keys, max_chars=160)
            base += "  " + " ".join(f"{k}={v}" for k, v in safe.items())
        return scrub_text(base)


def configure_logging(
    config: LoggingConfig, *, stream: t.TextIO | None = None
) -> logging.Logger:
    """Configure and return the package logger. Idempotent.

    Writes to **stderr** by default so that ``--json`` output on stdout stays
    pipeable into ``jq`` without log lines corrupting it.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        ConsoleFormatter(config.redact_keys)
        if config.format == "console"
        else JsonFormatter(config.redact_keys, max_chars=config.trace_max_field_chars)
    )
    logger.addHandler(handler)
    return logger


def get_logger(suffix: str = "") -> logging.Logger:
    """A child of the package logger. Never configures handlers itself."""
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME)


__all__ = [
    "LOGGER_NAME",
    "REDACTED",
    "ConsoleFormatter",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "redact",
    "scrub_text",
]
