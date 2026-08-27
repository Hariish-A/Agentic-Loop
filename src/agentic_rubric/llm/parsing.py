"""Tolerant JSON extraction for model output.

Models emit JSON wrapped in prose, fenced in ``` blocks, or with a trailing
comma. This module is rung two of the "unparseable LLM output" fallback ladder
(rung one is forced tool-use, rung three is a repair prompt): cheap, local
salvage before spending another API call.
"""

from __future__ import annotations

import json
import re
import typing as t

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _balanced_span(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch``..``close_ch`` span, string-aware."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def salvage_json(raw: str) -> t.Any | None:
    """Best-effort parse of a JSON value from noisy text. ``None`` if hopeless.

    Tries, in order: the raw string, any fenced code block, the first balanced
    object or array, and finally the same with trailing commas removed.
    """
    if not raw or not raw.strip():
        return None

    candidates: list[str] = [raw.strip()]
    candidates.extend(m.strip() for m in _FENCE.findall(raw))
    for opener, closer in (("{", "}"), ("[", "]")):
        span = _balanced_span(raw, opener, closer)
        if span:
            candidates.append(span)

    for candidate in candidates:
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def parse_tool_arguments(raw: str) -> tuple[dict[str, t.Any], str | None]:
    """Parse a tool-call ``arguments`` string.

    Returns ``(arguments, error)``. A non-``None`` error means the arguments
    could not be recovered; the caller decides whether to repair or fall back,
    so a malformed argument blob never crashes the loop on its own.
    """
    if not raw or not raw.strip():
        return {}, None
    parsed = salvage_json(raw)
    if isinstance(parsed, dict):
        return parsed, None
    if parsed is None:
        return {}, "arguments were not valid JSON"
    return {}, f"arguments parsed to {type(parsed).__name__}, expected an object"


__all__ = ["parse_tool_arguments", "salvage_json"]
