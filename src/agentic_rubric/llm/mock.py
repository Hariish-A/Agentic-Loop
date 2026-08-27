"""A deterministic, offline LLM provider.

Two jobs:

1. **Tests.** Every unit test and the end-to-end loop test run against this, so
   the suite needs no API key, costs nothing and cannot flake on a rate limit.
2. **Fault injection.** Scripting an exception at turn *n* is how the harness's
   retry, repair and failover paths get exercised on demand -- including in the
   demo video, via ``--simulate-failure``.
"""

from __future__ import annotations

import itertools
import json
import typing as t
from dataclasses import dataclass, field

from .base import LLMProvider, ToolChoice
from .types import (
    LLMResponse,
    Message,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
    Usage,
)


def tool_call(name: str, call_id: str | None = None, **arguments: t.Any) -> ToolCall:
    """Build a :class:`~.types.ToolCall` for a script, keeping tests readable."""
    raw = json.dumps(arguments, default=str)
    return ToolCall(id=call_id or f"mock_{name}", name=name, arguments=arguments, raw_arguments=raw)


@dataclass
class MockTurn:
    """One scripted reply. Set ``raises`` to make this turn fail instead."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raises: Exception | None = None
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=120, output_tokens=60))
    finish_reason: str = "stop"


@dataclass(frozen=True)
class MockCall:
    """What the caller asked for, recorded for assertions."""

    index: int
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    tool_choice: ToolChoice
    temperature: float | None
    max_tokens: int | None


Responder = t.Callable[[MockCall], "MockTurn | LLMResponse | Exception"]


class MockProvider(LLMProvider):
    """Replays a fixed script, or delegates to a ``responder`` callable.

    A responder receives the full :class:`MockCall` and can therefore react to
    conversation state -- which is what makes a genuinely offline end-to-end
    demo possible, rather than a fixed tape that ignores its input.
    """

    def __init__(
        self,
        script: t.Sequence[MockTurn] = (),
        *,
        responder: Responder | None = None,
        name: str = "mock",
        model: str = "mock-1",
        repeat_last: bool = False,
    ) -> None:
        if not script and responder is None:
            raise ValueError("MockProvider needs either a script or a responder")
        self.name = name
        self.model = model
        self.supports_tools = True
        self._script = list(script)
        self._responder = responder
        self._repeat_last = repeat_last
        self._counter = itertools.count()
        #: Every call made against this provider, in order.
        self.calls: list[MockCall] = []

    def complete(
        self,
        messages: t.Sequence[Message],
        *,
        tools: t.Sequence[ToolSpec] | None = None,
        tool_choice: ToolChoice = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        call = MockCall(
            index=next(self._counter),
            messages=tuple(messages),
            tools=tuple(tools or ()),
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.calls.append(call)

        outcome = self._responder(call) if self._responder else self._from_script(call.index)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, LLMResponse):
            return outcome
        if outcome.raises is not None:
            raise outcome.raises
        return LLMResponse(
            text=outcome.text,
            tool_calls=outcome.tool_calls,
            usage=outcome.usage,
            model=self.model,
            provider=self.name,
            finish_reason=outcome.finish_reason,
            latency_ms=0.0,
            raw={"mock": True, "call_index": call.index},
        )

    def _from_script(self, index: int) -> MockTurn:
        if index < len(self._script):
            return self._script[index]
        if self._repeat_last and self._script:
            return self._script[-1]
        raise ProviderUnavailableError(
            f"mock script exhausted after {len(self._script)} turn(s)", provider=self.name
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


__all__ = ["MockCall", "MockProvider", "MockTurn", "Responder", "tool_call"]
