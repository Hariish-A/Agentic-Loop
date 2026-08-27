"""Provider-neutral value objects and the LLM error taxonomy.

The error hierarchy is deliberately split by *what the caller should do*, not by
which HTTP code produced it. :class:`RetryableLLMError` means "back off and try
again"; :class:`TerminalLLMError` means "stop, retrying cannot help"; and
:class:`LLMParseError` means "the call succeeded but the body was unusable",
which the harness answers with a repair prompt rather than a plain retry.
That split is what lets the retry decorator in Milestone 3 stay a dozen lines.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field

Role = t.Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for every failure originating in the LLM layer."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


class RetryableLLMError(LLMError):
    """Transient failure: rate limit, timeout, connection reset, 5xx."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status=status)
        # Honoured by the retry decorator in preference to computed backoff.
        self.retry_after_s = retry_after_s


class RateLimitError(RetryableLLMError):
    """HTTP 429 -- the free-tier failure mode this project is most likely to hit."""


class LLMTimeoutError(RetryableLLMError):
    """The request exceeded the provider timeout."""


class TransientServerError(RetryableLLMError):
    """HTTP 5xx from the provider."""


class TerminalLLMError(LLMError):
    """Retrying will not help: bad key, malformed request, unknown model."""


class AuthError(TerminalLLMError):
    """HTTP 401/403 -- missing or rejected credentials."""


class BadRequestError(TerminalLLMError):
    """HTTP 400/404/422 -- the request itself is wrong."""


class ProviderUnavailableError(LLMError):
    """The provider cannot be used at all (no key configured, host unreachable).

    Not retryable against *this* provider, but it is the trigger for the
    harness to fail over to the next provider in the chain.
    """


class LLMParseError(LLMError):
    """The HTTP call succeeded but the payload could not be interpreted.

    Carries ``raw`` so the fallback ladder can attempt salvage or feed the text
    back to the model in a repair prompt.
    """

    def __init__(self, message: str, *, raw: str = "", provider: str = "") -> None:
        super().__init__(message, provider=provider)
        self.raw = raw


# ---------------------------------------------------------------------------
# Messages and tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A model request to invoke one tool."""

    id: str
    name: str
    arguments: dict[str, t.Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def signature(self) -> str:
        """Stable identity used by the loop-repetition guardrail."""
        return f"{self.name}({json.dumps(self.arguments, sort_keys=True, default=str)})"


@dataclass(frozen=True)
class Message:
    """One chat turn, in the shape every OpenAI-compatible endpoint expects."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_wire(self) -> dict[str, t.Any]:
        payload: dict[str, t.Any] = {"role": self.role}
        # An assistant turn that only requests tools legitimately has no content.
        payload["content"] = self.content or (None if self.tool_calls else "")
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.raw_arguments or json.dumps(c.arguments, default=str),
                    },
                }
                for c in self.tool_calls
            ]
        return payload


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str = "", tool_calls: t.Sequence[ToolCall] = ()) -> Message:
    return Message(role="assistant", content=content, tool_calls=tuple(tool_calls))


def tool_result(call: ToolCall, content: str) -> Message:
    return Message(role="tool", content=content, name=call.name, tool_call_id=call.id)


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: name, description, and a JSON Schema.

    Keeping the schema here (rather than inline at the call site) means the same
    definition drives the wire format, argument validation and the docs table.
    """

    name: str
    description: str
    parameters: dict[str, t.Any]

    def to_wire(self) -> dict[str, t.Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Token accounting. ``estimated`` is set when the provider omitted usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated=self.estimated or other.estimated,
        )

    @classmethod
    def estimate(cls, prompt_chars: int, completion_chars: int) -> Usage:
        """Crude ~4-chars-per-token fallback for providers that report nothing.

        Deliberately rough: it exists so the token-budget guardrail still has a
        signal on Ollama, not to be an accurate biller.
        """
        return cls(
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, completion_chars // 4),
            estimated=True,
        )


@dataclass(frozen=True)
class LLMResponse:
    """Everything one completion produced, provider-neutral."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: dict[str, t.Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def first_tool_call(self) -> ToolCall:
        if not self.tool_calls:
            raise LLMParseError(
                "expected a tool call but the model returned prose",
                raw=self.text,
                provider=self.provider,
            )
        return self.tool_calls[0]


__all__ = [
    "AuthError",
    "BadRequestError",
    "LLMError",
    "LLMParseError",
    "LLMResponse",
    "LLMTimeoutError",
    "Message",
    "ProviderUnavailableError",
    "RateLimitError",
    "RetryableLLMError",
    "Role",
    "TerminalLLMError",
    "ToolCall",
    "ToolSpec",
    "TransientServerError",
    "Usage",
    "assistant",
    "system",
    "tool_result",
    "user",
]
