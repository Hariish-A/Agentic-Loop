"""The provider interface every LLM backend implements.

The loop depends on this ABC and never on a concrete provider, which is what
allows Groq, Ollama and the deterministic mock to be swapped from
``config.yaml`` -- and what lets the Milestone 3 harness wrap a failover chain
around them without the loop noticing.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from .types import LLMResponse, Message, ToolSpec

# "auto" | "required" | "none" | an explicit tool name to force.
ToolChoice = t.Literal["auto", "required", "none"] | str | None


class LLMProvider(ABC):
    """A synchronous chat-completion backend."""

    #: Config key this provider was built from, e.g. ``"groq"``.
    name: str = "unknown"
    #: Concrete model identifier sent on the wire.
    model: str = ""
    #: Whether function/tool calling may be used with this backend.
    supports_tools: bool = True

    @abstractmethod
    def complete(
        self,
        messages: t.Sequence[Message],
        *,
        tools: t.Sequence[ToolSpec] | None = None,
        tool_choice: ToolChoice = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one completion.

        Implementations raise the taxonomy in :mod:`.types`: transient problems
        as :class:`~.types.RetryableLLMError`, permanent ones as
        :class:`~.types.TerminalLLMError`, and unusable payloads as
        :class:`~.types.LLMParseError`.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend holds resources
        """Release network resources. Safe to call more than once."""

    def __enter__(self) -> LLMProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


__all__ = ["LLMProvider", "ToolChoice"]
