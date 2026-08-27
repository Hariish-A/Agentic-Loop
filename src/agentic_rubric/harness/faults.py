"""Deliberate failure injection, so recovery can be demonstrated on demand.

Every path in :mod:`.fallbacks` is reachable from the command line::

    python -m agentic_rubric.cli --provider mock --input samples/weak_essay.txt \\
        --simulate-failure rate_limit

The alternative is asserting that the harness *would* recover if a provider ever
rate-limited us, which is a claim, not evidence. These six kinds cover one
distinct rung of the ladder each, and each is injected at the layer where the
real thing would occur -- a rate limit inside the provider, a tool fault inside
the registry, a memory outage inside the store -- rather than by short-circuiting
the harness into pretending.

============  =========================  ==========================================
kind          injected as                exercises
============  =========================  ==========================================
rate_limit    HTTP 429 with Retry-After  backoff, honoured Retry-After, recovery
server_error  HTTP 503                   backoff, jitter, recovery
bad_json      LLMParseError              local salvage -> repair call -> default
provider_down connection refused         failover to the next provider in the chain
tool_error    ToolError from a handler   typed recovery, then error-as-observation
memory_down   recall raises every time   circuit breaker, degraded_memory, continue
budget        a tiny token budget        graceful stop, best draft, budget_exhausted
============  =========================  ==========================================
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ..llm.base import LLMProvider
from ..llm.types import (
    LLMParseError,
    LLMResponse,
    Message,
    ProviderUnavailableError,
    RateLimitError,
    ToolSpec,
    TransientServerError,
)
from ..memory.base import MemoryRecord, MemoryStore
from ..prompts import STEP_JUDGE, STEPS, classify_step
from ..tools.registry import Handler, ToolContext, ToolOutput, ToolRegistry

#: Kinds injected at the LLM layer, via the mock responder's ``fail_on``.
LLM_KINDS = ("rate_limit", "bad_json", "server_error", "provider_down")
#: Kinds injected elsewhere in the stack.
STACK_KINDS = ("tool_error", "memory_down", "budget")
FAILURE_KINDS = (*LLM_KINDS, *STACK_KINDS)

#: Token budget forced by ``--simulate-failure budget``. Low enough that the
#: guardrail trips two or three iterations in, so the transcript still shows a
#: working loop before the stop.
SIMULATED_TOKEN_BUDGET = 900

#: The tool a simulated tool failure targets: the one whose failure matters.
FAULTY_TOOL = "revise_text"

#: Steps a provider-layer fault can be aimed at.
FAULT_STEPS = STEPS


def llm_failure(kind: str, message: str = "", *, provider: str = "mock") -> Exception:
    """Build an injectable provider-layer failure by name."""
    registry: dict[str, Exception] = {
        "rate_limit": RateLimitError(
            message or "429 Too Many Requests (injected)",
            provider=provider,
            status=429,
            retry_after_s=0.05,
        ),
        "server_error": TransientServerError(
            message or "503 Service Unavailable (injected)", provider=provider, status=503
        ),
        "bad_json": LLMParseError(
            message or "tool arguments were not valid JSON (injected)",
            raw="{unterminated",
            provider=provider,
        ),
        "provider_down": ProviderUnavailableError(
            message or "connection refused (injected)", provider=provider
        ),
    }
    if kind not in registry:
        raise ValueError(f"unknown LLM failure kind {kind!r}; known: {sorted(registry)}")
    return registry[kind]


class FaultyProvider(LLMProvider):
    """Wraps any provider and fails a named step once, then behaves normally.

    Exists because failure injection used to live in the mock responder, which
    made every recovery demo an offline demo. That is a weaker claim than it
    looks: proving the harness recovers from a *scripted* 429 says nothing about
    a real request. This decorator sits over the live client instead, so the
    same ladder is exercised against the same provider the run is really using.

    Which step is failing is inferred from the tools the request offered, using
    the one definition of that rule in :func:`...prompts.classify_step` -- the
    prompts stay free of markers that exist only for demos.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        kind: str,
        step: str = STEP_JUDGE,
        times: int = 1,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.supports_tools = inner.supports_tools
        self.kind = kind
        self.step = step
        self.remaining = times
        self.injected = 0

    def complete(
        self,
        messages: t.Sequence[Message],
        *,
        tools: t.Sequence[ToolSpec] | None = None,
        tool_choice: t.Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.remaining > 0 and classify_step(
            spec.name for spec in (tools or ())
        ) == self.step:
            self.remaining -= 1
            self.injected += 1
            raise llm_failure(self.kind, provider=self.inner.name)
        return self.inner.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def close(self) -> None:
        self.inner.close()

    def describe(self) -> str:
        pending = " [fault pending]" if self.remaining > 0 else " [fault fired]"
        return f"{self.inner.describe()}{pending}"


class FaultyRegistry(ToolRegistry):
    """Wraps a registry and makes one tool fail its first ``times`` calls.

    Subclasses rather than delegates, so it *is* a ToolRegistry everywhere one
    is expected -- including inside the recovery ladder, which must not be able
    to tell that this one is rigged. The fault is a real exception raised from a
    real handler, so it travels the registry's actual containment and
    classification path rather than being stubbed in beside it.
    """

    def __init__(
        self,
        inner: ToolRegistry,
        *,
        tool: str = FAULTY_TOOL,
        times: int = 1,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.target = tool if tool in inner else (inner.names[0] if inner.names else "")
        self.remaining = times
        self.injected = 0
        # The default is a rate limit rather than a made-up tool bug, because
        # that is the tool failure that actually happens: `revise_text` calls a
        # model, so the provider's 429 surfaces as a *tool* failure and is the
        # one worth a retry. A ToolError can be passed instead to exercise the
        # other branch -- a handler that declined, which is fed back to the
        # agent rather than retried.
        self.error = error or RateLimitError(
            f"injected fault: the model call inside {self.target} was rate limited",
            provider="mock",
            status=429,
            retry_after_s=0.05,
        )
        for spec in inner.specs():
            entry = inner.get(spec.name)
            assert entry is not None
            handler = self._rig(entry.handler) if spec.name == self.target else entry.handler
            self.register(entry.spec, handler, terminal=entry.terminal)

    def _rig(self, original: Handler) -> Handler:
        def handler(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
            if self.remaining > 0:
                self.remaining -= 1
                self.injected += 1
                raise self.error
            return original(arguments, ctx)

        return handler


@dataclass
class FaultyMemory(MemoryStore):
    """A store whose reads always fail. Trips the manager's circuit breaker.

    Writes are left working on purpose: a half-broken store is the realistic
    case (a corrupt index, a locked reader) and is strictly harder on the
    harness than one that is uniformly down.
    """

    inner: MemoryStore
    reads_attempted: int = field(default=0, init=False)

    def save(self, record: MemoryRecord) -> str:
        return self.inner.save(record)

    def recall(self, query: str, **kwargs: t.Any) -> t.NoReturn:
        self.reads_attempted += 1
        raise OSError("injected fault: memory index is unreadable")

    def clear_session(self, session_id: str) -> int:
        return self.inner.clear_session(session_id)

    def list_sessions(self) -> list[str]:
        return self.inner.list_sessions()

    def stats(self) -> dict[str, t.Any]:
        return self.inner.stats()

    def close(self) -> None:
        self.inner.close()

    @property
    def describe(self) -> str:
        return f"{self.inner.describe} [fault injected: reads fail]"


__all__ = [
    "FAILURE_KINDS",
    "FAULTY_TOOL",
    "FAULT_STEPS",
    "FaultyProvider",
    "LLM_KINDS",
    "SIMULATED_TOKEN_BUDGET",
    "STACK_KINDS",
    "FaultyMemory",
    "FaultyRegistry",
    "llm_failure",
]
