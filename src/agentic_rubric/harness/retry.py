"""Exponential backoff with jitter, for LLM calls and for tool calls.

The whole module rests on one decision made back in Phase 0: the error taxonomy
in :mod:`..llm.types` is split by *what the caller should do*, not by status
code. So "should I retry this?" is an ``isinstance`` check rather than a table
of HTTP codes, and this file stays short enough to audit.

Three failure modes it defends against
--------------------------------------

**Rate limits (HTTP 429).** The single most likely failure on a free tier.
Answered with backoff, and with ``Retry-After`` honoured in preference to our
own computed delay -- the provider knows when its window resets and we do not.

**Transient 5xx and timeouts.** Answered with the same backoff.

**Synchronised retry storms.** Answered with jitter. Without it, every client
that hit the same rate limit retries at the same instant and rebuilds the
spike that caused it. ``full`` jitter (sleep a uniform sample from ``[0, d]``)
spreads the herd best; ``equal`` (``d/2 + U(0, d/2)``) trades some spread for a
guaranteed minimum wait. Both are offered because which one is right depends on
how many clients share the quota, and that is a deployment fact, not a code
fact. See Amazon's "Exponential Backoff and Jitter" (2015).

What is deliberately **not** retried:

* :class:`~..llm.types.TerminalLLMError` -- a bad key or a malformed request is
  not going to fix itself, and four attempts turn one clear error into a
  30-second wait for the same error.
* :class:`~..llm.types.ProviderUnavailableError` -- nothing is listening. That
  is the trigger for failover to the next provider, not for a retry.
* :class:`~..llm.types.LLMParseError` -- the call *succeeded*; the payload was
  unusable. Answered by the repair ladder in :mod:`.fallbacks`, because a plain
  retry re-samples the same prompt and usually reproduces the same mistake.

Sleeping and randomness are injected, so the tests assert real bounds on the
delay sequence instead of waiting through it.
"""

from __future__ import annotations

import random
import time
import typing as t
from dataclasses import dataclass

from ..config import RetryConfig
from ..llm.types import (
    LLMError,
    LLMParseError,
    ProviderUnavailableError,
    RetryableLLMError,
)

T = t.TypeVar("T")

Sleeper = t.Callable[[float], None]
Rng = t.Callable[[float, float], float]

JITTER_MODES = ("full", "equal", "none")


class RetryExhausted(LLMError):
    """Every attempt failed. Carries the last error and the attempt count."""

    def __init__(self, message: str, *, last: BaseException, attempts: int) -> None:
        super().__init__(message, provider=getattr(last, "provider", ""))
        self.last = last
        self.attempts = attempts


@dataclass(frozen=True)
class RetryAttempt:
    """One retry decision, handed to the ``on_retry`` hook for the trace."""

    attempt: int  # 1-based index of the attempt that just failed
    delay_s: float
    error: BaseException
    honoured_retry_after: bool = False

    @property
    def error_type(self) -> str:
        return type(self.error).__name__


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts, how long between them, and what counts as retryable."""

    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: str = "full"
    max_retry_after_s: float = 60.0
    label: str = "llm"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.jitter not in JITTER_MODES:
            raise ValueError(f"unknown jitter mode {self.jitter!r}; known: {JITTER_MODES}")

    # -- construction -------------------------------------------------------

    @classmethod
    def for_llm(cls, config: RetryConfig) -> RetryPolicy:
        return cls(
            max_attempts=config.max_attempts,
            base_delay_s=config.base_delay_s,
            max_delay_s=config.max_delay_s,
            jitter=config.jitter,
            max_retry_after_s=config.max_retry_after_s,
            label="llm",
        )

    @classmethod
    def for_tool(cls, config: RetryConfig) -> RetryPolicy:
        """Tighter than the LLM policy, and that is the point.

        A tool that failed usually failed for a reason a second identical call
        will reproduce -- bad arguments, a missing criterion, an empty diff. The
        exception is an LLM-backed tool hitting a rate limit, which is what the
        one extra attempt is for.
        """
        return cls(
            max_attempts=config.tool_max_attempts,
            base_delay_s=config.tool_base_delay_s,
            max_delay_s=config.max_delay_s,
            jitter=config.jitter,
            max_retry_after_s=config.max_retry_after_s,
            label="tool",
        )

    # -- policy -------------------------------------------------------------

    def backoff(self, attempt: int) -> float:
        """Un-jittered delay after ``attempt`` failures. Exponential, capped."""
        raw = self.base_delay_s * (2 ** max(0, attempt - 1))
        return min(raw, self.max_delay_s)

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after_s: float | None = None,
        rng: Rng = random.uniform,
    ) -> tuple[float, bool]:
        """Delay before the next attempt, plus whether ``Retry-After`` won.

        The provider's hint takes precedence over our computed backoff, but is
        still capped: a header asking for twenty minutes should make the harness
        fail over, not make a run look like it hung.
        """
        if retry_after_s is not None and retry_after_s >= 0:
            return min(retry_after_s, self.max_retry_after_s), True

        ceiling = self.backoff(attempt)
        if self.jitter == "none":
            return ceiling, False
        if self.jitter == "equal":
            half = ceiling / 2.0
            return half + rng(0.0, half), False
        return rng(0.0, ceiling), False  # full jitter


def is_retryable(exc: BaseException) -> bool:
    """Should this exception be answered with another attempt?

    ``ProviderUnavailableError`` is checked first because it is a subclass of
    ``LLMError`` but emphatically not retryable *against the same provider* --
    it is the failover trigger.
    """
    if isinstance(exc, (ProviderUnavailableError, LLMParseError)):
        return False
    return isinstance(exc, RetryableLLMError) and exc.retryable


def retry_after_of(exc: BaseException) -> float | None:
    """The provider's own backoff hint, when it sent one."""
    value = getattr(exc, "retry_after_s", None)
    return float(value) if isinstance(value, (int, float)) else None


def call_with_retry(
    fn: t.Callable[[], T],
    policy: RetryPolicy,
    *,
    on_retry: t.Callable[[RetryAttempt], None] | None = None,
    retryable: t.Callable[[BaseException], bool] = is_retryable,
    sleep: Sleeper = time.sleep,
    rng: Rng = random.uniform,
) -> tuple[T, int]:
    """Call ``fn``, retrying transient failures. Returns ``(value, retries)``.

    Raises the original exception when it is terminal, and
    :class:`RetryExhausted` when a retryable one outlives the attempt budget --
    two different situations that the caller answers differently, so they get
    two different types rather than one flag.
    """
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(), attempt - 1
        except BaseException as exc:  # noqa: BLE001 - classified immediately below
            if not retryable(exc):
                raise
            last = exc
            if attempt >= policy.max_attempts:
                break
            delay, honoured = policy.delay_for(
                attempt, retry_after_s=retry_after_of(exc), rng=rng
            )
            if on_retry is not None:
                on_retry(
                    RetryAttempt(
                        attempt=attempt,
                        delay_s=delay,
                        error=exc,
                        honoured_retry_after=honoured,
                    )
                )
            sleep(delay)

    assert last is not None  # the loop only breaks after assigning `last`
    raise RetryExhausted(
        f"{policy.label} call failed after {policy.max_attempts} attempt(s): {last}",
        last=last,
        attempts=policy.max_attempts,
    ) from last


__all__ = [
    "JITTER_MODES",
    "RetryAttempt",
    "RetryExhausted",
    "RetryPolicy",
    "Rng",
    "Sleeper",
    "call_with_retry",
    "is_retryable",
    "retry_after_of",
]
