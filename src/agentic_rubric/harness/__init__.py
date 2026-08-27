"""Production scaffolding around the loop: retry, fallbacks, guardrails, tracing.

Nothing in :mod:`..core` imports this package. The dependency runs one way --
the harness wraps the loop, the loop never reaches back -- which is what lets
the four cognitive steps be read without wading through error handling, and
lets the harness be tested without running an agent.

Entry point: :class:`~.runner.Runner`.
"""

from .fallbacks import ProviderChain, ResilientProvider, ToolRecovery
from .faults import (
    FAILURE_KINDS,
    FAULT_STEPS,
    FaultyMemory,
    FaultyProvider,
    FaultyRegistry,
    llm_failure,
)
from .guardrails import Guardrails
from .loop_detect import StuckDetector, StuckVerdict
from .retry import RetryPolicy, call_with_retry, is_retryable
from .runner import Runner, RunnerReport

__all__ = [
    "FAILURE_KINDS",
    "FAULT_STEPS",
    "FaultyMemory",
    "FaultyProvider",
    "FaultyRegistry",
    "Guardrails",
    "ProviderChain",
    "ResilientProvider",
    "RetryPolicy",
    "Runner",
    "RunnerReport",
    "StuckDetector",
    "StuckVerdict",
    "ToolRecovery",
    "call_with_retry",
    "is_retryable",
    "llm_failure",
]
