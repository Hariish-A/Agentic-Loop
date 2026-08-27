"""One defined path per failure mode. No mode is left to chance.

Every entry below is a rung ladder, not a single answer: the cheap local
recovery is tried before the expensive one, and every ladder ends in a floor
that always succeeds. A harness whose last rung can itself fail has not defined
a path, it has moved the crash.

=============================  ================================================
failure mode                   path
=============================  ================================================
Provider rate-limited / 5xx    retry with backoff+jitter (:mod:`.retry`), then
                               fail over to the next provider in the chain
Provider unreachable / no key  fail over immediately -- no retry can help
Provider auth rejected         fail over; the key is wrong *for this provider*
Unparseable LLM output         forced tool schema -> local JSON salvage ->
                               one repair call -> safe read-only default action
Model answered with prose      same ladder: salvage the tool call out of the
                               prose, then repair, then default
Tool call failed               typed ``ErrorKind`` -> sanitise arguments and
                               retry once / back off on transient / route to an
                               alternative tool -> otherwise hand the error back
                               to the agent as an observation
Memory read or write failure   circuit breaker in ``memory/manager.py`` -> the
                               store behaves as ``NullMemory``, run continues
                               with ``degraded_memory=true``
Iteration cap hit              return the **best draft seen**, status
                               ``max_iterations_reached``
Token budget exhausted         graceful stop, best draft returned, status
                               ``budget_exhausted`` (see :mod:`.guardrails`)
=============================  ================================================

The two classes here are both *decorators over an existing interface*:
:class:`ResilientProvider` is an :class:`~..llm.base.LLMProvider`, and
:func:`resilient_act` has the signature of :func:`~..core.act.act`. Nothing in
``core/`` learns that a harness exists, which is what keeps the four steps
readable as the four steps.
"""

from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass, field

from ..config import AppConfig
from ..core.state import ActionResult, Decision, ErrorKind
from ..llm.base import LLMProvider, ToolChoice
from ..llm.parsing import salvage_json
from ..llm.types import (
    AuthError,
    LLMParseError,
    LLMResponse,
    Message,
    ProviderUnavailableError,
    TerminalLLMError,
    ToolCall,
    ToolSpec,
    Usage,
    user,
)
from ..tools.registry import ToolContext, ToolRegistry
from .retry import RetryExhausted, RetryPolicy, call_with_retry

#: Emitted to the trace, so every recovery is visible rather than silent.
Emit = t.Callable[..., None]

#: Read-only tools the harness may substitute for a broken one. A degraded
#: decision must never be one that rewrites the user's text.
SAFE_ALTERNATIVES = ("analyze_text", "score_against_rubric")

REPAIR_TEMPLATE = (
    "Your previous reply could not be used: {error}\n"
    "It must be a single tool call whose arguments are valid JSON matching the "
    "tool's schema. Do not explain, apologise, or add prose. Call exactly one "
    "tool now."
)


def _noop(*_args: t.Any, **_kwargs: t.Any) -> None:
    return None


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------


@dataclass
class ProviderChain:
    """Providers to try, in order, constructed lazily.

    Lazy because constructing a provider reads a key and can raise
    :class:`~..llm.types.ProviderUnavailableError`; a chain that built every
    link up front would fail on a backup that is never needed.
    """

    links: list[tuple[str, t.Callable[[], LLMProvider]]]
    _built: dict[str, LLMProvider] = field(default_factory=dict, init=False)

    @classmethod
    def of(cls, provider: LLMProvider) -> ProviderChain:
        """A single-link chain, for the common case of one configured backend."""
        return cls(links=[(provider.name, lambda: provider)])

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.links]

    def get(self, index: int) -> LLMProvider:
        name, factory = self.links[index]
        if name not in self._built:
            self._built[name] = factory()
        return self._built[name]

    def close(self) -> None:
        for provider in self._built.values():
            provider.close()


# ---------------------------------------------------------------------------
# Resilient provider
# ---------------------------------------------------------------------------


class ResilientProvider(LLMProvider):
    """Retry, repair and failover wrapped around a chain of providers.

    Failover is **sticky**: once the chain moves on, later calls start from the
    new provider rather than re-testing the dead one. Retrying a backend that
    just exhausted its retry budget costs the full backoff again on every
    subsequent call, and an agent loop makes three calls per iteration.
    """

    def __init__(
        self,
        chain: ProviderChain,
        *,
        policy: RetryPolicy,
        repair_attempts: int = 1,
        emit: Emit | None = None,
        sleep: t.Callable[[float], None] | None = None,
    ) -> None:
        if not chain.links:
            raise ValueError("a ResilientProvider needs at least one provider")
        self._chain = chain
        self._policy = policy
        self._repair_attempts = max(0, repair_attempts)
        self._emit = emit or _noop
        self._sleep_override = sleep
        self._index = 0
        #: Counters the runner folds into the RunResult.
        self.retries = 0
        self.failovers = 0
        self.repairs = 0
        self._sync_identity()

    # -- identity -----------------------------------------------------------

    def _sync_identity(self) -> None:
        active = self._chain.get(self._index)
        self.name = active.name
        self.model = active.model
        self.supports_tools = active.supports_tools

    def describe(self) -> str:
        active = self._chain.get(self._index)
        rest = self._chain.names[self._index + 1 :]
        tail = f" (fallbacks: {', '.join(rest)})" if rest else ""
        return f"{active.describe()}{tail}"

    def close(self) -> None:
        self._chain.close()

    # -- the call -----------------------------------------------------------

    def complete(
        self,
        messages: t.Sequence[Message],
        *,
        tools: t.Sequence[ToolSpec] | None = None,
        tool_choice: ToolChoice = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        errors: list[str] = []

        while self._index < len(self._chain.links):
            name = self._chain.names[self._index]
            try:
                provider = self._chain.get(self._index)
            except Exception as exc:  # noqa: BLE001 - an unbuildable link is a skip
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                self._advance(name, f"could not be constructed: {exc}")
                continue

            try:
                response = self._attempt(
                    provider, messages, tools, tool_choice, temperature, max_tokens
                )
            except (RetryExhausted, ProviderUnavailableError, AuthError) as exc:
                # All three mean "this backend is not going to serve this run".
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                self._advance(name, str(exc))
                continue
            except TerminalLLMError as exc:
                # A 404 is "this model id does not exist here" -- a deprecated
                # model is exactly what the backup provider is for. Any other
                # terminal error means our request is wrong, and failing over
                # would burn the whole chain to reproduce our own bug.
                if exc.status != 404:
                    raise
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                self._advance(name, str(exc))
                continue

            return response

        raise ProviderUnavailableError(
            "every provider in the chain failed: " + " | ".join(errors),
            provider=self._chain.names[-1] if self._chain.names else "",
        )

    def _attempt(
        self,
        provider: LLMProvider,
        messages: t.Sequence[Message],
        tools: t.Sequence[ToolSpec] | None,
        tool_choice: ToolChoice,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        """One provider's worth of effort: retry, then the repair ladder."""

        def issue(turns: t.Sequence[Message]) -> LLMResponse:
            kwargs: dict[str, t.Any] = {}
            if self._sleep_override is not None:
                kwargs["sleep"] = self._sleep_override
            value, _ = call_with_retry(
                lambda: provider.complete(
                    turns,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                self._policy,
                on_retry=lambda attempt: self._on_retry(provider.name, attempt),
                **kwargs,
            )
            return value

        turns = list(messages)
        wants_tool = bool(tools) and tool_choice not in (None, "none")

        for round_index in range(self._repair_attempts + 1):
            try:
                response = issue(turns)
            except LLMParseError as exc:
                problem = str(exc)
                salvaged = self._salvage(exc, tools)
                if salvaged is not None:
                    self._emit("repair", provider=provider.name, method="local_salvage")
                    self.repairs += 1
                    return salvaged
            else:
                if not wants_tool or response.has_tool_calls:
                    return response
                # Rung two: the model answered with prose. The tool call is
                # often sitting inside it, so try to lift it out before paying
                # for another round trip.
                salvaged = self._salvage_from_text(response, tools)
                if salvaged is not None:
                    self._emit("repair", provider=provider.name, method="local_salvage")
                    self.repairs += 1
                    return salvaged
                problem = "the reply contained prose but no tool call"

            if round_index >= self._repair_attempts:
                raise LLMParseError(
                    f"{provider.name}: unusable reply after "
                    f"{self._repair_attempts} repair attempt(s): {problem}",
                    provider=provider.name,
                )

            # Rung three: one repair round trip, stating exactly what was wrong.
            self.repairs += 1
            self._emit(
                "repair", provider=provider.name, method="repair_call", error=problem
            )
            turns = [*messages, user(REPAIR_TEMPLATE.format(error=problem))]

        raise AssertionError("unreachable: the repair loop always returns or raises")

    # -- salvage ------------------------------------------------------------

    def _salvage(
        self, exc: LLMParseError, tools: t.Sequence[ToolSpec] | None
    ) -> LLMResponse | None:
        """Recover a tool call from the raw text of a parse failure."""
        parsed = salvage_json(exc.raw)
        if not isinstance(parsed, dict) or not tools:
            return None
        return _response_from_arguments(parsed, tools, provider=self.name, raw=exc.raw)

    def _salvage_from_text(
        self, response: LLMResponse, tools: t.Sequence[ToolSpec] | None
    ) -> LLMResponse | None:
        """Recover a tool call from a prose reply that embedded JSON."""
        parsed = salvage_json(response.text)
        if not isinstance(parsed, dict) or not tools:
            return None
        rebuilt = _response_from_arguments(parsed, tools, provider=self.name, raw=response.text)
        if rebuilt is None:
            return None
        return LLMResponse(
            text=response.text,
            tool_calls=rebuilt.tool_calls,
            usage=response.usage,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            latency_ms=response.latency_ms,
            raw=response.raw,
        )

    # -- bookkeeping --------------------------------------------------------

    def _on_retry(self, provider: str, attempt: t.Any) -> None:
        self.retries += 1
        self._emit(
            "retry",
            provider=provider,
            attempt=attempt.attempt,
            delay_s=round(attempt.delay_s, 3),
            error=str(attempt.error),
            error_type=attempt.error_type,
            honoured_retry_after=attempt.honoured_retry_after,
        )

    def _advance(self, failed: str, reason: str) -> None:
        self._index += 1
        self.failovers += 1
        if self._index < len(self._chain.links):
            self._sync_identity()
            self._emit(
                "failover", **{"from": failed, "to": self._chain.names[self._index]},
                reason=reason,
            )
        else:
            self._emit("failover", **{"from": failed, "to": None}, reason=reason)


def _response_from_arguments(
    parsed: dict[str, t.Any],
    tools: t.Sequence[ToolSpec],
    *,
    provider: str,
    raw: str,
) -> LLMResponse | None:
    """Turn a salvaged JSON object into a tool call, when it names a real tool.

    Two shapes are accepted: the arguments themselves (attributed to the only
    offered tool, which is the case for the judge and Reflect), and an
    OpenAI-ish ``{"name": ..., "arguments": {...}}`` envelope.
    """
    name = str(parsed.get("name") or parsed.get("tool") or "").strip()
    arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else None

    known = {spec.name for spec in tools}
    if name in known:
        return _tool_response(name, arguments or {}, provider=provider, raw=raw)
    if len(tools) == 1 and arguments is None and not name:
        # A single offered tool means an un-named argument blob is unambiguous.
        return _tool_response(tools[0].name, parsed, provider=provider, raw=raw)
    return None


def _tool_response(
    name: str, arguments: dict[str, t.Any], *, provider: str, raw: str
) -> LLMResponse:
    import json

    return LLMResponse(
        text="",
        tool_calls=(
            ToolCall(
                id=f"salvaged_{name}",
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments, default=str),
            ),
        ),
        usage=Usage.estimate(len(raw), len(raw)),
        provider=provider,
        finish_reason="salvaged",
        raw={"salvaged": True},
    )


# ---------------------------------------------------------------------------
# Tool failure recovery
# ---------------------------------------------------------------------------


@dataclass
class ToolRecovery:
    """The Act seam with a recovery ladder attached.

    Substituted for :func:`~..core.act.act` by the runner. Keeping the same
    signature is what lets ``core/act.py`` stay four lines of dispatch: the
    recovery policy is *here*, and Act never learns it exists.
    """

    policy: RetryPolicy
    emit: Emit = _noop
    sleep: t.Callable[[float], None] = time.sleep
    recoveries: int = 0

    def __call__(
        self, decision: Decision, registry: ToolRegistry, ctx: ToolContext
    ) -> tuple[ActionResult, Usage]:
        from ..core.act import act

        result, usage = act(decision, registry, ctx)
        if result.ok:
            return result, usage

        attempt = self._plan(result, registry)
        if attempt is None:
            # A decision, not an omission: this failure is one the agent should
            # read and react to. Traced so that "the harness did nothing" and
            # "the harness chose to do nothing" are distinguishable afterwards.
            self.emit(
                "tool_recovery",
                action=result.action,
                method="fed_back_as_observation",
                error=result.error,
                error_kind=result.error_kind.value,
                retry_as=None,
            )
            return result, usage

        retry_decision, method = attempt
        self.emit(
            "tool_recovery",
            action=result.action,
            method=method,
            error=result.error,
            retry_as=retry_decision.action,
        )

        retried, extra = act(retry_decision, registry, ctx)
        total = usage + extra
        if retried.ok:
            self.recoveries += 1
            return (
                _annotate(retried, retry_count=1, recovered=True),
                total,
            )
        # Both attempts failed. Report the *original* error: it describes what
        # the agent actually asked for, which is what the agent must react to.
        return _annotate(result, retry_count=1), total

    def _plan(
        self, result: ActionResult, registry: ToolRegistry
    ) -> tuple[Decision, str] | None:
        """Choose one recovery move, or ``None`` to hand the error back."""
        kind = result.error_kind

        if kind is ErrorKind.TRANSIENT and self.policy.max_attempts > 1:
            self.sleep(self.policy.backoff(1))
            return (
                Decision(action=result.action, arguments=dict(result.arguments),
                         thought="[harness] transient tool failure; one retry after backoff"),
                "backoff_retry",
            )

        if kind is ErrorKind.VALIDATION:
            sanitised = _sanitise(result, registry)
            if sanitised is not None:
                return (
                    Decision(action=result.action, arguments=sanitised,
                             thought="[harness] dropped the arguments the schema rejected"),
                    "sanitised_arguments",
                )
            return None

        if kind is ErrorKind.UNKNOWN_TOOL:
            alternative = _alternative(result.action, registry)
            if alternative is not None:
                return (
                    Decision(action=alternative, arguments={},
                             thought=f"[harness] {result.action!r} does not exist; "
                                     f"substituting the read-only {alternative}"),
                    "alternative_tool",
                )
            return None

        # RECOVERABLE and TERMINAL both go back to the agent unchanged. A
        # handler that declined for a stated reason has already said the useful
        # thing, and repeating the call cannot improve on it.
        return None


def _annotate(result: ActionResult, *, retry_count: int, recovered: bool = False) -> ActionResult:
    from dataclasses import replace

    return replace(result, retry_count=retry_count, recovered=recovered)


def _sanitise(result: ActionResult, registry: ToolRegistry) -> dict[str, t.Any] | None:
    """Drop the arguments the schema rejected, keeping the rest.

    Only ever removes. Guessing a *replacement* value would put words in the
    agent's mouth and hide the mistake from the trace; dropping an argument
    lets the handler's own default apply, which is a documented behaviour.
    """
    entry = registry.get(result.action)
    if entry is None:
        return None
    schema = entry.validation_schema
    properties: dict[str, t.Any] = schema.get("properties", {})
    required = set(schema.get("required", []))

    from ..tools.registry import validate_arguments

    kept = {
        name: value
        for name, value in result.arguments.items()
        if name in properties and not validate_arguments(
            {"type": "object", "properties": {name: properties[name]}, "required": []},
            {name: value},
        )
    }
    if kept == dict(result.arguments):
        return None  # nothing was droppable, so a retry would fail identically
    if required - set(kept):
        return None  # dropping would leave a required argument missing
    return kept


def _alternative(missing: str, registry: ToolRegistry) -> str | None:
    """Nearest real tool to a hallucinated name, else a safe read-only one."""
    import difflib

    close = difflib.get_close_matches(missing, registry.names, n=1, cutoff=0.75)
    if close:
        return close[0]
    for candidate in SAFE_ALTERNATIVES:
        if candidate in registry:
            return candidate
    return None


def build_provider_chain(
    config: AppConfig,
    primary: LLMProvider,
    *,
    extra: t.Sequence[tuple[str, t.Callable[[], LLMProvider]]] = (),
) -> ProviderChain:
    """The configured failover chain, with an already-built primary in front."""
    links: list[tuple[str, t.Callable[[], LLMProvider]]] = [(primary.name, lambda: primary)]
    seen = {primary.name}
    for name, factory in extra:
        if name not in seen:
            links.append((name, factory))
            seen.add(name)
    return ProviderChain(links=links)


__all__ = [
    "REPAIR_TEMPLATE",
    "SAFE_ALTERNATIVES",
    "ProviderChain",
    "ResilientProvider",
    "ToolRecovery",
    "build_provider_chain",
]
