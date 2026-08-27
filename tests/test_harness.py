"""Retry, fallbacks, guardrails, stuck detection, tracing and the runner.

Every test here runs offline against ``MockProvider``, and every sleep is
injected, so the suite asserts real backoff *bounds* in milliseconds instead of
waiting through thirty seconds of exponential delay.

The assertions are deliberately behavioural. "It did not raise" is the weakest
possible claim about a harness, and Milestone 2 has the scar to prove it: a bug
that silently disabled memory was caught by a test asserting behaviour
*changed*, not by anything crashing.
"""

from __future__ import annotations

import json
import typing as t
from pathlib import Path

import pytest

from agentic_rubric.config import AppConfig, load_config
from agentic_rubric.core.loop import StopSignal
from agentic_rubric.core.rubric import CriterionScore, Rubric, ScoreCard
from agentic_rubric.core.state import (
    ActionResult,
    Decision,
    ErrorKind,
    IterationRecord,
    LoopState,
    Observation,
    Reflection,
    RunStatus,
    Workspace,
)
from agentic_rubric.harness.fallbacks import (
    ProviderChain,
    ResilientProvider,
    ToolRecovery,
)
from agentic_rubric.harness.faults import FaultyMemory, FaultyRegistry, llm_failure
from agentic_rubric.harness.guardrails import Guardrails
from agentic_rubric.harness.loop_detect import StuckDetector, draft_fingerprint
from agentic_rubric.harness.retry import (
    RetryExhausted,
    RetryPolicy,
    call_with_retry,
    is_retryable,
)
from agentic_rubric.harness.runner import Runner
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder
from agentic_rubric.llm.mock import MockCall, MockProvider, MockTurn, tool_call
from agentic_rubric.llm.types import (
    AuthError,
    BadRequestError,
    LLMParseError,
    ProviderUnavailableError,
    RateLimitError,
    ToolSpec,
    TransientServerError,
    Usage,
    user,
)
from agentic_rubric.memory.base import MemoryRecord, NullMemory
from agentic_rubric.memory.manager import CIRCUIT_BREAKER_THRESHOLD, MemoryManager
from agentic_rubric.observability.logger import JsonFormatter, redact, scrub_text
from agentic_rubric.observability.trace import RunTracer, fanout, read_trace
from agentic_rubric.tools.definitions import build_registry
from agentic_rubric.tools.registry import ToolContext, ToolError

CONFIG = "config/config.yaml"
ESSAY = "config/rubrics/essay_argumentative.yaml"


@pytest.fixture
def config() -> AppConfig:
    return load_config(CONFIG)


@pytest.fixture
def rubric() -> Rubric:
    return Rubric.from_yaml(ESSAY)


@pytest.fixture
def draft() -> str:
    return Path("samples/weak_essay.txt").read_text(encoding="utf-8")


class Clock:
    """A monotonic clock the test drives by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def collect(sink: list[float]) -> t.Callable[[float], None]:
    return sink.append


# ===========================================================================
# RETRY  (M3-1)
# ===========================================================================


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter="none")
    assert [policy.backoff(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 10.0]


def test_full_jitter_stays_inside_the_backoff_envelope() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter="full")
    for attempt in range(1, 6):
        ceiling = policy.backoff(attempt)
        for _ in range(200):
            delay, honoured = policy.delay_for(attempt)
            assert 0.0 <= delay <= ceiling
            assert honoured is False


def test_equal_jitter_guarantees_a_minimum_wait() -> None:
    """The whole point of `equal` over `full`: it never sleeps ~0 seconds."""
    policy = RetryPolicy(base_delay_s=4.0, max_delay_s=30.0, jitter="equal")
    delays = [policy.delay_for(1)[0] for _ in range(200)]
    assert all(2.0 <= d <= 4.0 for d in delays)
    assert max(delays) - min(delays) > 0.5  # still spread, not a constant


def test_jitter_actually_spreads_the_herd() -> None:
    policy = RetryPolicy(base_delay_s=2.0, jitter="full")
    delays = {round(policy.delay_for(2)[0], 4) for _ in range(200)}
    assert len(delays) > 150  # synchronised clients would collide here


def test_retry_after_beats_computed_backoff_but_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_retry_after_s=60.0, jitter="full")
    assert policy.delay_for(1, retry_after_s=7.5) == (7.5, True)
    # A provider asking for twenty minutes should trigger failover, not a hang.
    assert policy.delay_for(1, retry_after_s=1200.0) == (60.0, True)


def test_retryable_and_terminal_are_split_by_caller_action() -> None:
    assert is_retryable(RateLimitError("429")) is True
    assert is_retryable(TransientServerError("503")) is True
    assert is_retryable(AuthError("401")) is False
    assert is_retryable(BadRequestError("400")) is False
    # Not retryable against this provider: it is the failover trigger.
    assert is_retryable(ProviderUnavailableError("refused")) is False
    # The call succeeded; a plain retry re-samples the same prompt.
    assert is_retryable(LLMParseError("bad json")) is False


def test_a_transient_failure_is_retried_then_succeeds() -> None:
    calls: list[int] = []
    slept: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientServerError("503")
        return "ok"

    value, retries = call_with_retry(
        flaky, RetryPolicy(max_attempts=4, base_delay_s=0.01), sleep=collect(slept)
    )
    assert (value, retries) == ("ok", 2)
    assert len(slept) == 2


def test_a_terminal_failure_is_raised_immediately_without_sleeping() -> None:
    slept: list[float] = []
    attempts: list[int] = []

    def bad_key() -> str:
        attempts.append(1)
        raise AuthError("401 invalid api key")

    with pytest.raises(AuthError):
        call_with_retry(bad_key, RetryPolicy(max_attempts=4), sleep=collect(slept))
    assert attempts == [1]  # four attempts would turn one clear error into a wait
    assert slept == []


def test_exhausting_the_budget_raises_a_distinct_error() -> None:
    slept: list[float] = []
    with pytest.raises(RetryExhausted) as caught:
        call_with_retry(
            lambda: (_ for _ in ()).throw(RateLimitError("429")),
            RetryPolicy(max_attempts=3, base_delay_s=0.01),
            sleep=collect(slept),
        )
    assert caught.value.attempts == 3
    assert isinstance(caught.value.last, RateLimitError)
    assert len(slept) == 2  # n attempts means n-1 waits


def test_tool_policy_is_tighter_than_the_llm_policy(config: AppConfig) -> None:
    llm = RetryPolicy.for_llm(config.retry)
    tool = RetryPolicy.for_tool(config.retry)
    assert tool.max_attempts < llm.max_attempts
    assert tool.base_delay_s < llm.base_delay_s


# ===========================================================================
# PROVIDER FALLBACKS  (M3-2)
# ===========================================================================


def scripted(name: str, *turns: MockTurn) -> MockProvider:
    return MockProvider(list(turns), name=name, repeat_last=True)


SPEC = ToolSpec(
    name="submit",
    description="submit an answer",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}},
)


def resilient(chain: ProviderChain, **kwargs: t.Any) -> ResilientProvider:
    kwargs.setdefault("policy", RetryPolicy(max_attempts=3, base_delay_s=0.001))
    kwargs.setdefault("sleep", lambda _s: None)
    return ResilientProvider(chain, **kwargs)


def test_a_rate_limited_call_is_retried_and_then_succeeds() -> None:
    provider = scripted(
        "primary",
        MockTurn(raises=RateLimitError("429", retry_after_s=0.0)),
        MockTurn(text="recovered"),
    )
    events: list[str] = []
    wrapped = resilient(ProviderChain.of(provider), emit=lambda e, **_: events.append(e))

    assert wrapped.complete([user("hi")]).text == "recovered"
    assert wrapped.retries == 1
    assert events == ["retry"]


def test_an_unavailable_provider_fails_over_to_the_next_one() -> None:
    dead = MockProvider(responder=lambda _c: ProviderUnavailableError("refused"), name="dead")
    alive = scripted("backup", MockTurn(text="from the backup"))
    events: list[tuple[str, dict[str, t.Any]]] = []
    wrapped = resilient(
        ProviderChain(links=[("dead", lambda: dead), ("backup", lambda: alive)]),
        emit=lambda e, **p: events.append((e, p)),
    )

    assert wrapped.complete([user("hi")]).text == "from the backup"
    assert wrapped.failovers == 1
    assert [e for e, _ in events] == ["failover"]
    assert events[0][1]["from"] == "dead" and events[0][1]["to"] == "backup"


def test_failover_is_sticky_across_later_calls() -> None:
    """A backend that just died must not be re-tried on every subsequent call."""
    attempts: list[int] = []

    def refuse(_call: MockCall) -> Exception:
        attempts.append(1)
        return ProviderUnavailableError("refused")

    dead = MockProvider(responder=refuse, name="dead")
    alive = scripted("backup", MockTurn(text="ok"))
    wrapped = resilient(ProviderChain(links=[("dead", lambda: dead), ("backup", lambda: alive)]))

    for _ in range(4):
        assert wrapped.complete([user("hi")]).text == "ok"
    assert attempts == [1]  # asked once, never again
    assert wrapped.failovers == 1


def test_a_bad_key_fails_over_rather_than_ending_the_run() -> None:
    dead = MockProvider(responder=lambda _c: AuthError("401", status=401), name="dead")
    alive = scripted("backup", MockTurn(text="ok"))
    wrapped = resilient(ProviderChain(links=[("dead", lambda: dead), ("backup", lambda: alive)]))
    assert wrapped.complete([user("hi")]).text == "ok"


def test_a_malformed_request_is_raised_not_failed_over() -> None:
    """A 400 is our bug. Burning the chain to reproduce it helps nobody."""
    broken = MockProvider(
        responder=lambda _c: BadRequestError("400 unsupported parameter", status=400), name="a"
    )
    backup = scripted("b", MockTurn(text="unused"))
    wrapped = resilient(ProviderChain(links=[("a", lambda: broken), ("b", lambda: backup)]))
    with pytest.raises(BadRequestError):
        wrapped.complete([user("hi")])
    assert wrapped.failovers == 0


def test_a_missing_model_does_fail_over() -> None:
    """404 is the one terminal status a backup can genuinely answer."""
    gone = MockProvider(
        responder=lambda _c: BadRequestError("404 model not found", status=404), name="a"
    )
    backup = scripted("b", MockTurn(text="ok"))
    wrapped = resilient(ProviderChain(links=[("a", lambda: gone), ("b", lambda: backup)]))
    assert wrapped.complete([user("hi")]).text == "ok"


def test_unparseable_output_is_salvaged_locally_before_paying_for_a_repair() -> None:
    provider = scripted(
        "p",
        MockTurn(raises=LLMParseError("bad json", raw='Sure! ```json\n{"value": "42"}\n```')),
    )
    wrapped = resilient(ProviderChain.of(provider))
    response = wrapped.complete([user("hi")], tools=[SPEC], tool_choice="required")

    call = response.first_tool_call()
    assert (call.name, call.arguments) == ("submit", {"value": "42"})
    assert provider.call_count == 1  # salvage is rung two; no second round trip


def test_prose_instead_of_a_tool_call_triggers_one_repair_round_trip() -> None:
    provider = scripted(
        "p",
        MockTurn(text="I think we should probably submit something."),
        MockTurn(tool_calls=(tool_call("submit", value="ok"),)),
    )
    events: list[str] = []
    wrapped = resilient(ProviderChain.of(provider), emit=lambda e, **_: events.append(e))

    response = wrapped.complete([user("hi")], tools=[SPEC], tool_choice="required")
    assert response.first_tool_call().name == "submit"
    assert wrapped.repairs == 1
    assert events == ["repair"]
    # The repair prompt states what was wrong, appended to the original turns.
    assert "could not be used" in provider.calls[1].messages[-1].content


def test_the_repair_ladder_gives_up_after_the_configured_attempts() -> None:
    provider = MockProvider(responder=lambda _c: MockTurn(text="still prose"), name="p")
    wrapped = resilient(ProviderChain.of(provider), repair_attempts=1)
    with pytest.raises(LLMParseError):
        wrapped.complete([user("hi")], tools=[SPEC], tool_choice="required")
    assert provider.call_count == 2  # the original plus one repair, then stop


def test_an_exhausted_chain_reports_every_reason() -> None:
    a = MockProvider(responder=lambda _c: ProviderUnavailableError("no key"), name="a")
    b = MockProvider(responder=lambda _c: ProviderUnavailableError("refused"), name="b")
    wrapped = resilient(ProviderChain(links=[("a", lambda: a), ("b", lambda: b)]))
    with pytest.raises(ProviderUnavailableError) as caught:
        wrapped.complete([user("hi")])
    assert "no key" in str(caught.value) and "refused" in str(caught.value)


# ===========================================================================
# TOOL FALLBACKS  (M3-2)
# ===========================================================================


def tool_ctx(config: AppConfig, rubric: Rubric, draft: str) -> ToolContext:
    return ToolContext(
        config=config,
        rubric=rubric,
        workspace=Workspace(draft=draft),
        llm=MockProvider([MockTurn(text="unused")], repeat_last=True),
    )


def recovery(**kwargs: t.Any) -> ToolRecovery:
    kwargs.setdefault("policy", RetryPolicy(max_attempts=2, base_delay_s=0.0, label="tool"))
    kwargs.setdefault("sleep", lambda _s: None)
    return ToolRecovery(**kwargs)


def test_the_registry_classifies_a_contained_exception(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    registry = FaultyRegistry(build_registry(rubric), tool="analyze_text")
    result = registry.dispatch(
        Decision(action="analyze_text"), tool_ctx(config, rubric, draft)
    )
    assert result.ok is False
    assert result.error_kind is ErrorKind.TRANSIENT  # a rate limit inside the handler


def test_a_transient_tool_failure_is_retried_once_and_recovers(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    registry = FaultyRegistry(build_registry(rubric), tool="analyze_text", times=1)
    events: list[dict[str, t.Any]] = []
    act = recovery(emit=lambda _e, **p: events.append(p))

    result, _ = act(
        Decision(action="analyze_text"), registry, tool_ctx(config, rubric, draft)
    )
    assert result.ok is True
    assert result.recovered is True and result.retry_count == 1
    assert act.recoveries == 1
    assert events[0]["method"] == "backoff_retry"


def test_a_declined_tool_is_handed_back_to_the_agent_not_retried(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    """A handler that declined has already said the useful thing."""
    registry = FaultyRegistry(
        build_registry(rubric),
        tool="analyze_text",
        times=5,
        error=ToolError("the draft is empty", recoverable=True),
    )
    events: list[dict[str, t.Any]] = []
    act = recovery(emit=lambda _e, **p: events.append(p))

    result, _ = act(
        Decision(action="analyze_text"), registry, tool_ctx(config, rubric, draft)
    )
    assert result.ok is False
    assert result.error_kind is ErrorKind.RECOVERABLE
    assert registry.injected == 1  # exactly one attempt, no retry
    assert events[0]["method"] == "fed_back_as_observation"
    assert act.recoveries == 0


def test_bad_arguments_are_sanitised_and_retried(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    registry = build_registry(rubric)
    events: list[dict[str, t.Any]] = []
    act = recovery(emit=lambda _e, **p: events.append(p))

    decision = Decision(action="analyze_text", arguments={"nonsense": True})
    result, _ = act(decision, registry, tool_ctx(config, rubric, draft))

    assert result.ok is True and result.recovered is True
    assert events[0]["method"] == "sanitised_arguments"


def test_a_hallucinated_tool_is_routed_to_a_real_one(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    registry = build_registry(rubric)
    events: list[dict[str, t.Any]] = []
    act = recovery(emit=lambda _e, **p: events.append(p))

    result, _ = act(
        Decision(action="analyse_text"), registry, tool_ctx(config, rubric, draft)
    )
    assert result.ok is True
    assert events[0]["retry_as"] == "analyze_text"


def test_recovery_never_substitutes_a_tool_that_rewrites_the_draft(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    registry = build_registry(rubric)
    ctx = tool_ctx(config, rubric, draft)
    events: list[dict[str, t.Any]] = []
    act = recovery(emit=lambda _e, **p: events.append(p))

    act(Decision(action="totally_made_up_tool"), registry, ctx)
    assert events[0]["retry_as"] in ("analyze_text", "score_against_rubric")
    assert ctx.workspace.draft == draft  # untouched


# ===========================================================================
# GUARDRAILS  (M3-6)
# ===========================================================================


def make_state(rubric: Rubric, draft: str, **kwargs: t.Any) -> LoopState:
    return LoopState(
        rubric=rubric, workspace=Workspace(draft=draft), target_score=85.0, **kwargs
    )


def make_record(iteration: int, action: str, **arguments: t.Any) -> IterationRecord:
    decision = Decision(action=action, arguments=arguments)
    return IterationRecord(
        iteration=iteration,
        observation=t.cast(Observation, None),
        decision=decision,
        result=ActionResult(action=action, ok=True, summary="done"),
        reflection=Reflection(task_complete=False, reason="continuing"),
    )


def test_the_token_budget_stops_the_run_gracefully(rubric: Rubric, draft: str) -> None:
    config = load_config(CONFIG, overrides={"guardrails.token_budget": 1000})
    guardrails = Guardrails(config)
    state = make_state(rubric, draft)

    state.record_usage(Usage(input_tokens=600, output_tokens=100))
    assert guardrails.before_iteration(state) is None  # 70%: still fine

    state.record_usage(Usage(input_tokens=400, output_tokens=0))
    stop = guardrails.before_iteration(state)
    assert stop is not None
    assert stop.status is RunStatus.BUDGET_EXHAUSTED
    assert "1,100/1,000" in stop.reason
    assert guardrails.triggered == ["token_budget"]


def test_the_budget_warns_before_it_stops(rubric: Rubric, draft: str) -> None:
    config = load_config(
        CONFIG,
        overrides={"guardrails.token_budget": 1000, "guardrails.token_warn_ratio": 0.8},
    )
    events: list[str] = []
    guardrails = Guardrails(config, emit=lambda e, **_: events.append(e))
    state = make_state(rubric, draft)

    state.record_usage(Usage(input_tokens=850))
    assert guardrails.before_iteration(state) is None
    assert events == ["budget_warning"]
    assert any("token budget 85% spent" in note for note in state.notes)

    # The warning fires once, not on every subsequent check.
    state.record_usage(Usage(input_tokens=10))
    guardrails.before_iteration(state)
    assert events == ["budget_warning"]


def test_the_wall_clock_limit_stops_the_run(rubric: Rubric, draft: str) -> None:
    config = load_config(CONFIG, overrides={"guardrails.wall_clock_timeout_s": 30})
    clock = Clock()
    guardrails = Guardrails(config, clock=clock)
    state = make_state(rubric, draft)

    assert guardrails.before_iteration(state) is None
    clock.advance(31)
    stop = guardrails.before_iteration(state)
    assert stop is not None and stop.status is RunStatus.TIMEOUT


def test_the_iteration_cap_is_enforced_outside_the_model(
    rubric: Rubric, draft: str
) -> None:
    config = load_config(CONFIG, overrides={"loop.max_iterations": 3})
    guardrails = Guardrails(config)
    state = make_state(rubric, draft)

    state.iteration = 2
    assert guardrails.before_iteration(state) is None
    state.iteration = 3
    stop = guardrails.before_iteration(state)
    assert stop is not None and stop.status is RunStatus.MAX_ITERATIONS_REACHED


def test_an_oversized_document_is_capped_at_ingestion(config: AppConfig) -> None:
    limited = config.with_overrides(
        guardrails=load_config(
            CONFIG, overrides={"guardrails.max_document_chars": 100}
        ).guardrails
    )
    guardrails = Guardrails(limited)
    check = guardrails.check_input("x" * 5000)
    assert len(check.text) == 100
    assert check.truncated and "ingestion cap" in check.note


# ===========================================================================
# STUCK DETECTION  (M3-7)
# ===========================================================================


def test_consecutive_identical_actions_are_stuck(rubric: Rubric, draft: str) -> None:
    detector = StuckDetector(repeat_threshold=3, score_epsilon=0.0)
    state = make_state(rubric, draft)

    assert detector.observe(state, make_record(1, "analyze_text")) is None
    assert detector.observe(state, make_record(2, "analyze_text")) is None
    verdict = detector.observe(state, make_record(3, "analyze_text"))
    assert verdict is not None and verdict.signal == "repeated_action"


def test_alternating_actions_are_not_stuck(rubric: Rubric, draft: str) -> None:
    """score -> revise -> score -> revise is a healthy loop, not a cycle."""
    detector = StuckDetector(repeat_threshold=3, score_epsilon=0.0)
    state = make_state(rubric, draft)
    for index in range(8):
        action = "score_against_rubric" if index % 2 == 0 else "revise_text"
        state.workspace.draft = f"{draft} revision {index}"
        assert detector.observe(state, make_record(index + 1, action)) is None


def test_a_draft_that_returns_to_an_earlier_state_is_stuck(
    rubric: Rubric, draft: str
) -> None:
    detector = StuckDetector(repeat_threshold=99, score_epsilon=0.0)
    state = make_state(rubric, draft)

    assert detector.observe(state, make_record(1, "revise_text", n=1)) is None
    state.workspace.draft = draft + " version B"
    assert detector.observe(state, make_record(2, "revise_text", n=2)) is None
    state.workspace.draft = draft  # undone
    verdict = detector.observe(state, make_record(3, "revise_text", n=3))
    assert verdict is not None and verdict.signal == "draft_cycle"


def test_an_unchanged_draft_alone_is_not_a_cycle(rubric: Rubric, draft: str) -> None:
    """Scoring and analysis change nothing; that must not read as spinning."""
    detector = StuckDetector(repeat_threshold=99, score_epsilon=0.0)
    state = make_state(rubric, draft)
    for index in range(5):
        assert detector.observe(state, make_record(index + 1, f"tool_{index}")) is None


def test_a_frozen_score_is_stuck(rubric: Rubric, draft: str) -> None:
    detector = StuckDetector(repeat_threshold=99, score_epsilon=0.5, score_window=3)
    state = make_state(rubric, draft)
    for index in range(3):
        state.workspace.record_score(
            ScoreCard.build(
                rubric, [CriterionScore(criterion_id=c.id, score=3) for c in rubric.criteria]
            )
        )
        state.workspace.draft = f"{draft} v{index}"
        verdict = detector.observe(state, make_record(index + 1, f"tool_{index}"))
    assert verdict is not None and verdict.signal == "score_plateau"


def test_the_fingerprint_ignores_whitespace_and_case() -> None:
    assert draft_fingerprint("Hello   World\n") == draft_fingerprint("hello world")
    assert draft_fingerprint("hello world") != draft_fingerprint("hello worlds")


# ===========================================================================
# LOGGING AND TRACING  (M3-3, M3-4)
# ===========================================================================


def test_secrets_are_redacted_by_key_and_by_pattern() -> None:
    payload = {
        "api_key": "gsk_liveKeyMaterial1234",
        "nested": {"Authorization": "Bearer abcdef1234567890"},
        "message": "call failed with Authorization: Bearer abcdef1234567890",
        "url": "https://api/v1?token=sk-abcdefghijklmnop",
        "safe": "nothing to see",
    }
    safe = redact(payload, ["api_key", "authorization"])
    assert safe["api_key"] == "[REDACTED]"
    assert safe["nested"]["Authorization"] == "[REDACTED]"
    assert "abcdef1234567890" not in safe["message"]
    assert "sk-abcdefghijklmnop" not in safe["url"]
    assert safe["safe"] == "nothing to see"


def test_redaction_truncates_long_fields_visibly() -> None:
    safe = redact({"draft": "x" * 5000}, [], max_chars=100)
    assert safe["draft"].startswith("x" * 100)
    assert "+4900 chars" in safe["draft"]


def test_the_json_formatter_carries_extra_fields() -> None:
    import logging

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "failover", None, None)
    record.__dict__.update({"provider": "groq", "api_key": "gsk_secret_material"})
    line = json.loads(JsonFormatter(["api_key"]).format(record))
    assert line["message"] == "failover"
    assert line["provider"] == "groq"
    assert line["api_key"] == "[REDACTED]"


def test_scrub_leaves_ordinary_text_alone() -> None:
    assert scrub_text("the score rose from 28.7% to 96.2%") == (
        "the score rose from 28.7% to 96.2%"
    )


def test_a_failing_subscriber_cannot_kill_the_run() -> None:
    seen: list[str] = []
    emit = fanout(
        lambda _e, _p: (_ for _ in ()).throw(RuntimeError("renderer bug")),
        lambda e, _p: seen.append(e),
    )
    emit("reason", {})
    assert seen == ["reason"]  # the second subscriber still ran


def test_the_tracer_writes_one_envelope_per_event(
    tmp_path: Path, config: AppConfig
) -> None:
    tracer = RunTracer(tmp_path, config=config, provider_name="mock")
    tracer("run_start", {"run_id": "run_test", "session_id": "sess_test"})
    tracer("act", {"iteration": 2, "action": "revise_text", "duration_ms": 12.5, "tokens": 40})
    tracer.close()

    rows = read_trace(tmp_path / "run_test" / "trace.jsonl")
    assert [r["event"] for r in rows] == ["run_start", "act"]
    act_row = rows[1]
    assert act_row["run_id"] == "run_test"
    assert act_row["session_id"] == "sess_test"  # carried from run_start
    assert (act_row["step"], act_row["tool"], act_row["tokens"]) == ("act", "revise_text", 40)
    assert act_row["duration_ms"] == 12.5
    assert set(act_row) >= {"iteration", "cost_est", "error", "retry_count", "detail"}


def test_tracing_can_be_switched_off_entirely(tmp_path: Path, config: AppConfig) -> None:
    tracer = RunTracer(tmp_path, config=config, enabled=False)
    tracer("run_start", {"run_id": "run_test"})
    assert list(tmp_path.iterdir()) == []


def test_a_trace_write_failure_does_not_raise(tmp_path: Path, config: AppConfig) -> None:
    tracer = RunTracer(tmp_path, config=config)
    tracer("run_start", {"run_id": "run_test"})
    tracer.close()
    tracer.write("act", {"iteration": 1})  # the handle is gone
    assert tracer.events_written == 1


def test_cost_is_zero_until_prices_are_configured(tmp_path: Path) -> None:
    free = load_config(CONFIG)
    priced = load_config(
        CONFIG, overrides={"llm.providers.groq.cost_per_1k_input": 0.15}
    )
    from agentic_rubric.observability.trace import estimate_cost

    assert estimate_cost(free, "groq", 10_000, 5_000) == 0.0
    assert estimate_cost(priced, "groq", 10_000, 0) == pytest.approx(1.5)


# ===========================================================================
# THE RUNNER, END TO END  (M3-8, M3-9)
# ===========================================================================


def build_runner(
    config: AppConfig, rubric: Rubric, **kwargs: t.Any
) -> Runner:
    responder = kwargs.pop("responder", None) or ScriptedAgentResponder(rubric=rubric)
    provider = kwargs.pop("provider", None) or MockProvider(responder=responder, name="mock")
    kwargs.setdefault("memory", NullMemory())
    kwargs.setdefault("trace", False)
    kwargs.setdefault("sleep", lambda _s: None)
    return Runner(config=config, rubric=rubric, provider=provider, **kwargs)


def test_the_runner_completes_a_clean_run(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    report = build_runner(config, rubric).run(draft)
    assert report.result.status is RunStatus.TARGET_REACHED
    assert report.result.iterations >= 3
    assert report.result.retry_count == 0 and report.result.failover_count == 0


def test_the_runner_recovers_from_an_injected_rate_limit(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    responder = ScriptedAgentResponder(rubric=rubric)
    responder.fail_on["judge"] = llm_failure("rate_limit")
    report = build_runner(config, rubric, responder=responder).run(draft)

    assert report.result.status is RunStatus.TARGET_REACHED
    assert report.result.retry_count == 1  # the failure happened and was absorbed


def test_the_runner_recovers_from_unparseable_output(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    responder = ScriptedAgentResponder(rubric=rubric)
    responder.fail_on["reason"] = llm_failure("bad_json")
    report = build_runner(config, rubric, responder=responder).run(draft)

    assert report.result.repair_count >= 1
    assert report.result.status.is_success


def test_the_runner_fails_over_mid_run(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    healthy = MockProvider(responder=ScriptedAgentResponder(rubric=rubric), name="mock")
    dead = MockProvider(responder=lambda _c: llm_failure("provider_down"), name="dead")
    chain = ProviderChain(links=[("dead", lambda: dead), ("mock", lambda: healthy)])

    report = build_runner(config, rubric, provider=dead, chain=chain).run(draft)
    assert report.result.failover_count == 1
    assert report.result.status.is_success


def test_the_budget_guardrail_returns_the_best_draft_seen(
    rubric: Rubric, draft: str
) -> None:
    config = load_config(CONFIG, overrides={"guardrails.token_budget": 900})
    report = build_runner(config, rubric).run(draft)

    assert report.result.status is RunStatus.BUDGET_EXHAUSTED
    assert report.result.best_draft  # work already paid for is not thrown away
    assert report.result.best_score is not None
    assert "budget exhausted" in " ".join(report.result.notes)


def test_memory_failure_degrades_instead_of_ending_the_run(
    config: AppConfig, rubric: Rubric, draft: str, tmp_path: Path
) -> None:
    class Writable(NullMemory):
        """Reads fail, writes work -- the realistic half-outage."""

        def __init__(self) -> None:
            self.saved = 0

        def save(self, record: MemoryRecord) -> str:
            self.saved += 1
            return "id"

    memory = MemoryManager(FaultyMemory(Writable()), config.memory)
    report = build_runner(config, rubric, memory=memory).run(draft)

    assert report.result.status.is_success  # the run finished anyway
    assert memory.degraded is True
    assert report.result.degraded_memory is True
    assert memory.failures_of("memory recall") >= CIRCUIT_BREAKER_THRESHOLD
    # And the run said so while it was happening, not only afterwards.
    later = report.result.records[-1].observation
    assert any("running without memory" in note for note in later.notes)


def test_a_working_write_does_not_reset_a_failing_read(config: AppConfig) -> None:
    """The bug `--simulate-failure memory_down` found: one shared counter meant
    successful writes kept resetting the read streak, so the breaker never
    opened and every iteration paid for a failing read."""

    class Writable(NullMemory):
        def save(self, record: MemoryRecord) -> str:
            return "id"

    memory = MemoryManager(FaultyMemory(Writable()), config.memory)
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        memory.recall("q")
        memory.save(MemoryRecord(kind="lesson", content="something worth keeping"))
    assert memory.degraded is True


def test_a_stuck_loop_is_detected_and_stopped(
    rubric: Rubric, draft: str
) -> None:
    """An agent that keeps making the same call gets stopped, not indulged."""
    config = load_config(
        CONFIG,
        overrides={"loop.max_iterations": 10, "guardrails.repeat_action_threshold": 3},
    )
    stubborn = MockProvider(
        responder=lambda call: MockTurn(
            tool_calls=(
                tool_call("analyze_text", thought="measuring again"),
            )
        )
        if any(spec.name == "analyze_text" for spec in call.tools)
        else MockTurn(tool_calls=(tool_call("submit_reflection", critique="", lesson=""),)),
        name="mock",
    )
    report = build_runner(config, rubric, provider=stubborn).run(draft)

    assert report.result.status is RunStatus.STUCK
    assert report.result.iterations == 3  # stopped early, not at the cap of 10
    assert "repeated_action" in " ".join(report.result.notes)


def test_the_runner_writes_a_trace_and_a_summary(
    config: AppConfig, rubric: Rubric, draft: str, tmp_path: Path
) -> None:
    traced = load_config(CONFIG, overrides={"logging.trace_dir": str(tmp_path)})
    report = build_runner(traced, rubric, trace=True).run(draft)

    assert report.run_dir is not None and report.run_dir.exists()
    rows = read_trace(report.run_dir / "trace.jsonl")
    events = [row["event"] for row in rows]
    for expected in ("run_start", "perceive", "reason", "act", "reflect", "run_end"):
        assert expected in events
    # Every iteration produced all four steps.
    per_step = [row for row in rows if row["step"] == "reason"]
    assert len(per_step) == report.result.iterations

    summary = json.loads((report.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == report.result.status.value
    assert summary["guardrails"]["tokens"]["budget"] == traced.guardrails.token_budget
    assert summary["harness"]["provider"].startswith("mock")


def test_the_loop_itself_still_runs_without_a_harness(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    """The seams default to 'no harness'; Milestone 1 behaviour is unchanged."""
    from agentic_rubric.core.loop import AgenticLoop

    loop = AgenticLoop(
        config=config,
        provider=MockProvider(responder=ScriptedAgentResponder(rubric=rubric)),
        rubric=rubric,
    )
    result = loop.run(draft)
    assert result.status is RunStatus.TARGET_REACHED


def test_a_controller_can_only_stop_a_run_never_steer_it(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    from agentic_rubric.core.loop import AgenticLoop

    class StopAfterOne:
        def before_iteration(self, state: LoopState) -> StopSignal | None:
            return None

        def after_iteration(
            self, state: LoopState, record: IterationRecord
        ) -> StopSignal | None:
            return StopSignal(RunStatus.STUCK, "controller says stop")

    loop = AgenticLoop(
        config=config,
        provider=MockProvider(responder=ScriptedAgentResponder(rubric=rubric)),
        rubric=rubric,
        controller=StopAfterOne(),
    )
    result = loop.run(draft)
    assert result.status is RunStatus.STUCK
    assert result.iterations == 1
    assert "controller says stop" in " ".join(result.notes)


def test_token_accounting_includes_every_call(
    config: AppConfig, rubric: Rubric, draft: str
) -> None:
    """Reason, the LLM-backed tools and Reflect all spend. The budget sees all three.

    Counted against the provider's own call log rather than a formula, because
    the number of calls per iteration varies: a ``analyze_text`` turn spends
    nothing beyond Reason and Reflect, while a scoring turn adds the judge.
    Before Milestone 3 the Reflect call was silently dropped from the total,
    which is exactly how a token budget overruns while reporting itself healthy.
    """
    provider = MockProvider(responder=ScriptedAgentResponder(rubric=rubric), name="mock")
    report = build_runner(config, rubric, provider=provider).run(draft)
    result = report.result

    assert provider.call_count > result.iterations  # more calls than iterations
    assert result.total_input_tokens == 120 * provider.call_count
    assert result.total_output_tokens == 60 * provider.call_count
