"""Tool schemas, argument validation, dispatch containment, and the handlers."""

from __future__ import annotations

import pytest

from agentic_rubric.config import load_config
from agentic_rubric.core.rubric import CriterionScore, Rubric, ScoreCard
from agentic_rubric.core.state import Decision, ErrorKind, Workspace
from agentic_rubric.llm.mock import MockProvider, MockTurn, tool_call
from agentic_rubric.llm.types import RateLimitError
from agentic_rubric.tools.definitions import build_registry
from agentic_rubric.tools.registry import (
    THOUGHT_FIELD,
    ToolContext,
    ToolError,
    ToolOutput,
    ToolRegistry,
    build_spec,
    validate_arguments,
)
from agentic_rubric.tools.text_stats import compute_metrics, unified_diff_summary

ESSAY = "config/rubrics/essay_argumentative.yaml"
DRAFT = (
    "Remote work is somewhat complicated. There are many views on it. It could be "
    "said that both sides have points worth considering in various ways."
)


@pytest.fixture
def rubric() -> Rubric:
    return Rubric.from_yaml(ESSAY)


def make_ctx(rubric: Rubric, provider: MockProvider, draft: str = DRAFT) -> ToolContext:
    return ToolContext(
        config=load_config("config/config.yaml"),
        rubric=rubric,
        workspace=Workspace(draft=draft),
        llm=provider,
        iteration=1,
    )


def judge_turn(rubric: Rubric, score: int) -> MockTurn:
    return MockTurn(
        tool_calls=(
            tool_call(
                "submit_rubric_scores",
                scores=[
                    {
                        "criterion_id": c.id,
                        "evidence": "quoted",
                        "justification": "because",
                        "score": score,
                    }
                    for c in rubric.criteria
                ],
            ),
        )
    )


# --- schema construction ---------------------------------------------------


def test_every_tool_requires_a_thought(rubric: Rubric) -> None:
    # ReAct fidelity is enforced at the schema level, not by convention.
    for spec in build_registry(rubric).specs():
        assert THOUGHT_FIELD in spec.parameters["required"]


def test_criterion_arguments_are_constrained_to_the_actual_rubric(rubric: Rubric) -> None:
    spec = build_registry(rubric).get("revise_text").spec
    enum = spec.parameters["properties"]["focus_criteria"]["items"]["enum"]
    assert enum == list(rubric.ids)


def test_a_different_rubric_produces_a_different_tool_set() -> None:
    bug = Rubric.from_yaml("config/rubrics/bug_report.yaml")
    spec = build_registry(bug).get("revise_text").spec
    assert "reproducibility" in spec.parameters["properties"]["focus_criteria"]["items"]["enum"]


# --- argument validation ---------------------------------------------------


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 3},
        "count": {"type": "integer", "minimum": 1, "maximum": 10},
        "tags": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}, "minItems": 1},
        "flag": {"type": "boolean"},
    },
    "required": ["name"],
    "additionalProperties": False,
}


def test_valid_arguments_produce_no_errors() -> None:
    assert validate_arguments(SCHEMA, {"name": "abc", "count": 3, "tags": ["x"]}) == []


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "missing required argument"),
        ({"name": "abc", "extra": 1}, "unknown argument"),
        ({"name": 5}, "expected string"),
        ({"name": "abc", "count": "many"}, "expected integer"),
        ({"name": "abc", "count": True}, "expected integer, got boolean"),
        ({"name": "abc", "count": 99}, "above the maximum"),
        ({"name": "abc", "count": 0}, "below the minimum"),
        ({"name": "ab"}, "at least 3 character"),
        ({"name": "abc", "tags": ["z"]}, "is not one of"),
        ({"name": "abc", "tags": []}, "at least 1 item"),
    ],
)
def test_invalid_arguments_are_described_precisely(arguments: dict, fragment: str) -> None:
    errors = validate_arguments(SCHEMA, arguments)
    assert any(fragment in error for error in errors), errors


def test_thought_is_stripped_before_validation(rubric: Rubric) -> None:
    # The model must supply `thought`; the handler must never be asked for it.
    entry = build_registry(rubric).get("diff_drafts")
    assert THOUGHT_FIELD in entry.spec.parameters["required"]
    assert THOUGHT_FIELD not in entry.validation_schema["required"]
    assert THOUGHT_FIELD not in entry.validation_schema["properties"]


# --- dispatch containment --------------------------------------------------


def test_unknown_tool_returns_a_result_not_an_exception(rubric: Rubric) -> None:
    registry = build_registry(rubric)
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    result = registry.dispatch(Decision(action="teleport", arguments={}), ctx)
    assert result.ok is False
    assert "unknown tool" in result.error
    assert "revise_text" in result.error  # tells the agent what it may call


def test_handler_exceptions_are_contained(rubric: Rubric) -> None:
    registry = ToolRegistry()

    def explode(arguments: dict, ctx: ToolContext) -> ToolOutput:
        raise RuntimeError("kaboom")

    registry.register(build_spec("boom", "explodes", {}), explode)
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    result = registry.dispatch(Decision(action="boom", arguments={"thought": "t"}), ctx)
    assert result.ok is False
    assert "RuntimeError: kaboom" in result.error


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    spec = build_spec("dup", "d", {})
    registry.register(spec, lambda a, c: ToolOutput(summary="ok"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec, lambda a, c: ToolOutput(summary="ok"))


# --- scoring handler -------------------------------------------------------


def test_scoring_records_a_card_and_reports_headroom(rubric: Rubric) -> None:
    registry = build_registry(rubric)
    provider = MockProvider([judge_turn(rubric, 2)])
    ctx = make_ctx(rubric, provider)

    result = registry.dispatch(
        Decision(action="score_against_rubric", arguments={"thought": "measure"}), ctx
    )
    assert result.ok
    assert ctx.workspace.scorecard is not None
    assert result.output["weighted_percent"] == pytest.approx(25.0)
    assert "Biggest opportunities" in result.summary
    assert ctx.usage.total_tokens > 0  # the tool's tokens are attributed


def test_judge_transport_failure_becomes_a_failed_action(rubric: Rubric) -> None:
    registry = build_registry(rubric)
    provider = MockProvider([MockTurn(raises=RateLimitError("429", provider="mock"))])
    result = registry.dispatch(
        Decision(action="score_against_rubric", arguments={"thought": "t"}),
        make_ctx(rubric, provider),
    )
    assert result.ok is False
    assert "judge call failed" in result.error


def test_judge_prose_instead_of_scores_is_a_failed_action(rubric: Rubric) -> None:
    registry = build_registry(rubric)
    provider = MockProvider([MockTurn(text="I think it is quite good, honestly.")])
    result = registry.dispatch(
        Decision(action="score_against_rubric", arguments={"thought": "t"}),
        make_ctx(rubric, provider),
    )
    assert result.ok is False
    assert "prose instead of scores" in result.error


def test_hallucinated_criterion_is_dropped_not_fatal(rubric: Rubric) -> None:
    turn = MockTurn(
        tool_calls=(
            tool_call(
                "submit_rubric_scores",
                scores=[
                    {"criterion_id": "thesis", "evidence": "", "justification": "", "score": 4},
                    {"criterion_id": "vibes", "evidence": "", "justification": "", "score": 5},
                ],
            ),
        )
    )
    ctx = make_ctx(rubric, MockProvider([turn]))
    result = build_registry(rubric).dispatch(
        Decision(action="score_against_rubric", arguments={"thought": "t"}), ctx
    )
    assert result.ok  # one bad id must not discard the good scores
    assert "vibes" in ctx.workspace.scorecard.judge_notes


# --- revision handler ------------------------------------------------------


def test_revision_replaces_the_draft_and_clears_the_stale_score(rubric: Rubric) -> None:
    long_draft = DRAFT * 6
    ctx = make_ctx(rubric, MockProvider([MockTurn(text=long_draft + " A new concrete sentence.")]),
                   draft=long_draft)
    ctx.workspace.record_score(
        ScoreCard.build(
            rubric, [CriterionScore(criterion_id=c.id, score=2) for c in rubric.criteria]
        )
    )

    result = build_registry(rubric).dispatch(
        Decision(
            action="revise_text",
            arguments={
                "thought": "t",
                "focus_criteria": ["thesis"],
                "instructions": "state the claim explicitly up front",
            },
        ),
        ctx,
    )
    assert result.ok
    assert ctx.workspace.previous_draft == long_draft
    # The old card describes the old text; leaving it would steer on a stale number.
    assert ctx.workspace.scorecard is None


def test_revision_that_changes_nothing_is_a_failure(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text=DRAFT)]))
    result = build_registry(rubric).dispatch(
        Decision(
            action="revise_text",
            arguments={
                "thought": "t",
                "focus_criteria": ["thesis"],
                "instructions": "make the claim specific and arguable",
            },
        ),
        ctx,
    )
    assert result.ok is False
    assert "identical to the input" in result.error


def test_revision_that_returns_a_summary_is_a_failure(rubric: Rubric) -> None:
    long_draft = DRAFT * 10
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="Remote work is complex.")]),
                   draft=long_draft)
    result = build_registry(rubric).dispatch(
        Decision(
            action="revise_text",
            arguments={
                "thought": "t",
                "focus_criteria": ["thesis"],
                "instructions": "make the claim specific and arguable",
            },
        ),
        ctx,
    )
    assert result.ok is False
    assert "summary, not a revision" in result.error


def test_unknown_focus_criterion_is_rejected_by_the_handler(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    # Bypass schema validation to prove the handler defends itself too.
    from agentic_rubric.tools.handlers import revision

    with pytest.raises(ToolError, match="unknown criteria"):
        revision.handle(
            {"focus_criteria": ["ghost"], "instructions": "do something specific here"}, ctx
        )


def test_tree_of_thoughts_branch_scores_candidates_and_keeps_the_best(rubric: Rubric) -> None:
    long_draft = DRAFT * 6
    turns = [
        MockTurn(text=long_draft + " Candidate one adds a modest clarification."),
        MockTurn(text=long_draft + " Candidate two adds a named 2021 study and a figure."),
        judge_turn(rubric, 2),  # scores candidate one
        judge_turn(rubric, 5),  # scores candidate two -- the winner
    ]
    config = load_config("config/config.yaml", overrides={"loop.revise_candidates": 2})
    ctx = ToolContext(
        config=config,
        rubric=rubric,
        workspace=Workspace(draft=long_draft),
        llm=MockProvider(turns),
        iteration=1,
    )
    result = build_registry(rubric).dispatch(
        Decision(
            action="revise_text",
            arguments={
                "thought": "t",
                "focus_criteria": ["evidence"],
                "instructions": "attribute the productivity claim to a named source",
            },
        ),
        ctx,
    )
    assert result.ok
    assert result.output["candidates_generated"] == 2
    assert "Candidate two" in ctx.workspace.draft
    assert "best of 2 candidates" in result.summary


# --- analysis and control --------------------------------------------------


def test_analyze_reports_metrics_and_failing_probes(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    result = build_registry(rubric).dispatch(
        Decision(action="analyze_text", arguments={"thought": "t"}), ctx
    )
    assert result.ok
    assert result.output["metrics"]["word_count"] > 0
    assert result.output["failing_probe_ids"]  # the weak draft fails several


def test_diff_requires_a_previous_draft(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    result = build_registry(rubric).dispatch(
        Decision(action="diff_drafts", arguments={"thought": "t"}), ctx
    )
    assert result.ok is False
    assert "no previous draft" in result.error


def test_finalize_refuses_an_unscored_draft(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    result = build_registry(rubric).dispatch(
        Decision(action="finalize", arguments={"thought": "t", "reason": "looks good to me"}), ctx
    )
    assert result.ok is False
    assert "has not been scored" in result.error
    assert ctx.workspace.finalized is False


def test_finalize_sets_the_flag_once_a_score_exists(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, MockProvider([MockTurn(text="x")]))
    ctx.workspace.record_score(
        ScoreCard.build(
            rubric, [CriterionScore(criterion_id=c.id, score=5) for c in rubric.criteria]
        )
    )
    result = build_registry(rubric).dispatch(
        Decision(action="finalize", arguments={"thought": "t", "reason": "target comfortably met"}),
        ctx,
    )
    assert result.ok
    assert ctx.workspace.finalized is True


# --- text statistics -------------------------------------------------------


def test_metrics_are_computed_on_real_text() -> None:
    metrics = compute_metrics(DRAFT)
    assert metrics.word_count > 10
    assert metrics.sentence_count >= 3
    assert metrics.hedge_count >= 1  # "somewhat", "it could be said"
    assert metrics.filler_count >= 1  # "there are"


def test_metrics_survive_empty_input() -> None:
    assert compute_metrics("").word_count == 0


def test_diff_summary_detects_no_change() -> None:
    assert unified_diff_summary(DRAFT, DRAFT)["changed"] is False
    assert unified_diff_summary(DRAFT, DRAFT + "\nAnd one more line.")["changed"] is True


# ===========================================================================
# Handler contracts that the happy path never reaches
# ===========================================================================


def idle(name: str = "idle") -> MockProvider:
    """A provider these tests must never reach: they exercise pure-Python tools."""
    return MockProvider([MockTurn(text="should never be called")], name=name)


def test_analyze_refuses_an_empty_draft(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, idle(), draft="   \n  ")
    result = build_registry(rubric).dispatch(Decision(action="analyze_text"), ctx)
    assert result.ok is False
    assert result.error_kind is ErrorKind.RECOVERABLE   # the agent should read this
    assert "nothing to analyse" in (result.error or "")


def test_analyze_can_be_narrowed_to_named_criteria(rubric: Rubric) -> None:
    """So a targeted revision can check only the probes it is trying to fix."""
    ctx = make_ctx(rubric, idle())
    registry = build_registry(rubric)

    everything = registry.dispatch(Decision(action="analyze_text"), ctx)
    narrowed = registry.dispatch(
        Decision(action="analyze_text", arguments={"criteria": ["evidence"]}), ctx
    )
    assert narrowed.ok is True
    assert len(narrowed.output["probes"]) < len(everything.output["probes"])
    assert narrowed.output["probes"]  # and it is not empty


def test_analyze_rejects_a_criterion_this_rubric_does_not_have(rubric: Rubric) -> None:
    """The error names the real ids, because an LLM is what reads it next."""
    result = build_registry(rubric).dispatch(
        Decision(action="analyze_text", arguments={"criteria": ["nonexistent"]}),
        make_ctx(rubric, idle()),
    )
    assert result.ok is False
    assert "nonexistent" in (result.error or "")
    assert "thesis" in (result.error or "")


def test_diff_reports_what_a_revision_actually_changed(rubric: Rubric) -> None:
    ctx = make_ctx(rubric, idle())
    ctx.workspace.replace_draft(
        DRAFT + "\n\nA newly added paragraph carrying real substance and a 2023 figure."
    )

    result = build_registry(rubric).dispatch(Decision(action="diff_drafts"), ctx)
    assert result.ok is True
    assert "line(s) added" in result.summary
    assert result.output["diff"]["word_delta"] > 0
    assert result.output["added_preview"]


def test_diff_calls_an_unchanged_revision_what_it_is(rubric: Rubric) -> None:
    """Stops the agent grading its own homework on a no-op edit."""
    ctx = make_ctx(rubric, idle())
    ctx.workspace.replace_draft(DRAFT)

    result = build_registry(rubric).dispatch(Decision(action="diff_drafts"), ctx)
    assert result.ok is True
    assert "no meaningful change" in result.summary
