"""The admission gate and the symmetric reviser guards.

Both exist because of one observed live failure: given ``hi, good morning`` the
agent scored it 0.0%, expanded three words into a 170-word essay with 0.00
similarity to the input, scored its own invention at 85.0%, and reported
``target_reached``. These tests pin every part of that chain shut.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rubric.config import load_config
from agentic_rubric.core.admission import check_admission, thresholds
from agentic_rubric.core.loop import AgenticLoop
from agentic_rubric.core.rubric import Rubric
from agentic_rubric.core.state import RunStatus, Workspace
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder
from agentic_rubric.llm.mock import MockProvider, MockTurn
from agentic_rubric.tools.definitions import build_registry
from agentic_rubric.tools.handlers.revision import (
    MAX_EXPANSION_RATIO,
    _validate_candidate,
    word_retention,
)
from agentic_rubric.tools.registry import ToolContext

CONFIG = "config/config.yaml"
ESSAY = "config/rubrics/essay_argumentative.yaml"
BUG = "config/rubrics/bug_report.yaml"

#: The exact input that produced the live failure.
GREETING = "hi, good morning"


@pytest.fixture
def config():  # noqa: ANN201
    return load_config(CONFIG)


@pytest.fixture
def essay() -> Rubric:
    return Rubric.from_yaml(ESSAY)


@pytest.fixture
def weak_essay() -> str:
    return Path("samples/weak_essay.txt").read_text(encoding="utf-8")


# ===========================================================================
# The gate
# ===========================================================================


def test_the_reported_failure_case_is_refused(essay: Rubric, config) -> None:  # noqa: ANN001
    verdict = check_admission(GREETING, essay, config)
    assert verdict.ok is False
    assert verdict.measurements["words"] == 3
    assert {c.name for c in verdict.failures} == {"min_words", "min_sentences"}


def test_a_real_draft_is_admitted(essay: Rubric, config, weak_essay: str) -> None:  # noqa: ANN001
    assert check_admission(weak_essay, essay, config).ok is True


def test_an_empty_submission_says_so_plainly(essay: Rubric, config) -> None:  # noqa: ANN001
    verdict = check_admission("   \n  ", essay, config)
    assert verdict.ok is False
    assert "Nothing was submitted" in verdict.reason


def test_long_enough_but_a_single_sentence_is_refused(essay: Rubric, config) -> None:  # noqa: ANN001
    """Word count alone is not enough: a 70-word fragment is not an essay."""
    fragment = " ".join(["word"] * 70)
    verdict = check_admission(fragment, essay, config)
    assert verdict.ok is False
    assert [c.name for c in verdict.failures] == ["min_sentences"]


def test_the_rejection_message_is_written_for_a_person(essay: Rubric, config) -> None:  # noqa: ANN001
    reason = check_admission(GREETING, essay, config).reason
    assert "Argumentative Essay" in reason  # what it was measured against
    assert "3 word" in reason  # what was wrong
    assert "Submit a longer draft" in reason  # what to do about it


def test_thresholds_come_from_the_rubric_then_the_config(config) -> None:  # noqa: ANN001
    essay = Rubric.from_yaml(ESSAY)
    bug = Rubric.from_yaml(BUG)
    assert thresholds(essay, config) == (60, 3)
    # A bug report is legitimately terser than an essay; the floor is a property
    # of the rubric, not of the agent.
    assert thresholds(bug, config) == (30, 2)


def test_a_rubric_without_a_floor_falls_back_to_config(config) -> None:  # noqa: ANN001
    bare = Rubric.from_dict(
        {
            "id": "bare",
            "name": "Bare",
            "criteria": [{"id": "a", "name": "A", "weight": 1.0, "description": ""}],
        }
    )
    assert bare.min_words is None
    assert thresholds(bare, config) == (
        config.loop.min_input_words,
        config.loop.min_input_sentences,
    )


def test_the_floor_is_configurable_without_touching_code() -> None:
    relaxed = load_config(CONFIG, overrides={"loop.min_input_words": 1})
    bare = Rubric.from_dict(
        {
            "id": "bare",
            "name": "Bare",
            "criteria": [{"id": "a", "name": "A", "weight": 1.0, "description": ""}],
        }
    )
    assert thresholds(bare, relaxed)[0] == 1


def test_the_gate_never_calls_a_model(essay: Rubric, config) -> None:  # noqa: ANN001
    provider = MockProvider([MockTurn(text="must never be called")])
    check_admission(GREETING, essay, config)
    check_admission("a real draft " * 40, essay, config)
    assert provider.call_count == 0


# ===========================================================================
# The gate, through the loop
# ===========================================================================


def run(rubric: Rubric, text: str, **overrides: object):  # noqa: ANN201
    config = load_config(CONFIG, overrides=overrides or None)
    provider = MockProvider(
        responder=ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    )
    loop = AgenticLoop(config=config, provider=provider, rubric=rubric)
    return loop.run(text, session_id="admission-test"), provider


def test_a_refused_submission_costs_zero_model_calls(essay: Rubric) -> None:
    result, provider = run(essay, GREETING)
    assert result.status is RunStatus.INPUT_REJECTED
    assert result.iterations == 0
    assert provider.call_count == 0  # the whole point


def test_a_refused_submission_hands_the_text_back_unchanged(essay: Rubric) -> None:
    result, _ = run(essay, GREETING)
    # Nothing was scored and nothing was rewritten, so the best draft the run
    # produced is the input. It must never be a fabrication.
    assert result.best_draft == GREETING
    assert result.final_draft == GREETING
    assert result.best_score is None
    assert result.initial_score is None


def test_a_refused_submission_explains_itself_in_the_notes(essay: Rubric) -> None:
    result, _ = run(essay, GREETING)
    assert result.notes
    assert any("cannot be scored" in note for note in result.notes)


def test_a_refusal_is_not_reported_as_success(essay: Rubric) -> None:
    result, _ = run(essay, GREETING)
    assert result.status.is_success is False
    assert result.status.is_guardrail_stop is False  # it is a verdict, not a stop


def test_the_refusal_is_emitted_as_an_event(essay: Rubric) -> None:
    config = load_config(CONFIG)
    events: list[str] = []
    loop = AgenticLoop(
        config=config,
        provider=MockProvider([MockTurn(text="unused")]),
        rubric=essay,
        on_event=lambda name, payload: events.append(name),
    )
    loop.run(GREETING)
    assert events == ["input_rejected"]


def test_an_admissible_draft_still_runs_normally(essay: Rubric, weak_essay: str) -> None:
    result, provider = run(essay, weak_essay)
    assert result.status is RunStatus.TARGET_REACHED
    assert provider.call_count > 0


def test_a_bug_report_shorter_than_an_essay_is_still_admitted() -> None:
    bug = Rubric.from_yaml(BUG)
    report = Path("samples/vague_bug_report.txt").read_text(encoding="utf-8")
    result, _ = run(bug, report)
    assert result.status is not RunStatus.INPUT_REJECTED


# ===========================================================================
# The reviser guards, now symmetric
# ===========================================================================


def test_a_composition_is_rejected() -> None:
    """The second half of the live failure: 3 words became 170."""
    reason = _validate_candidate(GREETING, "An essay about remote work. " * 30)
    assert reason is not None
    assert "composition, not a revision" in reason


def test_a_summary_is_still_rejected() -> None:
    reason = _validate_candidate("word " * 100, "Short.")
    assert reason is not None
    assert "summary, not a revision" in reason


def test_an_unchanged_draft_is_still_rejected() -> None:
    text = "Remote work is complicated.\nThere are many views on it.\nBoth sides have points."
    assert "identical to the input" in (_validate_candidate(text, text) or "")


def test_a_normal_revision_passes_all_four_guards() -> None:
    original = (
        "Remote work is somewhat complicated. There are many views on it.\n"
        "Both sides have points that are worth considering carefully here.\n"
    ) * 3
    revised = original + "\nA 2023 survey found that 62 percent of respondents agreed."
    assert _validate_candidate(original, revised) is None


def test_growth_is_allowed_up_to_the_expansion_bound() -> None:
    original = "word " * 100
    just_under = "word " * int(100 * MAX_EXPANSION_RATIO - 5)
    just_over = "word " * int(100 * MAX_EXPANSION_RATIO + 20)
    assert _validate_candidate(original, just_under) is None
    assert "composition" in (_validate_candidate(original, just_over) or "")


def test_a_wholesale_replacement_is_rejected_on_vocabulary() -> None:
    """Line similarity cannot tell a rewrite from a fabrication; vocabulary can."""
    # Comfortably above RETENTION_FLOOR_MIN_WORDS, below which the denominator
    # is too small for vocabulary overlap to mean anything.
    original = (
        "Remote work reduces attrition at large technology firms. "
        "Commuting time falls and retention improves measurably across teams. "
        "Managers worry about collaboration and company culture suffering. "
        "The evidence on productivity remains genuinely contested among researchers."
    )
    unrelated = (
        "Photosynthesis converts light into chemical energy inside chloroplasts. "
        "Plants absorb carbon dioxide through stomata during daylight hours. "
        "Sugars produced then travel through phloem toward developing roots. "
        "Chlorophyll pigments determine which wavelengths a leaf can capture."
    )
    reason = _validate_candidate(original, unrelated)
    assert reason is not None
    assert "original wording survives" in reason


def test_vocabulary_retention_is_not_applied_to_tiny_originals() -> None:
    # Below the floor the denominator is too small to mean anything, so the
    # expansion bound is what catches the case instead.
    assert word_retention("hi there", "completely different words entirely") == 0.0


def test_the_reviser_rejects_a_composition_through_the_registry(essay: Rubric) -> None:
    """End to end: the guard holds when reached through tool dispatch."""
    from agentic_rubric.core.state import Decision

    original = "Remote work is complicated. Views differ widely. Both sides have points."
    ctx = ToolContext(
        config=load_config(CONFIG),
        rubric=essay,
        workspace=Workspace(draft=original),
        llm=MockProvider([MockTurn(text="An invented essay paragraph. " * 40)]),
    )
    result = build_registry(essay).dispatch(
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
    assert result.ok is False
    assert "composition" in (result.error or "")
    assert ctx.workspace.draft == original  # the user's text is untouched
