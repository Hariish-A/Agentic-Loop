"""The four steps, their separation, and the loop end to end.

Everything here runs against MockProvider: no API key, no network, no cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rubric.config import load_config
from agentic_rubric.core.act import act
from agentic_rubric.core.loop import AgenticLoop
from agentic_rubric.core.perceive import build_recall_query, perceive
from agentic_rubric.core.reason import reason
from agentic_rubric.core.reflect import assess, reflect
from agentic_rubric.core.rubric import CriterionScore, Rubric, ScoreCard
from agentic_rubric.core.state import (
    ActionResult,
    Decision,
    LoopState,
    MemoryHit,
    RunStatus,
    Workspace,
)
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder
from agentic_rubric.llm.mock import MockProvider, MockTurn, tool_call
from agentic_rubric.llm.types import RateLimitError
from agentic_rubric.memory.base import MemoryRecord, MemoryStore, NullMemory
from agentic_rubric.tools.definitions import build_registry
from agentic_rubric.tools.registry import ToolContext

ESSAY = "config/rubrics/essay_argumentative.yaml"
CONFIG = "config/config.yaml"


@pytest.fixture
def rubric() -> Rubric:
    return Rubric.from_yaml(ESSAY)


@pytest.fixture
def draft() -> str:
    return Path("samples/weak_essay.txt").read_text(encoding="utf-8")


def make_state(rubric: Rubric, draft: str, **kwargs: object) -> LoopState:
    return LoopState(
        rubric=rubric,
        workspace=Workspace(draft=draft),
        target_score=85.0,
        **kwargs,  # type: ignore[arg-type]
    )


def full_card(rubric: Rubric, score: float) -> ScoreCard:
    return ScoreCard.build(
        rubric, [CriterionScore(criterion_id=c.id, score=score) for c in rubric.criteria]
    )


# ===========================================================================
# PERCEIVE
# ===========================================================================


def test_perceive_uses_no_llm(rubric: Rubric, draft: str) -> None:
    provider = MockProvider([MockTurn(text="should never be called")])
    observation = perceive(make_state(rubric, draft), load_config(CONFIG), NullMemory())
    assert provider.call_count == 0
    assert observation.metrics.word_count > 100
    assert observation.probe_results  # deterministic evidence is present


def test_perceive_reports_the_feedback_edge(rubric: Rubric, draft: str) -> None:
    state = make_state(rubric, draft)
    state.workspace.record_score(full_card(rubric, 3))
    state.iteration = 2
    observation = perceive(state, load_config(CONFIG))
    assert observation.current_percent == pytest.approx(50.0)
    assert observation.score_history == (pytest.approx(50.0),)


def test_perceive_truncates_an_oversized_draft_but_measures_all_of_it(rubric: Rubric) -> None:
    huge = "This is a sentence about remote work. " * 2000
    config = load_config(CONFIG, overrides={"guardrails.max_input_chars": 500})
    observation = perceive(make_state(rubric, huge), config)
    assert len(observation.draft) < len(huge)
    assert any("truncated" in note for note in observation.notes)
    # Metrics cover the whole document, not the truncated prompt view.
    assert observation.metrics.word_count > 1000


def test_perceive_survives_a_broken_memory_store(rubric: Rubric, draft: str) -> None:
    class Broken(NullMemory):
        def recall(self, *args: object, **kwargs: object) -> list[MemoryHit]:
            raise OSError("database is locked")

    observation = perceive(make_state(rubric, draft), load_config(CONFIG), Broken())
    assert observation.recalled == ()
    assert any("memory recall failed" in note for note in observation.notes)


def test_recall_query_describes_the_problem_not_the_text(rubric: Rubric, draft: str) -> None:
    state = make_state(rubric, draft)
    state.workspace.record_score(
        ScoreCard.build(
            rubric,
            [
                CriterionScore(criterion_id="thesis", score=1),
                CriterionScore(criterion_id="evidence", score=5),
                CriterionScore(criterion_id="reasoning", score=5),
                CriterionScore(criterion_id="organization", score=5),
                CriterionScore(criterion_id="style_clarity", score=5),
            ],
        )
    )
    query = build_recall_query(state)
    assert "Thesis and Position" in query
    assert "Remote work" not in query  # not the draft


# ===========================================================================
# REASON
# ===========================================================================


def test_reason_forces_tool_use_and_captures_the_thought(rubric: Rubric, draft: str) -> None:
    provider = MockProvider(
        [MockTurn(tool_calls=(tool_call("analyze_text", thought="check the hedging first"),))]
    )
    config = load_config(CONFIG)
    registry = build_registry(rubric)
    observation = perceive(make_state(rubric, draft), config)

    decision = reason(observation, provider, registry, config)
    assert decision.action == "analyze_text"
    assert decision.thought == "check the hedging first"
    assert decision.degraded is False
    assert provider.calls[0].tool_choice == "required"


def test_reason_falls_back_when_the_model_returns_prose(rubric: Rubric, draft: str) -> None:
    provider = MockProvider([MockTurn(text="I think you should make it better.")])
    config = load_config(CONFIG)
    observation = perceive(make_state(rubric, draft), config)

    decision = reason(observation, provider, build_registry(rubric), config)
    assert decision.degraded is True
    assert decision.action == "score_against_rubric"  # safe, read-only


def test_the_degraded_fallback_never_edits_the_document(rubric: Rubric, draft: str) -> None:
    from agentic_rubric.core.reason import fallback_decision

    config = load_config(CONFIG)
    state = make_state(rubric, draft)
    state.workspace.record_score(full_card(rubric, 4))
    observation = perceive(state, config)

    decision = fallback_decision(observation, build_registry(rubric), "unparseable output")
    assert decision.action != "revise_text"
    assert decision.degraded is True


def test_reason_survives_unparseable_tool_arguments(rubric: Rubric, draft: str) -> None:
    from agentic_rubric.llm.types import LLMParseError

    provider = MockProvider([MockTurn(raises=LLMParseError("bad args", raw="{oops"))])
    config = load_config(CONFIG)
    decision = reason(
        perceive(make_state(rubric, draft), config), provider, build_registry(rubric), config
    )
    assert decision.degraded is True


# ===========================================================================
# ACT
# ===========================================================================


def test_act_dispatches_and_attributes_tool_tokens(rubric: Rubric, draft: str) -> None:
    turn = MockTurn(
        tool_calls=(
            tool_call(
                "submit_rubric_scores",
                scores=[
                    {"criterion_id": c.id, "evidence": "", "justification": "", "score": 3}
                    for c in rubric.criteria
                ],
            ),
        )
    )
    ctx = ToolContext(
        config=load_config(CONFIG),
        rubric=rubric,
        workspace=Workspace(draft=draft),
        llm=MockProvider([turn]),
    )
    result, usage = act(
        Decision(action="score_against_rubric", arguments={}), build_registry(rubric), ctx
    )
    assert result.ok
    assert usage.total_tokens > 0  # the judge's tokens, not the reasoner's


def test_act_makes_no_decisions(rubric: Rubric, draft: str) -> None:
    # Given an unusable action, Act reports failure rather than substituting one.
    ctx = ToolContext(
        config=load_config(CONFIG),
        rubric=rubric,
        workspace=Workspace(draft=draft),
        llm=MockProvider([MockTurn(text="x")]),
    )
    result, _ = act(Decision(action="not_a_tool"), build_registry(rubric), ctx)
    assert result.ok is False
    assert result.action == "not_a_tool"


# ===========================================================================
# REFLECT
# ===========================================================================


def scored_state(rubric: Rubric, draft: str, *scores: float) -> LoopState:
    state = make_state(rubric, draft)
    for value in scores:
        state.workspace.record_score(full_card(rubric, value))
    return state


def scoring_result() -> ActionResult:
    return ActionResult(action="score_against_rubric", ok=True, summary="scored")


def test_assess_measures_between_scorecards_not_iterations(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 2, 4)
    config = load_config(CONFIG)
    result = assess(state, perceive(state, config), scoring_result(), config)
    assert result.score_before == pytest.approx(25.0)
    assert result.score_after == pytest.approx(75.0)
    assert result.delta == pytest.approx(50.0)
    assert result.plateau is False


def test_a_revision_turn_produces_no_measurement(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 3)
    revise = ActionResult(action="revise_text", ok=True, summary="revised")
    config = load_config(CONFIG)
    result = assess(state, perceive(state, config), revise, config)
    # Treating every iteration as a data point would flag a plateau on every
    # revision and stop the loop after two turns.
    assert result.delta is None
    assert result.plateau is False


def test_plateau_is_flagged_when_the_score_barely_moves(rubric: Rubric, draft: str) -> None:
    state = make_state(rubric, draft)
    state.workspace.record_score(full_card(rubric, 3))
    state.workspace.record_score(
        ScoreCard.build(
            rubric,
            [
                CriterionScore(criterion_id=c.id, score=3 if c.id != "style_clarity" else 3.02)
                for c in rubric.criteria
            ],
        )
    )
    config = load_config(CONFIG)
    result = assess(state, perceive(state, config), scoring_result(), config)
    assert result.plateau is True


def test_target_met_ends_the_run(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 5)
    config = load_config(CONFIG)
    observation = perceive(state, config)
    reflection = reflect(
        observation,
        Decision(action="score_against_rubric", thought="t"),
        scoring_result(),
        state=state,
        config=config,
        provider=MockProvider([MockTurn(text="no tool call")]),
    )
    assert reflection.task_complete is True
    assert reflection.status is RunStatus.TARGET_REACHED


def test_finalize_below_target_is_declined(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 2)  # 25%, far below the 85% target
    state.workspace.finalized = True
    state.workspace.finalize_reason = "good enough"
    config = load_config(CONFIG)

    reflection = reflect(
        perceive(state, config),
        Decision(action="finalize", thought="calling it done"),
        ActionResult(action="finalize", ok=True, summary="requested"),
        state=state,
        config=config,
        provider=MockProvider([MockTurn(text="")]),
    )
    assert reflection.task_complete is False
    assert "finalize declined" in reflection.critique
    assert state.workspace.finalized is False  # the flag is cleared, not left set


def test_the_model_cannot_vote_itself_done_while_improving(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 2, 4)  # a big jump; clearly still improving
    config = load_config(CONFIG)
    turn = MockTurn(
        tool_calls=(
            tool_call(
                "submit_reflection",
                critique="looks great",
                lesson="",
                next_focus="",
                task_complete=True,  # the model says stop
                reason="I am satisfied",
            ),
        )
    )
    reflection = reflect(
        perceive(state, config),
        Decision(action="score_against_rubric", thought="t"),
        scoring_result(),
        state=state,
        config=config,
        provider=MockProvider([turn]),
    )
    assert reflection.model_votes_done is True
    assert reflection.task_complete is False  # the rules win


def test_reflection_degrades_to_rules_when_the_model_fails(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 2, 3)
    config = load_config(CONFIG)
    reflection = reflect(
        perceive(state, config),
        Decision(action="score_against_rubric", thought="t"),
        scoring_result(),
        state=state,
        config=config,
        provider=MockProvider([MockTurn(raises=RateLimitError("429", provider="mock"))]),
    )
    assert reflection.degraded is True
    assert reflection.critique  # the feedback edge survives
    assert reflection.task_complete is False


def test_next_focus_is_never_empty_while_work_remains(rubric: Rubric, draft: str) -> None:
    state = scored_state(rubric, draft, 2)
    config = load_config(CONFIG)
    turn = MockTurn(
        tool_calls=(
            tool_call(
                "submit_reflection",
                critique="c",
                lesson="",
                next_focus="",  # the model declined to nominate one
                task_complete=False,
                reason="r",
            ),
        )
    )
    reflection = reflect(
        perceive(state, config),
        Decision(action="score_against_rubric", thought="t"),
        scoring_result(),
        state=state,
        config=config,
        provider=MockProvider([turn]),
    )
    assert reflection.next_focus in rubric.ids


# ===========================================================================
# THE LOOP
# ===========================================================================


def run_offline(rubric: Rubric, draft: str, **overrides: object) -> object:
    config = load_config(CONFIG, overrides=overrides or None)
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    loop = AgenticLoop(config=config, provider=MockProvider(responder=responder), rubric=rubric)
    return loop.run(draft, target_score=85.0)


def test_the_loop_actually_iterates_and_improves(rubric: Rubric, draft: str) -> None:
    result = run_offline(rubric, draft)
    assert result.iterations >= 3
    assert result.status is RunStatus.TARGET_REACHED

    trajectory = [card.weighted_percent() for card in result.scorecards]
    assert len(trajectory) >= 3
    assert trajectory == sorted(trajectory)  # monotonically improving
    assert result.best_score > result.initial_score
    assert result.best_score >= 85.0


def test_the_loop_alternates_measuring_and_editing(rubric: Rubric, draft: str) -> None:
    result = run_offline(rubric, draft)
    actions = [record.decision.action for record in result.records]
    assert actions[0] == "score_against_rubric"  # measure before editing
    assert "revise_text" in actions
    # Every revision is followed by a re-score; an unscored edit is unverified.
    for index, action in enumerate(actions[:-1]):
        if action == "revise_text":
            assert actions[index + 1] == "score_against_rubric"


def test_reflection_feeds_into_the_next_iteration(rubric: Rubric, draft: str) -> None:
    result = run_offline(rubric, draft)
    for previous, current in zip(result.records, result.records[1:], strict=False):
        assert current.observation.last_reflection is previous.reflection


def test_the_iteration_cap_is_enforced_by_the_loop(rubric: Rubric, draft: str) -> None:
    result = run_offline(rubric, draft, **{"loop.max_iterations": 2})
    assert result.iterations == 2
    assert result.status is RunStatus.MAX_ITERATIONS_REACHED
    assert any("cap" in note for note in result.notes)


def test_hitting_the_cap_still_returns_the_best_draft(rubric: Rubric, draft: str) -> None:
    result = run_offline(rubric, draft, **{"loop.max_iterations": 4})
    assert result.status is RunStatus.MAX_ITERATIONS_REACHED
    assert result.best_score is not None
    assert result.best_draft  # never empty, never worse than the input
    assert result.best_score >= (result.initial_score or 0)


def test_an_empty_draft_is_rejected_before_any_api_call(rubric: Rubric) -> None:
    config = load_config(CONFIG)
    provider = MockProvider([MockTurn(text="never called")])
    loop = AgenticLoop(config=config, provider=provider, rubric=rubric)
    with pytest.raises(ValueError, match="empty draft"):
        loop.run("   ")
    assert provider.call_count == 0


def test_a_judge_outage_does_not_end_the_run(rubric: Rubric, draft: str) -> None:
    config = load_config(CONFIG)
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    responder.fail_on["judge"] = RateLimitError("429 injected", provider="mock")
    loop = AgenticLoop(config=config, provider=MockProvider(responder=responder), rubric=rubric)

    result = loop.run(draft, target_score=85.0)
    failures = [record for record in result.records if not record.result.ok]
    assert failures  # the injected failure was seen
    assert result.status is RunStatus.TARGET_REACHED  # and recovered from


def test_events_are_emitted_for_every_step(rubric: Rubric, draft: str) -> None:
    config = load_config(CONFIG)
    events: list[str] = []
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    loop = AgenticLoop(
        config=config,
        provider=MockProvider(responder=responder),
        rubric=rubric,
        on_event=lambda name, payload: events.append(name),
    )
    loop.run(draft, target_score=85.0)
    for step in ("run_start", "perceive", "reason", "act", "reflect", "run_end"):
        assert step in events


def test_memory_is_written_after_every_reflection(rubric: Rubric, draft: str) -> None:
    class Recorder(MemoryStore):
        def __init__(self) -> None:
            self.saved: list[MemoryRecord] = []

        def save(self, record: MemoryRecord) -> str:
            self.saved.append(record)
            return "id"

        def recall(self, query: str, **kwargs: object) -> list[MemoryHit]:
            return []

        def clear_session(self, session_id: str) -> int:
            return 0

    memory = Recorder()
    config = load_config(CONFIG)
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    loop = AgenticLoop(
        config=config, provider=MockProvider(responder=responder), rubric=rubric, memory=memory
    )
    result = loop.run(draft, target_score=85.0)

    episodic = [r for r in memory.saved if r.kind == "episodic"]
    lessons = [r for r in memory.saved if r.kind == "lesson"]
    assert len(episodic) == result.iterations  # one per iteration
    assert lessons  # Reflexion lessons were extracted
    assert all(r.session_id == result.session_id for r in memory.saved)


def test_a_memory_write_failure_does_not_end_the_run(rubric: Rubric, draft: str) -> None:
    class Broken(NullMemory):
        def save(self, record: MemoryRecord) -> str:
            raise OSError("disk full")

    config = load_config(CONFIG)
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    loop = AgenticLoop(
        config=config, provider=MockProvider(responder=responder), rubric=rubric, memory=Broken()
    )
    result = loop.run(draft, target_score=85.0)
    assert result.status is RunStatus.TARGET_REACHED
    assert any("memory write failed" in note for note in result.notes)


def test_the_same_loop_code_runs_a_different_domain(draft: str) -> None:
    bug_rubric = Rubric.from_yaml("config/rubrics/bug_report.yaml")
    report = Path("samples/weak_bug_report.txt").read_text(encoding="utf-8")
    result = run_offline(bug_rubric, report)
    assert result.rubric_id == "bug_report"
    assert result.status is RunStatus.TARGET_REACHED
