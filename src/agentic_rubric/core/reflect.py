"""REFLECT -- evaluate the iteration and decide whether the task is done.

Split into two halves that must not be confused:

**Deterministic** (:func:`assess`) computes what actually happened -- score
before, score after, delta, plateau -- and applies the termination rules. Target
met, plateau streak, and whether a ``finalize`` request is credible are all
decided in Python from numbers.

**Verbal** (the LLM call) produces the critique, the ``next_focus``, and the
*lesson*. This is the Reflexion half: the lesson is written to memory and
replayed into later reasoning, which is how the agent stops repeating a strategy
that did not work.

The reason for the split is simple. Ask a model whether its work is finished and
it will eventually say yes, particularly when the remaining points are the hard
ones. Its vote is recorded as ``model_votes_done`` and counts only as an input
to the plateau rule -- never as an override. A ``finalize`` call made below
target with the score still climbing is declined, and the agent is told why.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.types import LLMError, LLMParseError, Usage
from ..prompts import reflect as reflect_prompt
from .state import (
    ActionResult,
    Decision,
    LoopState,
    Observation,
    Reflection,
    RunStatus,
)

SCORING_ACTION = "score_against_rubric"


@dataclass(frozen=True)
class Assessment:
    """The measurable facts about one iteration, computed without a model."""

    scored_this_turn: bool
    score_before: float | None
    score_after: float | None
    delta: float | None
    plateau: bool
    target_met: bool
    plateau_stop: bool
    prospective_streak: int


def assess(state: LoopState, observation: Observation, result: ActionResult,
           config: AppConfig) -> Assessment:
    """Compute score movement and the termination conditions.

    Movement is measured **between consecutive scorecards**, not between
    iterations. A ``revise_text`` turn produces no measurement at all, so
    treating every iteration as a data point would report a plateau on every
    revision and stop the loop after two turns.
    """
    history = state.workspace.scorecard_history
    scored_this_turn = result.ok and result.action == SCORING_ACTION

    score_before: float | None = None
    score_after: float | None = state.workspace.percent
    delta: float | None = None
    plateau = False

    if scored_this_turn and history:
        score_after = history[-1].weighted_percent()
        if len(history) >= 2:
            score_before = history[-2].weighted_percent()
            delta = score_after - score_before
            plateau = delta < config.loop.min_improvement

    target_met = score_after is not None and score_after >= state.target_score
    prospective_streak = state.plateau_streak + 1 if plateau else 0
    plateau_stop = (
        prospective_streak >= config.loop.plateau_patience and score_after is not None
    )

    return Assessment(
        scored_this_turn=scored_this_turn,
        score_before=score_before,
        score_after=score_after,
        delta=delta,
        plateau=plateau,
        target_met=target_met,
        plateau_stop=plateau_stop,
        prospective_streak=prospective_streak,
    )


def _resolve_completion(
    state: LoopState, assessment: Assessment
) -> tuple[bool, RunStatus, str, str]:
    """Apply the termination rules. Returns (done, status, reason, extra_note)."""
    finalize_requested = state.workspace.finalized
    credible = assessment.target_met or assessment.plateau_stop

    if finalize_requested and not credible:
        # Decline, and undo the flag so the workspace does not stay poisoned.
        state.workspace.finalized = False
        current = f"{assessment.score_after:.1f}%" if assessment.score_after else "unknown"
        note = (
            f"finalize declined: the draft is at {current} against a target of "
            f"{state.target_score:.0f}% and the score is still moving. Keep improving."
        )
        return False, RunStatus.RUNNING, note, note

    if assessment.target_met:
        return (
            True,
            RunStatus.TARGET_REACHED,
            f"target met: {assessment.score_after:.1f}% >= {state.target_score:.0f}%",
            "",
        )

    if finalize_requested:
        return (
            True,
            RunStatus.AGENT_FINALIZED,
            f"agent finalized at {assessment.score_after:.1f}%: "
            f"{state.workspace.finalize_reason}",
            "",
        )

    if assessment.plateau_stop:
        return (
            True,
            RunStatus.PLATEAU,
            f"no meaningful improvement for {assessment.prospective_streak} consecutive "
            f"scored iterations; stopping at {assessment.score_after:.1f}%",
            "",
        )

    return False, RunStatus.RUNNING, "work continues", ""


def _deterministic_critique(
    decision: Decision, result: ActionResult, assessment: Assessment
) -> str:
    """Fallback critique when the reflection model is unavailable.

    Weaker than the model's, but it keeps the loop's feedback edge intact: the
    next iteration still learns whether the last action worked.
    """
    if not result.ok:
        return f"The {decision.action} call failed: {result.error}"
    if assessment.delta is None:
        return f"{decision.action} completed; no new measurement was taken this turn."
    verdict = "moved the score" if assessment.delta > 0 else "did not help"
    return (
        f"{decision.action} {verdict} ({assessment.delta:+.1f} points). "
        "Assessed without the reflection model."
    )


def reflect(
    observation: Observation,
    decision: Decision,
    result: ActionResult,
    *,
    state: LoopState,
    config: AppConfig,
    provider: LLMProvider,
) -> Reflection:
    """Evaluate the iteration and produce the input to the next Perceive."""
    assessment = assess(state, observation, result, config)
    done, status, reason, extra_note = _resolve_completion(state, assessment)

    critique = ""
    lesson: str | None = None
    next_focus: str | None = None
    model_votes_done = False
    degraded = False
    usage = Usage()

    messages = reflect_prompt.build_messages(
        observation,
        decision,
        result,
        score_before=assessment.score_before,
        score_after=assessment.score_after,
        delta=assessment.delta,
        plateau=assessment.plateau,
        target_score=state.target_score,
    )
    spec = reflect_prompt.build_spec(state.rubric)

    try:
        response = provider.complete(
            messages,
            tools=[spec],
            tool_choice=spec.name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        usage = response.usage
        arguments = response.first_tool_call().arguments
    except (LLMError, LLMParseError) as exc:
        # Reflection is valuable, not load-bearing. Losing it degrades the
        # quality of the next decision; it must not end the run.
        degraded = True
        critique = _deterministic_critique(decision, result, assessment)
        state.note(f"reflection model unavailable ({type(exc).__name__}); used rule-based critique")
    else:
        critique = str(arguments.get("critique", "")).strip()
        raw_lesson = str(arguments.get("lesson", "")).strip()
        lesson = raw_lesson or None
        model_votes_done = bool(arguments.get("task_complete", False))
        raw_focus = str(arguments.get("next_focus", "")).strip()
        next_focus = raw_focus if raw_focus in state.rubric.ids else None

    # A model that votes done while plateauing is corroboration, not authority:
    # it only matters once the deterministic plateau rule already fired.
    if not done and model_votes_done and assessment.plateau_stop:
        done, status = True, RunStatus.PLATEAU
        reason = "plateau confirmed by both the score history and the reflection model"

    if not done and next_focus is None and state.workspace.scorecard is not None:
        # Never hand the next iteration an empty focus while work remains.
        next_focus = state.workspace.scorecard.weakest(1)[0].criterion_id

    if extra_note:
        critique = f"{critique} {extra_note}".strip() if critique else extra_note

    return Reflection(
        task_complete=done,
        reason=reason,
        critique=critique,
        lesson=lesson,
        next_focus=next_focus,
        score_delta=assessment.delta,
        plateau=assessment.plateau,
        model_votes_done=model_votes_done,
        status=status,
        usage=usage,
        degraded=degraded,
    )


__all__ = ["Assessment", "SCORING_ACTION", "assess", "reflect"]
