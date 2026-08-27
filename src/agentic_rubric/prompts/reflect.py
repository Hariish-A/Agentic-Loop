"""Prompt and schema for the Reflect step -- the Reflexion half of the loop.

Reflect does two different jobs, and the split matters:

* **Deterministic rules decide whether the run is over.** Target met, plateau
  streak, iteration cap. Those are computed in code from the scores.
* **The model produces the verbal reflection**: what went wrong, what to do
  next, and -- the part that makes this Reflexion rather than plain
  self-critique -- a *lesson*, phrased to be reusable on a different text.

``task_complete`` appears in the schema so the model's opinion is captured, but
it is named ``model_votes_done`` downstream and only ever counts as one input to
the plateau rule. An agent allowed to declare its own success will do so as soon
as the remaining work gets hard.

The lesson field is where cross-session memory comes from in Milestone 2, so the
prompt pushes hard for generalisable phrasing. "Add a statistic to paragraph
two" is worthless next session; "on this rubric, unattributed figures score no
better than no figures" is worth carrying forever.
"""

from __future__ import annotations

from ..core.rubric import Rubric
from ..core.state import ActionResult, Decision, Observation
from ..llm.types import Message, ToolSpec, system, user


def build_spec(rubric: Rubric) -> ToolSpec:
    """Reflection schema, with ``next_focus`` constrained to real criterion ids."""
    return ToolSpec(
        name="submit_reflection",
        description="Record what this iteration achieved and what should happen next.",
        parameters={
            "type": "object",
            "properties": {
                "critique": {
                    "type": "string",
                    "description": (
                        "Two or three sentences: did this action achieve what the thought "
                        "predicted? If the score barely moved, say specifically why you "
                        "think that is."
                    ),
                },
                "lesson": {
                    "type": "string",
                    "description": (
                        "A single durable, transferable lesson, or an empty string if this "
                        "iteration taught nothing new. It must make sense applied to a "
                        "DIFFERENT text scored against this rubric. Good: 'unattributed "
                        "statistics score no higher than no statistics on evidence'. Bad: "
                        "'the second paragraph needed a source'."
                    ),
                },
                "next_focus": {
                    "type": "string",
                    "enum": [*rubric.ids, ""],
                    "description": (
                        "The criterion the next iteration should target, or an empty "
                        "string if the work is done."
                    ),
                },
                "task_complete": {
                    "type": "boolean",
                    "description": (
                        "Your opinion on whether further revision can still help. Advisory "
                        "only; the loop checks this against the actual scores."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence supporting your task_complete answer.",
                },
            },
            "required": ["critique", "lesson", "next_focus", "task_complete", "reason"],
        },
    )


SYSTEM = """You are the reflection step of an agentic loop that improves text
against a rubric.

You are looking at one completed iteration: what the agent intended, what it
did, and what happened to the score. Be blunt. A reflection that congratulates a
failed edit is worse than no reflection, because the next iteration will repeat
it.

Judge the action against its own stated intent, not against whether the text
reads nicely. If the thought predicted a gain on `evidence` and `evidence` did
not move, that is a failure even if the prose improved.

Extract a lesson only when there is a real one. An empty lesson is an honest
answer; a generic one ("be more specific") pollutes memory for every future run.

Call submit_reflection exactly once. Do not reply with prose."""


def build_messages(
    observation: Observation,
    decision: Decision,
    result: ActionResult,
    *,
    score_before: float | None,
    score_after: float | None,
    delta: float | None,
    plateau: bool,
    target_score: float,
) -> list[Message]:
    """Assemble the Reflect conversation for one completed iteration."""
    movement = (
        f"{score_before:.1f}% -> {score_after:.1f}% ({delta:+.1f} points)"
        if score_before is not None and score_after is not None and delta is not None
        else (
            f"{score_after:.1f}% (first measurement)"
            if score_after is not None
            else "no score available - the draft is currently unscored"
        )
    )

    outcome = (
        f"SUCCEEDED\n{result.summary}"
        if result.ok
        else f"FAILED\n{result.error}\n"
        "A failed tool call is recoverable. Say what should be done differently."
    )

    flags = []
    if plateau:
        flags.append(
            "PLATEAU: the score moved less than the configured minimum improvement. "
            "Repeating this kind of edit will not work."
        )
    if observation.iterations_remaining <= 1:
        flags.append(
            f"BUDGET: only {observation.iterations_remaining} iteration(s) remain "
            "after this one."
        )
    flag_block = ("\n" + "\n".join(f"  ! {f}" for f in flags) + "\n") if flags else ""

    return [
        system(SYSTEM),
        user(
            f"ITERATION {observation.iteration} of {observation.max_iterations}\n"
            f"Rubric: {observation.rubric.name} (criteria: {', '.join(observation.rubric.ids)})\n"
            f"Target: {target_score:.0f}%\n\n"
            f"THE AGENT'S THOUGHT\n  {decision.thought or '(none recorded)'}\n\n"
            f"THE ACTION\n  {decision.action}({decision.arguments})\n\n"
            f"THE OUTCOME: {outcome}\n\n"
            f"SCORE MOVEMENT: {movement}\n"
            f"{flag_block}\n"
            "Reflect on this iteration now."
        ),
    ]


__all__ = ["SYSTEM", "build_messages", "build_spec"]
