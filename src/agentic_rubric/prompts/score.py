"""Prompt and schema for the LLM judge behind ``score_against_rubric``.

The judge is chain-of-thought with a receipt. Its output schema orders the
fields ``evidence -> justification -> score``, and models fill a JSON object in
schema order, so the quote and the argument are generated *before* the number.
Reversing that order lets the model pick a number first and rationalise it
after, which is exactly the failure mode that makes LLM judges drift.

Two more anti-drift measures:

* The judge sees the rubric's own level descriptors, so "3" means what the
  rubric says it means rather than what the model feels.
* It is told to score the draft as written, not as intended. An improvement
  loop whose judge is generous will report progress that is not there.
"""

from __future__ import annotations

import typing as t

from ..core.rubric import Rubric
from ..llm.types import Message, ToolSpec, system, user

#: Named separately because it is the wire signature the step is recognised by.
#: See :func:`..prompts.classify_step`.
SUBMIT_SCORES_TOOL = "submit_rubric_scores"

SUBMIT_SCORES = ToolSpec(
    name=SUBMIT_SCORES_TOOL,
    description="Submit one score per rubric criterion, with the evidence behind each.",
    parameters={
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": "Exactly one entry per criterion, in rubric order.",
                "items": {
                    "type": "object",
                    # Order matters: evidence and justification are generated
                    # before the score, so the number follows the argument.
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "evidence": {
                            "type": "string",
                            "description": (
                                "A short verbatim quote from the text that justifies this "
                                "score. Use an empty string only if the relevant content is "
                                "entirely absent."
                            ),
                        },
                        "justification": {
                            "type": "string",
                            "description": (
                                "One or two sentences tying the evidence to the level "
                                "descriptor you are applying."
                            ),
                        },
                        "score": {
                            "type": "integer",
                            "description": "The level from the rubric scale.",
                        },
                    },
                    "required": ["criterion_id", "evidence", "justification", "score"],
                },
            },
            "notes": {
                "type": "string",
                "description": "Optional overall observation, one sentence.",
            },
        },
        "required": ["scores"],
    },
)

SYSTEM = """You are a strict, consistent rubric grader.

Rules you must follow:
1. Score the text AS WRITTEN, never as you imagine it was intended.
2. For every criterion, quote the evidence first, then justify, then score.
   If you cannot quote supporting text, the score cannot be above the midpoint.
3. Use the rubric level descriptors literally. Do not invent intermediate
   standards of your own.
4. Be consistent across calls: the same text and rubric must produce the same
   scores. You are a measuring instrument, not a coach.
5. Score every criterion exactly once, using the criterion ids given.

Call submit_rubric_scores exactly once. Do not reply with prose."""


def build_messages(
    rubric: Rubric,
    draft: str,
    *,
    focus_criteria: t.Sequence[str] | None = None,
    deterministic_evidence: str = "",
) -> list[Message]:
    """Assemble the judge conversation."""
    focus_note = ""
    if focus_criteria:
        focus_note = (
            "\nThe caller is most interested in: "
            + ", ".join(focus_criteria)
            + ". Score every criterion anyway, but be especially careful on those.\n"
        )

    evidence_block = ""
    if deterministic_evidence:
        evidence_block = (
            "\nAutomated checks on this text (these are measured facts, not "
            "opinions -- do not contradict them):\n"
            f"{deterministic_evidence}\n"
        )

    return [
        system(SYSTEM),
        user(
            f"{rubric.render()}\n"
            f"{focus_note}"
            f"{evidence_block}\n"
            "----- TEXT TO SCORE -----\n"
            f"{draft}\n"
            "----- END OF TEXT -----\n\n"
            f"Score all {len(rubric.criteria)} criteria: {', '.join(rubric.ids)}.\n"
            f"Each score is an integer from {rubric.scale.min} to {rubric.scale.max}."
        ),
    ]


__all__ = ["SUBMIT_SCORES", "SUBMIT_SCORES_TOOL", "SYSTEM", "build_messages"]
