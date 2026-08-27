"""The agent's tool set, assembled against a specific rubric.

Schemas are built *from the rubric*, not written as constants. Every argument
that names a criterion carries an ``enum`` of that rubric's actual criterion
ids, so the model cannot ask to improve a dimension that does not exist -- the
class of error is designed out rather than validated after the fact. It also
means the same code produces a different, correct tool set for the essay rubric
and the bug-report rubric.

Five tools, chosen so the agent has a real decision to make each turn:

============================  ======  ==================================
tool                          LLM?    what it is for
============================  ======  ==================================
``score_against_rubric``      yes     measure -- the loop's only signal
``revise_text``               yes     act -- the only tool that edits
``analyze_text``              no      ground truth about the current draft
``diff_drafts``               no      verify the last edit was real
``finalize``                  no      request termination
============================  ======  ==================================

Two of the five never call a model. That is deliberate: an agent whose every
tool is another prompt has no way to check itself.
"""

from __future__ import annotations

from ..core.rubric import Rubric
from .handlers import analysis, control, revision, scoring
from .registry import ToolRegistry, build_spec


def build_registry(rubric: Rubric) -> ToolRegistry:
    """Construct the tool set for one rubric."""
    criterion_ids = list(rubric.ids)
    criterion_enum = {
        "type": "string",
        "enum": criterion_ids,
        "description": f"A criterion id from the {rubric.name} rubric.",
    }
    registry = ToolRegistry()

    registry.register(
        build_spec(
            name="score_against_rubric",
            description=(
                "Score the CURRENT draft against every rubric criterion and record the "
                "result. This is the only way to learn how good the text is, and the only "
                "way to see whether your last revision helped. You do not pass the text: "
                "the tool always scores whatever the working draft currently is. Call this "
                "first, and again after every revision."
            ),
            properties={
                "focus_criteria": {
                    "type": "array",
                    "items": criterion_enum,
                    "description": (
                        "Optional. Criteria you most want scrutinised. All criteria are "
                        "scored regardless; this only sharpens the judge's attention."
                    ),
                }
            },
        ),
        scoring.handle,
    )

    registry.register(
        build_spec(
            name="revise_text",
            description=(
                "Rewrite the CURRENT draft to improve the named criteria. This is the only "
                "tool that changes the text. Target the criteria with the most headroom "
                "(weight x remaining points), not simply the lowest raw score. Give a "
                "concrete instruction, not a restatement of the criterion. After revising, "
                "the draft is unscored until you score it again."
            ),
            properties={
                "focus_criteria": {
                    "type": "array",
                    "items": criterion_enum,
                    "minItems": 1,
                    "description": (
                        "One or two criteria to improve. Targeting everything at once "
                        "produces a shallow rewrite that moves nothing."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "minLength": 15,
                    "description": (
                        "A specific editing instruction, e.g. 'replace the vague opening "
                        "with a one-sentence claim that remote work reduces attrition, and "
                        "attribute the productivity claim to a named study'."
                    ),
                },
                "max_words": {
                    "type": "integer",
                    "minimum": 50,
                    "description": "Optional hard ceiling on the revised draft length.",
                },
                "apply_lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Lessons recalled from memory that the reviser should "
                        "apply to this edit."
                    ),
                },
            },
            required=["focus_criteria", "instructions"],
        ),
        revision.handle,
    )

    registry.register(
        build_spec(
            name="analyze_text",
            description=(
                "Measure the CURRENT draft without an LLM: readability, sentence-length "
                "variance, hedging and filler counts, and the rubric's own structural "
                "checks. These are facts, not opinions. Use it when you need to know "
                "whether a stylistic problem is real before spending a revision on it."
            ),
            properties={
                "criteria": {
                    "type": "array",
                    "items": criterion_enum,
                    "description": "Optional. Restrict structural checks to these criteria.",
                }
            },
        ),
        analysis.handle_analyze,
    )

    registry.register(
        build_spec(
            name="diff_drafts",
            description=(
                "Show what actually changed between the previous draft and the current "
                "one. Use it when a revision did not move the score as much as expected, "
                "to tell 'the edit was too small' apart from 'the edit was wrong'."
            ),
            properties={},
        ),
        analysis.handle_diff,
    )

    registry.register(
        build_spec(
            name="finalize",
            description=(
                "Declare the work complete and stop. Only valid once the current draft has "
                "been scored. Use it when the target is met, or when you judge that further "
                "revision cannot help. Your request is advisory: the loop confirms it "
                "against the actual scores."
            ),
            properties={
                "reason": {
                    "type": "string",
                    "minLength": 10,
                    "description": "Why the work is complete, referring to the scores.",
                }
            },
            required=["reason"],
        ),
        control.handle,
        terminal=True,
    )

    return registry


__all__ = ["build_registry"]
