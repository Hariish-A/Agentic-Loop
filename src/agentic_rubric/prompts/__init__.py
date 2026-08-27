"""Prompt templates, one module per LLM-touching step.

Kept out of the step implementations so a prompt can be inspected, diffed and
reasoned about on its own. Each module exposes `build_messages(...)` and, where
the step needs structured output, the `ToolSpec` that forces it.
"""

from __future__ import annotations

import typing as t

from . import reason, reflect, revise, score

#: The four LLM-touching call sites, named.
STEP_JUDGE = "judge"
STEP_REFLECT = "reflect"
STEP_REASON = "reason"
STEP_REVISE = "revise"

STEPS = (STEP_REASON, STEP_JUDGE, STEP_REVISE, STEP_REFLECT)

#: The tool that uniquely identifies a step's request on the wire.
_SIGNATURE_TOOL = {
    score.SUBMIT_SCORES_TOOL: STEP_JUDGE,
    reflect.SUBMIT_REFLECTION_TOOL: STEP_REFLECT,
}


def classify_step(tool_names: t.Iterable[str]) -> str:
    """Which loop step issued a completion request, from the tools it offered.

    Inferred from the request rather than from a marker planted in the prompt,
    so the production prompts stay free of scaffolding that exists only for
    tests and demos. Defined once, here, because it is a fact about *these
    prompts*: change which tools a step offers and this is the one place that
    has to follow.

    Used by the offline responder to answer in character, and by the harness's
    fault injection to target a named step against a live provider.
    """
    names = set(tool_names)
    for tool, step in _SIGNATURE_TOOL.items():
        if tool in names:
            return step
    # Reason offers the agent's tool set; the reviser offers nothing at all.
    return STEP_REASON if names else STEP_REVISE


__all__ = [
    "STEPS",
    "STEP_JUDGE",
    "STEP_REASON",
    "STEP_REFLECT",
    "STEP_REVISE",
    "classify_step",
    "reason",
    "reflect",
    "revise",
    "score",
]
