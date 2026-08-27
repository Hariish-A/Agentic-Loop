"""REASON -- one LLM call that chooses the next action. The only step that decides.

Tool use is *forced* (``tool_choice="required"``), so the model cannot answer
with prose. Combined with the rubric-derived enums in the tool schemas, the
space of possible outputs is narrow enough that most malformed-output failure
modes are designed out rather than caught.

The remaining ones are caught here. If the model returns prose anyway -- some
providers ignore ``required`` under load -- :func:`fallback_decision` picks a
safe, useful action instead of crashing the run, and marks the Decision
``degraded=True`` so the trace records that the agent did not really choose it.
Milestone 3 adds a repair round trip in front of this fallback; the fallback
itself stays as the floor beneath it.
"""

from __future__ import annotations

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.types import LLMParseError
from ..prompts import reason as reason_prompt
from ..tools.registry import THOUGHT_FIELD, ToolRegistry
from .state import Decision, Observation

#: Actions the fallback may choose. Both are read-only: a degraded decision must
#: never be one that rewrites the document.
SAFE_FALLBACK_ACTIONS = ("score_against_rubric", "analyze_text")


def fallback_decision(observation: Observation, registry: ToolRegistry, reason: str) -> Decision:
    """Pick a safe action when the model did not give a usable one.

    Never revises. A degraded decision is one we do not fully understand, and
    the wrong response to not understanding the situation is to start editing
    the user's text.
    """
    if not observation.has_been_scored and "score_against_rubric" in registry:
        action = "score_against_rubric"
    elif "analyze_text" in registry:
        action = "analyze_text"
    else:
        action = registry.names[0]

    return Decision(
        action=action,
        arguments={},
        thought=f"[fallback] {reason}; taking the safe read-only action {action}.",
        degraded=True,
    )


def reason(
    observation: Observation,
    provider: LLMProvider,
    registry: ToolRegistry,
    config: AppConfig,
) -> Decision:
    """Choose exactly one next action."""
    messages = reason_prompt.build_messages(observation, registry.describe())

    try:
        response = provider.complete(
            messages,
            tools=registry.specs(),
            tool_choice="required",
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except LLMParseError as exc:
        # The call succeeded but the payload was unusable (e.g. tool arguments
        # that were not JSON). Retrying identical input rarely helps; degrade.
        return fallback_decision(observation, registry, f"unparseable model output: {exc}")

    try:
        call = response.first_tool_call()
    except LLMParseError:
        decision = fallback_decision(
            observation, registry, "the model replied with prose instead of a tool call"
        )
        return Decision(
            action=decision.action,
            arguments=decision.arguments,
            thought=decision.thought,
            raw_text=response.text,
            degraded=True,
            usage=response.usage,
        )

    arguments = dict(call.arguments)
    thought = str(arguments.pop(THOUGHT_FIELD, "")).strip()
    if not thought:
        # Providers occasionally drop a required property. Fall back to any
        # prose the model emitted alongside the call before giving up on it.
        thought = response.text.strip() or "(no thought recorded)"

    return Decision(
        action=call.name,
        arguments=arguments,
        thought=thought,
        call_id=call.id,
        raw_text=response.text,
        usage=response.usage,
    )


__all__ = ["SAFE_FALLBACK_ACTIONS", "fallback_decision", "reason"]
