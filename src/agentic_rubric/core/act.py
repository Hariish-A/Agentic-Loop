"""ACT -- execute the chosen action. Makes no decisions.

Thin on purpose. Everything interesting about tool execution lives in the
registry (schema validation, error containment, timing); Act's job is to be the
single, obvious place where a Decision becomes an ActionResult, and to attribute
the token cost of LLM-backed tools to the iteration that spent it.

The step exists as its own module even though it is short, because collapsing it
into the loop would blur the boundary the challenge asks for: Reason decides,
Act executes, and neither does the other's job. Keeping Act incapable of
choosing is what makes that separation real rather than stylistic.
"""

from __future__ import annotations

from ..llm.types import Usage
from ..tools.registry import ToolContext, ToolRegistry
from .state import ActionResult, Decision


def act(decision: Decision, registry: ToolRegistry, ctx: ToolContext) -> tuple[ActionResult, Usage]:
    """Run the decided action and report what it cost.

    Returns the result plus the token usage the tool itself incurred, measured
    as the delta on the shared context. LLM-backed tools (the judge, the
    reviser) can spend more than the Reason call that chose them, so attributing
    it here is what makes the per-iteration token numbers in the trace honest.
    """
    before = ctx.usage
    result = registry.dispatch(decision, ctx)
    tool_usage = Usage(
        input_tokens=ctx.usage.input_tokens - before.input_tokens,
        output_tokens=ctx.usage.output_tokens - before.output_tokens,
        estimated=ctx.usage.estimated,
    )
    return result, tool_usage


__all__ = ["act"]
