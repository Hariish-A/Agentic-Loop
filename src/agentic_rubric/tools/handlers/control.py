"""``finalize`` -- the agent's request to stop.

Registered as a *terminal* tool, but note what it does not do: it does not end
the run by itself. It sets a flag on the workspace, and Reflect decides whether
to honour it. An agent that can unilaterally declare victory will, especially
when the remaining work is hard.

The tool refuses to finalize an unscored draft. Stopping without ever having
measured the thing is the one outcome this loop must never produce.
"""

from __future__ import annotations

import typing as t

from ..registry import ToolContext, ToolError, ToolOutput


def handle(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
    reason = str(arguments.get("reason", "")).strip()
    if not reason:
        raise ToolError("finalize requires a reason explaining why the work is complete")

    card = ctx.workspace.scorecard
    if card is None:
        raise ToolError(
            "the current draft has not been scored, so it cannot be finalized. "
            "Call score_against_rubric first."
        )

    ctx.workspace.finalized = True
    ctx.workspace.finalize_reason = reason
    percent = card.weighted_percent()

    return ToolOutput(
        summary=f"Requested finalization at {percent:.1f}%: {reason}",
        payload={
            "reason": reason,
            "weighted_percent": round(percent, 2),
            "scorecard": card.to_dict(),
        },
    )


__all__ = ["handle"]
