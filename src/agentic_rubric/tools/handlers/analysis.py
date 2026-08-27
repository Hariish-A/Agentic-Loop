"""``analyze_text`` and ``diff_drafts`` -- the two tools that never call an LLM.

They exist to give the agent facts it cannot argue with. ``analyze_text``
reports readability statistics plus the rubric's own declared probes;
``diff_drafts`` reports what a revision actually changed.

``diff_drafts`` in particular is a check on the agent's self-report. A reviser
that returned a near-identical draft, or that quietly deleted a third of the
document, will be described in the ReAct scratchpad as "revised for evidence"
either way. The diff is the only thing in the loop that can contradict that.
"""

from __future__ import annotations

import typing as t

from ..registry import ToolContext, ToolError, ToolOutput
from ..text_stats import compute_metrics, unified_diff_summary
from .scoring import run_probes


def handle_analyze(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
    draft = ctx.workspace.draft
    if not draft.strip():
        raise ToolError("the draft is empty; there is nothing to analyse")

    metrics = compute_metrics(draft)
    results = run_probes(ctx.rubric, draft)

    wanted = {str(c) for c in arguments.get("criteria", [])}
    if wanted:
        unknown = wanted - set(ctx.rubric.ids)
        if unknown:
            raise ToolError(
                f"unknown criteria {sorted(unknown)}; this rubric has {list(ctx.rubric.ids)}"
            )
        allowed = {
            probe.id
            for criterion in ctx.rubric.criteria
            if criterion.id in wanted
            for probe in criterion.probes
        }
        results = tuple(r for r in results if r.id in allowed)

    failing = [r for r in results if not r.passed]
    probe_note = (
        f"{len(failing)}/{len(results)} structural checks failing: "
        + "; ".join(r.describe for r in failing)
        if failing
        else (f"all {len(results)} structural checks pass" if results else "no probes declared")
    )

    return ToolOutput(
        summary=f"{metrics.render()}. {probe_note}.",
        payload={
            "metrics": metrics.to_dict(),
            "probes": [
                {
                    "id": r.id,
                    "describe": r.describe,
                    "expect": r.expect,
                    "count": r.count,
                    "passed": r.passed,
                }
                for r in results
            ],
            "failing_probe_ids": [r.id for r in failing],
        },
    )


def handle_diff(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
    previous = ctx.workspace.previous_draft
    if previous is None:
        raise ToolError(
            "there is no previous draft to compare against; the text has not been revised yet"
        )

    diff = unified_diff_summary(ctx.workspace.draft, previous)
    # unified_diff_summary(before, after) was called with the current draft as
    # `before`, so flip the labels back to previous -> current.
    forward = unified_diff_summary(previous, ctx.workspace.draft)

    verdict = (
        "the revision made no meaningful change"
        if not forward["changed"]
        else f"{forward['lines_added']} line(s) added, {forward['lines_removed']} removed"
    )
    added = t.cast("list[str]", forward["added_sample"])

    return ToolOutput(
        summary=(
            f"Diff vs previous draft: {verdict}; "
            f"{forward['words_before']} -> {forward['words_after']} words "
            f"({forward['word_delta']:+d}), similarity {forward['similarity']:.2f}."
        ),
        payload={
            "diff": forward,
            "reverse_similarity": diff["similarity"],
            "added_preview": added[:5],
        },
    )


__all__ = ["handle_analyze", "handle_diff"]
