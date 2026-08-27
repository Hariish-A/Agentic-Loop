"""``score_against_rubric`` -- the LLM judge, and the loop's source of signal.

Note what this tool does *not* take: the text. It scores whatever is currently
in the workspace. Passing the draft as an argument would force the model to
re-emit the entire document on every call -- thousands of wasted tokens, plus a
standing opportunity to corrupt the text in transit. Tools that operate on the
working document take a reference to it, not a copy.

:func:`judge` is separated from the handler so the shallow Tree-of-Thoughts
branch in ``revise_text`` can score candidate drafts without any of them being
recorded as the agent's actual progress.
"""

from __future__ import annotations

import typing as t

from ...core.rubric import CriterionScore, ProbeResult, Rubric, RubricError, ScoreCard
from ...llm.types import LLMError, LLMParseError
from ...prompts import score as score_prompt
from ..registry import ToolContext, ToolError, ToolOutput

# The judge must be reproducible: the same text and rubric should score the
# same twice. Sampling noise in a measuring instrument is indistinguishable
# from progress, and this loop steers on the difference between two scores.
JUDGE_TEMPERATURE = 0.0


def run_probes(rubric: Rubric, text: str) -> tuple[ProbeResult, ...]:
    return tuple(probe.run(text) for probe in rubric.all_probes)


def probe_evidence(results: t.Sequence[ProbeResult]) -> str:
    """Render probe outcomes for the judge prompt, failures first."""
    if not results:
        return ""
    ordered = sorted(results, key=lambda r: r.passed)
    return "\n".join(f"  - {r.summary()}" for r in ordered)


def judge(
    ctx: ToolContext,
    text: str,
    *,
    focus_criteria: t.Sequence[str] | None = None,
    iteration: int | None = None,
) -> ScoreCard:
    """Score arbitrary text against the context rubric. No side effects.

    Raises :class:`ToolError` on any failure, so a judge problem surfaces as a
    failed action the agent can see rather than an exception that ends the run.
    """
    if not text.strip():
        raise ToolError("cannot score an empty draft")

    results = run_probes(ctx.rubric, text)
    messages = score_prompt.build_messages(
        ctx.rubric,
        text,
        focus_criteria=focus_criteria,
        deterministic_evidence=probe_evidence(results),
    )

    try:
        response = ctx.llm.complete(
            messages,
            tools=[score_prompt.SUBMIT_SCORES],
            tool_choice=score_prompt.SUBMIT_SCORES.name,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=ctx.config.llm.max_tokens,
        )
    except LLMError as exc:
        raise ToolError(f"judge call failed: {exc}") from exc

    ctx.add_usage(response.usage)

    try:
        call = response.first_tool_call()
    except LLMParseError as exc:
        raise ToolError(f"judge returned prose instead of scores: {exc}") from exc

    raw_scores = call.arguments.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise ToolError("judge returned no scores")

    valid_ids = set(ctx.rubric.ids)
    scores: list[CriterionScore] = []
    skipped: list[str] = []
    for entry in raw_scores:
        if not isinstance(entry, dict):
            continue
        criterion_id = str(entry.get("criterion_id", "")).strip()
        if criterion_id not in valid_ids:
            # Drop rather than fail: one hallucinated id should not throw away
            # four correct scores. ScoreCard.build defaults anything missing.
            skipped.append(criterion_id or "<blank>")
            continue
        try:
            raw_value = float(entry.get("score"))
        except (TypeError, ValueError):
            skipped.append(criterion_id)
            continue
        scores.append(
            CriterionScore(
                criterion_id=criterion_id,
                score=ctx.rubric.scale.clamp(raw_value),
                justification=str(entry.get("justification", "")).strip(),
                evidence=str(entry.get("evidence", "")).strip(),
            )
        )

    if not scores:
        raise ToolError(f"judge produced no usable scores (rejected: {skipped})")

    notes = str(call.arguments.get("notes", "")).strip()
    if skipped:
        notes = (notes + f" [rejected unrecognised criteria: {skipped}]").strip()

    try:
        return ScoreCard.build(
            ctx.rubric,
            scores,
            iteration=ctx.iteration if iteration is None else iteration,
            judge_notes=notes,
        )
    except RubricError as exc:
        raise ToolError(str(exc)) from exc


def handle(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
    """Score the current draft and record the result as the agent's progress."""
    focus = arguments.get("focus_criteria") or None
    previous = ctx.workspace.scorecard

    card = judge(ctx, ctx.workspace.draft, focus_criteria=focus)
    ctx.workspace.record_score(card)

    percent = card.weighted_percent()
    delta = card.delta_from(previous)
    weakest = card.weakest(2)
    weakest_text = ", ".join(
        f"{s.criterion_id} {s.score:.0f}/{card.scale.max} "
        f"({card.headroom(s.criterion_id):.1f}pts available)"
        for s in weakest
    )
    movement = f", {delta:+.1f}pts vs previous" if delta is not None else ""

    return ToolOutput(
        summary=(
            f"Scored {percent:.1f}%{movement}. "
            f"Biggest opportunities: {weakest_text}."
        ),
        payload={
            "scorecard": card.to_dict(),
            "weighted_percent": round(percent, 2),
            "delta": round(delta, 2) if delta is not None else None,
            "weakest": [s.criterion_id for s in weakest],
        },
    )


__all__ = ["JUDGE_TEMPERATURE", "handle", "judge", "probe_evidence", "run_probes"]
