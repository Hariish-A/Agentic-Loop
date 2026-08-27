"""``revise_text`` -- the only tool that changes the document.

Contains the optional **shallow Tree-of-Thoughts** branch. When
``loop.revise_candidates > 1`` the tool generates N independent revisions at
spread temperatures, scores each with the judge, and keeps the winner. That is a
one-level search tree with value-based pruning: real ToT machinery, honestly
labelled as one level deep rather than dressed up as the full method. It costs
N revise calls plus N judge calls per iteration, which is why it defaults to 1.

Two guards matter more than they look:

* **A revision that changed nothing is a failure, not a success.** Without the
  similarity check the agent can loop forever reporting work it did not do.
* **A revision that lost most of the document is a failure.** Models
  occasionally return a summary instead of a rewrite; silently accepting it
  would destroy the author's text.
"""

from __future__ import annotations

import typing as t

from ...core.rubric import ScoreCard
from ...llm.types import LLMError
from ...prompts import revise as revise_prompt
from ..registry import ToolContext, ToolError, ToolOutput
from ..text_stats import unified_diff_summary, words
from .scoring import judge

#: Below this fraction of the original word count, the output is a summary.
MIN_RETAINED_WORD_RATIO = 0.35
#: Above this similarity, nothing meaningful changed.
NO_CHANGE_SIMILARITY = 0.995
#: Above this multiple of the original word count, the output is a composition
#: rather than a revision. The bound that matters: a live run turned a
#: three-word greeting into 170 words -- a 56x expansion -- and reported it as
#: an improvement. Growth is still allowed, just one bounded step per iteration,
#: because each iteration re-baselines against the draft it starts from.
MAX_EXPANSION_RATIO = 3.0
#: Fraction of the original's distinct words that must survive into the
#: revision. Only applied above :data:`RETENTION_FLOOR_MIN_WORDS`, because on a
#: very short original the denominator is too small to mean anything.
MIN_WORD_RETENTION = 0.25
RETENTION_FLOOR_MIN_WORDS = 25
#: Temperature spread used when generating multiple candidates. A single
#: candidate runs at the first value.
CANDIDATE_TEMPERATURES = (0.4, 0.7, 0.9, 1.0)


def _generate(
    ctx: ToolContext,
    *,
    focus: list[str],
    instructions: str,
    max_words: int | None,
    lessons: t.Sequence[str],
    temperature: float,
) -> str:
    messages = revise_prompt.build_messages(
        ctx.rubric,
        ctx.workspace.draft,
        focus_criteria=focus,
        instructions=instructions,
        max_words=max_words,
        lessons=lessons,
    )
    try:
        response = ctx.llm.complete(
            messages,
            temperature=temperature,
            max_tokens=ctx.config.llm.max_tokens,
        )
    except LLMError as exc:
        raise ToolError(f"reviser call failed: {exc}") from exc

    ctx.add_usage(response.usage)
    return revise_prompt.clean_output(response.text)


def word_retention(original: str, candidate: str) -> float:
    """Fraction of the original's distinct words that survive the revision.

    Case-insensitive set overlap rather than sequence similarity, because
    ``difflib`` compares *lines*: a legitimate paragraph-level rewrite and a
    fabricated-from-nothing draft both score near zero there, so line
    similarity cannot tell them apart. Vocabulary survival can -- a real
    revision keeps the subject matter even when every sentence is rewritten.
    """
    before = {word.lower() for word in words(original)}
    if not before:
        return 0.0
    after = {word.lower() for word in words(candidate)}
    return len(before & after) / len(before)


def _validate_candidate(original: str, candidate: str) -> str | None:
    """Return a rejection reason, or ``None`` if the candidate is usable.

    The guards are deliberately **symmetric**. Before, only shrinkage and
    no-change were caught, so a reviser that replaced the draft with something
    far longer and entirely unrelated passed every check -- which is exactly
    what happened on a live run against a three-word input.
    """
    if not candidate.strip():
        return "the reviser returned an empty draft"

    original_words = len(words(original))
    candidate_words = len(words(candidate))

    if original_words and candidate_words / original_words < MIN_RETAINED_WORD_RATIO:
        return (
            f"the reviser returned {candidate_words} words against an original of "
            f"{original_words}; that is a summary, not a revision"
        )

    if original_words and candidate_words / original_words > MAX_EXPANSION_RATIO:
        return (
            f"the reviser returned {candidate_words} words against an original of "
            f"{original_words} ({candidate_words / original_words:.1f}x); that is a "
            "composition, not a revision"
        )

    retention = word_retention(original, candidate)
    if original_words >= RETENTION_FLOOR_MIN_WORDS and retention < MIN_WORD_RETENTION:
        return (
            f"only {retention:.0%} of the original wording survives; the reviser "
            "replaced the draft rather than improving it"
        )

    similarity = unified_diff_summary(original, candidate).get("similarity")
    if isinstance(similarity, (int, float)) and similarity >= NO_CHANGE_SIMILARITY:
        return "the reviser returned text identical to the input; no edit was made"
    return None


def handle(arguments: dict[str, t.Any], ctx: ToolContext) -> ToolOutput:
    focus = [str(c) for c in arguments.get("focus_criteria", [])]
    instructions = str(arguments.get("instructions", "")).strip()

    valid = set(ctx.rubric.ids)
    unknown = [c for c in focus if c not in valid]
    if unknown:
        raise ToolError(
            f"unknown criteria {unknown}; this rubric has {sorted(valid)}"
        )
    if not focus:
        raise ToolError("focus_criteria must name at least one criterion to improve")

    max_words = arguments.get("max_words")
    max_words = int(max_words) if max_words else None
    lessons = [str(item) for item in arguments.get("apply_lessons", [])]

    original = ctx.workspace.draft
    wanted = max(1, int(ctx.config.loop.revise_candidates))

    candidates: list[str] = []
    rejections: list[str] = []
    for index in range(wanted):
        temperature = CANDIDATE_TEMPERATURES[min(index, len(CANDIDATE_TEMPERATURES) - 1)]
        candidate = _generate(
            ctx,
            focus=focus,
            instructions=instructions,
            max_words=max_words,
            lessons=lessons,
            temperature=temperature,
        )
        reason = _validate_candidate(original, candidate)
        if reason:
            rejections.append(f"candidate {index + 1}: {reason}")
            continue
        candidates.append(candidate)

    if not candidates:
        raise ToolError("; ".join(rejections) or "the reviser produced nothing usable")

    chosen = candidates[0]
    selection_note = ""
    scored: list[tuple[float, str]] = []

    if len(candidates) > 1:
        # The pruning half of the shallow tree: score every branch, keep the best.
        for candidate in candidates:
            try:
                card: ScoreCard = judge(ctx, candidate, focus_criteria=focus)
            except ToolError:
                continue  # a candidate we cannot score cannot win
            scored.append((card.weighted_percent(), candidate))
        if scored:
            scored.sort(key=lambda pair: -pair[0])
            best_score, chosen = scored[0]
            spread = best_score - scored[-1][0]
            selection_note = (
                f" Chose the best of {len(scored)} candidates "
                f"({best_score:.1f}%, spread {spread:.1f}pts)."
            )

    ctx.workspace.replace_draft(chosen)
    # The scorecard now describes the previous draft, so retire it. Leaving it
    # in place would let the next Reason step steer on a stale number.
    ctx.workspace.scorecard = None

    diff = unified_diff_summary(original, chosen)
    return ToolOutput(
        summary=(
            f"Revised for {', '.join(focus)}: "
            f"{diff['words_before']} -> {diff['words_after']} words "
            f"({diff['word_delta']:+d}), similarity {diff['similarity']:.2f}."
            f"{selection_note} The draft is now unscored - score it to see the effect."
        ),
        payload={
            "focus_criteria": focus,
            "instructions": instructions,
            "diff": diff,
            "candidates_generated": len(candidates),
            "candidates_rejected": rejections,
            "candidate_scores": [round(s, 2) for s, _ in scored],
        },
    )


__all__ = ["CANDIDATE_TEMPERATURES", "MIN_RETAINED_WORD_RATIO", "handle"]
