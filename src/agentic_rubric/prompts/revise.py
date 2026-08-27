"""Prompt for the LLM reviser behind ``revise_text``.

Returns plain text, not JSON. That is deliberate: asking a model to embed a
multi-paragraph document inside a JSON string invites escaping bugs, truncation
at the quote boundary, and a whole class of parse failures for zero benefit --
there is exactly one field to return.

The prompt is written to fight the two ways an improvement loop degrades:

* **Drift.** Told to preserve meaning and voice, and to change only what the
  named criteria require. Without this the text slowly becomes the model's own
  essay rather than an improved version of the author's.
* **Padding.** Told that length is not the goal and given a word ceiling. A
  reviser rewarded for "more detail" will inflate the draft indefinitely, which
  raises no rubric score but burns the token budget.
"""

from __future__ import annotations

import re
import typing as t

from ..core.rubric import Rubric
from ..llm.types import Message, system, user

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n:]*:|revised (?:text|version|draft)[^\n:]*:|sure[^\n]*:)\s*\n",
    re.IGNORECASE,
)

SYSTEM = """You are a careful editor working against an explicit rubric.

Rules:
1. Preserve the author's argument, factual claims and voice. You are improving
   this text, not replacing it with your own.
2. Change what the named criteria require, and leave the rest alone.
3. Do not invent facts, statistics, sources or quotations. If the rubric asks
   for evidence the text does not contain, add a clearly marked placeholder
   such as [SOURCE NEEDED: ...] rather than fabricating one.
4. Length is not the goal. A shorter, sharper draft beats a longer one.
5. Return ONLY the revised text. No preamble, no explanation, no code fences,
   no commentary about what you changed."""


def build_messages(
    rubric: Rubric,
    draft: str,
    *,
    focus_criteria: t.Sequence[str],
    instructions: str,
    max_words: int | None = None,
    lessons: t.Sequence[str] = (),
) -> list[Message]:
    """Assemble the reviser conversation."""
    criteria_blocks = []
    for criterion_id in focus_criteria:
        criterion = rubric.criterion(criterion_id)
        block = criterion.render(rubric.scale)
        hints = criterion.improvement_hints
        if hints:
            block += "\nHow this criterion is usually improved:\n" + "\n".join(
                f"  - {hint}" for hint in hints
            )
        criteria_blocks.append(block)

    ceiling = ""
    if max_words:
        ceiling = f"\nHard ceiling: {max_words} words. Cut before you add.\n"

    lesson_block = ""
    if lessons:
        lesson_block = (
            "\nLessons carried over from earlier work (apply these):\n"
            + "\n".join(f"  - {lesson}" for lesson in lessons)
            + "\n"
        )

    return [
        system(SYSTEM),
        user(
            "Improve the text below. Target these criteria and only these:\n\n"
            + "\n\n".join(criteria_blocks)
            + f"\n\nSpecific instruction from the agent:\n{instructions.strip()}\n"
            + ceiling
            + lesson_block
            + "\n----- CURRENT TEXT -----\n"
            + draft
            + "\n----- END OF TEXT -----\n\n"
            "Return the full revised text only."
        ),
    ]


def clean_output(text: str) -> str:
    """Strip the wrappers models add despite being told not to.

    Cheaper and more reliable than another round trip to ask for it again.
    """
    cleaned = text.strip()
    fenced = _FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    cleaned = _PREAMBLE.sub("", cleaned)
    return cleaned.strip()


__all__ = ["SYSTEM", "build_messages", "clean_output"]
