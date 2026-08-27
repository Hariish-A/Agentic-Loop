"""The admission gate: is this text gradeable at all?

Run **once, before the loop starts, with no LLM call**. It exists because of a
real failure observed against a live provider: given the input ``hi, good
morning``, the agent scored it 0.0%, then "revised" three words into a 170-word
essay with 0.00 similarity to the input, scored its own invention at 85.0%, and
reported ``target_reached``. Roughly 17,000 tokens and eight rate-limit retries
were spent fabricating a document the user never wrote.

That is the worst failure available to this product. Every other failure mode
either stops the run or degrades it visibly; this one *succeeds loudly* while
silently replacing the user's text. The gate closes it at the cheapest possible
point -- before the first token is spent.

Two design choices worth defending:

**It is deterministic.** Asking a model "is this gradeable?" would cost a call,
could be wrong, and could be talked out of its answer by the next prompt. Word
and sentence counts cannot.

**It lives outside the four steps.** Admission is not a cognitive step: it does
not perceive, reason, act or reflect. Putting it in Perceive would re-run it on
every iteration against a draft the agent has since legitimately grown, and
would blur a boundary the whole design is built on.

Thresholds are rubric-declared with a config default, because "long enough to
grade" is a property of the rubric, not of the agent. An essay rubric needs a
few paragraphs; a bug report can be graded shorter.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ..config import AppConfig
from ..tools.text_stats import sentences, words
from .rubric import Rubric


@dataclass(frozen=True)
class AdmissionCheck:
    """One gate condition and how the submitted text fared against it."""

    name: str
    passed: bool
    detail: str
    observed: float = 0.0
    required: float = 0.0

    def render(self) -> str:
        mark = "ok" if self.passed else "FAILED"
        return f"{self.name}: {mark} - {self.detail}"


@dataclass(frozen=True)
class AdmissionVerdict:
    """Whether the text may enter the loop, and why not if it may not."""

    ok: bool
    reason: str = ""
    checks: tuple[AdmissionCheck, ...] = ()
    measurements: dict[str, t.Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[AdmissionCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "measurements": self.measurements,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "observed": c.observed,
                    "required": c.required,
                }
                for c in self.checks
            ],
        }


def thresholds(rubric: Rubric, config: AppConfig) -> tuple[int, int]:
    """Resolve ``(min_words, min_sentences)``, rubric first, then config."""
    min_words = (
        rubric.min_words if rubric.min_words is not None else config.loop.min_input_words
    )
    min_sentences = (
        rubric.min_sentences
        if rubric.min_sentences is not None
        else config.loop.min_input_sentences
    )
    return max(0, int(min_words)), max(0, int(min_sentences))


def check_admission(text: str, rubric: Rubric, config: AppConfig) -> AdmissionVerdict:
    """Decide whether ``text`` is worth spending a model call on.

    Never raises, never calls a model. A rejection is a normal outcome carrying
    an explanation the user can act on, not an error.
    """
    stripped = text.strip()
    word_list = words(stripped)
    sentence_list = sentences(stripped) if stripped else []
    word_count = len(word_list)
    sentence_count = len(sentence_list)

    min_words, min_sentences = thresholds(rubric, config)
    measurements = {
        "words": word_count,
        "sentences": sentence_count,
        "characters": len(stripped),
        "min_words": min_words,
        "min_sentences": min_sentences,
        "rubric_id": rubric.id,
    }

    checks: list[AdmissionCheck] = [
        AdmissionCheck(
            name="not_empty",
            passed=bool(stripped),
            detail="the submission is empty" if not stripped else "text was submitted",
            observed=float(len(stripped)),
            required=1.0,
        )
    ]

    if stripped:
        checks.append(
            AdmissionCheck(
                name="min_words",
                passed=word_count >= min_words,
                detail=(
                    f"{word_count} words; the {rubric.name} rubric needs at least {min_words}"
                ),
                observed=float(word_count),
                required=float(min_words),
            )
        )
        checks.append(
            AdmissionCheck(
                name="min_sentences",
                passed=sentence_count >= min_sentences,
                detail=(
                    f"{sentence_count} sentence(s); the {rubric.name} rubric needs at "
                    f"least {min_sentences}"
                ),
                observed=float(sentence_count),
                required=float(min_sentences),
            )
        )

    failed = [check for check in checks if not check.passed]
    if not failed:
        return AdmissionVerdict(ok=True, checks=tuple(checks), measurements=measurements)

    return AdmissionVerdict(
        ok=False,
        reason=explain(failed, rubric, word_count, min_words),
        checks=tuple(checks),
        measurements=measurements,
    )


def explain(
    failed: t.Sequence[AdmissionCheck], rubric: Rubric, word_count: int, min_words: int
) -> str:
    """A message for a person, not a log line.

    It has to say what was wrong *and* what to do about it, because the only
    reader is someone who just pasted the wrong thing into a box.
    """
    if any(check.name == "not_empty" for check in failed):
        return "Nothing was submitted. Paste the text you want scored."

    shortfall = ", ".join(check.detail for check in failed)
    return (
        f"This text cannot be scored against the {rubric.name} rubric: {shortfall}. "
        f"The rubric grades {rubric.domain or 'documents'} on "
        f"{len(rubric.criteria)} criteria such as "
        f"{', '.join(c.name for c in rubric.criteria[:2])} - a submission of "
        f"{word_count} word(s) has nothing for those criteria to measure. "
        "Submit a longer draft, or choose a rubric that fits this text."
    )


__all__ = ["AdmissionCheck", "AdmissionVerdict", "check_admission", "explain", "thresholds"]
