"""A stateful offline stand-in for the whole agent.

:class:`MockProvider` replays a tape. This gives it a brain: a responder that
inspects each request, works out which step of the loop is asking, and answers
in character -- judging, revising, reasoning or reflecting -- while tracking a
simulated quality model across the run.

It exists for three reasons:

* **The end-to-end test needs no API key.** CI runs the full loop, asserts the
  score climbs across iterations and that termination is clean, at zero cost and
  with zero flakiness.
* **The demo works offline.** ``--provider mock`` shows a complete run even with
  no key, no network and no free-tier quota left.
* **Failure injection has somewhere to live.** ``fail_on`` makes any step fail
  on demand, which is how the harness's recovery paths get demonstrated.

It is a simulation and is labelled as one. The scores it produces are generated
by a rule, not by reading the text. What it faithfully reproduces is the
*shape* of a run: revisions raise the criteria they target, the agent must
re-score to learn that, and the loop terminates on the real rules.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass, field

from ..core.rubric import Rubric
from .mock import MockCall, MockTurn, tool_call
from .types import LLMError

_TEXT_BLOCK = re.compile(
    r"-----\s*CURRENT TEXT\s*-----\s*\n(.*?)\n-----\s*END OF TEXT\s*-----",
    re.DOTALL,
)

#: Recalled lessons reach the Reason prompt as
#: ``  - [lesson | session ab12cd34, iter 2 | relevance 0.47] <content>``.
_RECALLED_LESSON = re.compile(r"^\s*-\s*\[lesson\s*\|[^\]]*\]\s*(.+)$", re.MULTILINE)

#: How much a targeted criterion improves per revision, and how much the rest
#: drift up as a side effect of a general rewrite. Fractional so the simulated
#: trajectory is smooth; the judge reports rounded integers, as a real one would.
#: Tuned so the canned demo reaches target inside the default 6-iteration cap.
FOCUS_GAIN = 2.0
SPILLOVER_GAIN = 0.6

STEP_JUDGE = "judge"
STEP_REFLECT = "reflect"
STEP_REASON = "reason"
STEP_REVISE = "revise"


@dataclass
class ScriptedAgentResponder:
    """Simulates a competent agent working through one rubric.

    Which loop step is calling is inferred from the tools offered, not from a
    marker planted in the prompt: the judge is the only caller that offers
    ``submit_rubric_scores``, Reflect the only one offering
    ``submit_reflection``, Reason offers the agent tool set, and the reviser
    offers no tools at all. That keeps the production prompts free of test
    scaffolding.
    """

    rubric: Rubric
    target_score: float = 85.0
    #: Step name -> exception to raise the first time that step is called.
    fail_on: dict[str, Exception] = field(default_factory=dict)

    scores: dict[str, float] = field(default_factory=dict)
    pending_focus: list[str] = field(default_factory=list)
    last_percent: float | None = None
    needs_scoring: bool = True
    revisions: int = 0
    explored: bool = False
    steps_seen: list[str] = field(default_factory=list)
    lessons_emitted: list[str] = field(default_factory=list)
    lessons_applied: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scores:
            # A deliberately weak starting point, worst on the heaviest criteria.
            ranked = sorted(self.rubric.criteria, key=lambda c: -c.weight)
            self.scores = {
                criterion.id: 2.0 if index < 2 else 2.5 + (index % 2) * 0.5
                for index, criterion in enumerate(ranked)
            }

    # -- simulated quality model -------------------------------------------

    def _rounded(self) -> dict[str, int]:
        scale = self.rubric.scale
        return {
            criterion_id: int(round(scale.clamp(value)))
            for criterion_id, value in self.scores.items()
        }

    def _percent(self) -> float:
        rounded = self._rounded()
        return 100.0 * sum(
            criterion.weight * self.rubric.scale.normalise(rounded[criterion.id])
            for criterion in self.rubric.criteria
        )

    def _headroom_ranked(self) -> list[str]:
        rounded = self._rounded()
        return [
            criterion.id
            for criterion in sorted(
                self.rubric.criteria,
                key=lambda c: -(c.weight * (1.0 - self.rubric.scale.normalise(rounded[c.id]))),
            )
        ]

    def _apply_revision(self, focus: t.Sequence[str]) -> None:
        for criterion in self.rubric.criteria:
            gain = FOCUS_GAIN if criterion.id in focus else SPILLOVER_GAIN
            self.scores[criterion.id] = self.rubric.scale.clamp(self.scores[criterion.id] + gain)
        self.revisions += 1
        self.needs_scoring = True

    # -- step routing -------------------------------------------------------

    @staticmethod
    def recalled_lessons(call: MockCall) -> list[str]:
        """Pull lessons out of the RECALLED FROM MEMORY block, if there is one.

        The responder reads the prompt the same way a real model would, rather
        than being handed the memory object. That is what makes the A/B demo
        meaningful: the *only* difference between the two runs is what appeared
        in the prompt.
        """
        prompt = call.messages[-1].content if call.messages else ""
        if "RECALLED FROM MEMORY" not in prompt:
            return []
        return [match.strip() for match in _RECALLED_LESSON.findall(prompt)]

    @staticmethod
    def classify(call: MockCall) -> str:
        names = {spec.name for spec in call.tools}
        if "submit_rubric_scores" in names:
            return STEP_JUDGE
        if "submit_reflection" in names:
            return STEP_REFLECT
        if names:
            return STEP_REASON
        return STEP_REVISE

    def __call__(self, call: MockCall) -> MockTurn:
        step = self.classify(call)
        self.steps_seen.append(step)

        injected = self.fail_on.pop(step, None)
        if injected is not None:
            return MockTurn(raises=injected)

        if step == STEP_JUDGE:
            return self._judge()
        if step == STEP_REFLECT:
            return self._reflect()
        if step == STEP_REVISE:
            return self._revise(call)
        return self._reason(call)

    # -- per-step behaviour -------------------------------------------------

    def _judge(self) -> MockTurn:
        rounded = self._rounded()
        self.needs_scoring = False
        self.last_percent = self._percent()
        return MockTurn(
            tool_calls=(
                tool_call(
                    "submit_rubric_scores",
                    scores=[
                        {
                            "criterion_id": criterion.id,
                            "evidence": f"(simulated) representative line for {criterion.id}",
                            "justification": (
                                f"simulated grader: {criterion.name} sits at "
                                f"{rounded[criterion.id]}/{self.rubric.scale.max} "
                                f"after {self.revisions} revision(s)"
                            ),
                            "score": rounded[criterion.id],
                        }
                        for criterion in self.rubric.criteria
                    ],
                    notes="simulated judge; scores follow a rule, not the text",
                ),
            ),
        )

    def _reason(self, call: MockCall) -> MockTurn:
        if self.needs_scoring or self.last_percent is None:
            return MockTurn(
                tool_calls=(
                    tool_call(
                        "score_against_rubric",
                        thought=(
                            "The draft is unscored, so I cannot tell where the points are. "
                            "Measure first."
                        ),
                    ),
                )
            )

        if self.last_percent >= self.target_score:
            return MockTurn(
                tool_calls=(
                    tool_call(
                        "finalize",
                        thought=(
                            f"At {self.last_percent:.1f}% the draft is past the "
                            f"{self.target_score:.0f}% target; further edits risk drift."
                        ),
                        reason=(
                            f"Target met at {self.last_percent:.1f}% after "
                            f"{self.revisions} revision(s)."
                        ),
                    ),
                )
            )

        lessons = self.recalled_lessons(call)

        # THE MEMORY EFFECT, effect 1 of 2: exploration is skipped when the
        # agent already knows where the points are. With no prior experience it
        # spends a turn measuring the draft before committing to an edit; with a
        # recalled lesson it goes straight to revising. That is one whole
        # iteration saved, and it is visible in the transcript.
        if not lessons and not self.explored:
            self.explored = True
            return MockTurn(
                tool_calls=(
                    tool_call(
                        "analyze_text",
                        thought=(
                            "No prior experience with this rubric was recalled. Measure the "
                            "draft directly before spending a revision on a guess."
                        ),
                    ),
                )
            )

        focus = self._headroom_ranked()[:2]
        self.pending_focus = focus
        names = ", ".join(self.rubric.criterion(c).name for c in focus)

        # THE MEMORY EFFECT, effect 2 of 2: recalled lessons are handed to the
        # reviser as apply_lessons, so they reach the rewrite prompt itself.
        arguments: dict[str, t.Any] = {
            "focus_criteria": focus,
            "instructions": (
                f"Rewrite to strengthen {names}: make the claim specific, attribute "
                "every figure, and cut hedging."
            ),
        }
        if lessons:
            arguments["apply_lessons"] = lessons[:2]
            self.lessons_applied.extend(lessons[:2])
            thought = (
                f"Memory says: {lessons[0][:80]}... Applying that directly to "
                f"{names} rather than rediscovering it."
            )
        else:
            thought = (
                f"{names} carry the most weighted headroom at "
                f"{self.last_percent:.1f}%, so editing there buys the most points."
            )

        return MockTurn(tool_calls=(tool_call("revise_text", thought=thought, **arguments),))

    def _revise(self, call: MockCall) -> MockTurn:
        source = call.messages[-1].content
        match = _TEXT_BLOCK.search(source)
        current = match.group(1).strip() if match else source
        focus = self.pending_focus or self._headroom_ranked()[:1]
        self._apply_revision(focus)

        labels = ", ".join(self.rubric.criterion(c).name for c in focus)
        # Appending keeps the edit visible to diff_drafts and keeps the word
        # count growing, so the "returned a summary" guard is not tripped.
        addition = (
            f"\n\nRevision {self.revisions} targeted {labels}. According to a 2023 "
            "industry survey, 62 percent of respondents reported the effect described "
            "above; critics argue the sample skews toward larger firms, which is a fair "
            "objection and one this draft now addresses directly."
        )
        return MockTurn(text=current + addition)

    def _reflect(self) -> MockTurn:
        percent = self.last_percent
        ranked = self._headroom_ranked()
        next_focus = "" if percent is not None and percent >= self.target_score else ranked[0]

        # Emit each lesson at most once. A real Reflect returns an empty lesson
        # on most turns, and repeating one would put duplicates into memory.
        candidate = {
            1: (
                "On this rubric, targeting the two highest-weighted criteria first moves the "
                "total faster than fixing the lowest raw score."
            ),
            2: (
                "Unattributed figures score no better than no figures; naming the source is "
                "what lifts the evidence criterion."
            ),
        }.get(self.revisions, "")
        lesson = "" if candidate in self.lessons_emitted else candidate
        if lesson:
            self.lessons_emitted.append(lesson)
        return MockTurn(
            tool_calls=(
                tool_call(
                    "submit_reflection",
                    critique=(
                        f"(simulated) the run is at "
                        f"{percent:.1f}%" if percent is not None else "(simulated) unscored"
                    )
                    + f" after {self.revisions} revision(s).",
                    lesson=lesson,
                    next_focus=next_focus,
                    task_complete=bool(percent is not None and percent >= self.target_score),
                    reason=(
                        "target met" if next_focus == "" else f"{next_focus} still has headroom"
                    ),
                ),
            )
        )


def make_failure(kind: str, message: str = "") -> Exception:
    """Build an injectable failure by name, for ``--simulate-failure``."""
    from .types import (
        LLMParseError,
        ProviderUnavailableError,
        RateLimitError,
        TransientServerError,
    )

    registry: dict[str, Exception] = {
        "rate_limit": RateLimitError(
            message or "429 Too Many Requests (injected)", provider="mock", status=429,
            retry_after_s=1.0,
        ),
        "bad_json": LLMParseError(
            message or "tool arguments were not valid JSON (injected)",
            raw="{unterminated",
            provider="mock",
        ),
        "server_error": TransientServerError(
            message or "503 Service Unavailable (injected)", provider="mock", status=503
        ),
        "provider_down": ProviderUnavailableError(
            message or "connection refused (injected)", provider="mock"
        ),
    }
    if kind not in registry:
        raise ValueError(f"unknown failure kind {kind!r}; known: {sorted(registry)}")
    return registry[kind]


__all__ = [
    "FOCUS_GAIN",
    "SPILLOVER_GAIN",
    "STEP_JUDGE",
    "STEP_REASON",
    "STEP_REFLECT",
    "STEP_REVISE",
    "LLMError",
    "ScriptedAgentResponder",
    "make_failure",
]
