"""Rubrics and scorecards -- the measurable signal the whole loop turns on.

A rubric is *data*, loaded from YAML. Swapping `essay_argumentative.yaml` for
`bug_report.yaml` changes the agent's entire domain without touching a line of
loop code, which is the point: the loop scores, targets a weakness, revises and
re-scores, and none of those steps know what kind of text they are looking at.

Two design decisions worth calling out:

* **Scores are normalised to a percentage.** ``weighted_percent`` maps the
  rubric's own scale onto 0-100 so ``loop.target_score`` means the same thing
  for a 1-5 rubric and a 0-10 one.
* **"Weakest" means most weighted headroom, not lowest score.** A 25%-weighted
  criterion sitting at 2/5 is worth more than a 15%-weighted one at 1/5. Chasing
  raw low scores would send the agent after cheap points; chasing headroom sends
  it after the highest-value edit available.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml

WEIGHT_SUM_TOLERANCE = 0.01


class RubricError(ValueError):
    """Raised when a rubric file is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class Scale:
    """The integer range a criterion is scored on."""

    min: int = 1
    max: int = 5

    def __post_init__(self) -> None:
        if self.max <= self.min:
            raise RubricError(f"scale.max ({self.max}) must exceed scale.min ({self.min})")

    @property
    def span(self) -> int:
        return self.max - self.min

    def clamp(self, value: float) -> float:
        return max(float(self.min), min(float(self.max), float(value)))

    def normalise(self, value: float) -> float:
        """Map a raw score onto 0.0-1.0."""
        return (self.clamp(value) - self.min) / self.span


@dataclass(frozen=True)
class Probe:
    """A deterministic regex check declared by the rubric.

    Probes are how a rubric expresses a structural expectation that does not
    need an LLM to verify -- "numbered reproduction steps exist", "no hedging
    phrases". They give the agent grounded, non-hallucinated evidence, and they
    are declared alongside the criterion they inform rather than hardcoded.
    """

    id: str
    pattern: str
    describe: str
    expect: t.Literal["present", "absent"] = "present"
    min_count: int = 1

    def run(self, text: str) -> ProbeResult:
        try:
            matches = re.findall(self.pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise RubricError(f"probe {self.id!r} has an invalid pattern: {exc}") from exc
        count = len(matches)
        passed = count >= self.min_count if self.expect == "present" else count == 0
        return ProbeResult(id=self.id, describe=self.describe, count=count,
                           expect=self.expect, passed=passed)


@dataclass(frozen=True)
class ProbeResult:
    id: str
    describe: str
    count: int
    expect: str
    passed: bool

    def summary(self) -> str:
        verdict = "ok" if self.passed else "MISSING" if self.expect == "present" else "PRESENT"
        return f"{self.describe}: {verdict} (matches={self.count}, expected {self.expect})"


@dataclass(frozen=True)
class RubricCriterion:
    """One weighted dimension of the rubric."""

    id: str
    name: str
    weight: float
    description: str
    levels: dict[int, str] = field(default_factory=dict)
    improvement_hints: tuple[str, ...] = ()
    probes: tuple[Probe, ...] = ()

    def render(self, scale: Scale) -> str:
        """Human-readable block for the judge and reviser prompts."""
        lines = [
            f"### {self.name}  (id: {self.id}, weight: {self.weight:.0%})",
            self.description.strip(),
        ]
        if self.levels:
            lines.append("Level descriptors:")
            lines.extend(
                f"  {level}/{scale.max}: {text.strip()}"
                for level, text in sorted(self.levels.items())
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Rubric:
    """A complete, validated rubric."""

    id: str
    name: str
    criteria: tuple[RubricCriterion, ...]
    scale: Scale = field(default_factory=Scale)
    domain: str = ""
    description: str = ""
    target_score: float | None = None
    source_path: str = ""

    def __post_init__(self) -> None:
        if not self.criteria:
            raise RubricError(f"rubric {self.id!r} has no criteria")
        ids = [c.id for c in self.criteria]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise RubricError(
                f"rubric {self.id!r} has duplicate criterion ids: {sorted(duplicates)}"
            )
        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise RubricError(
                f"rubric {self.id!r} weights sum to {total:.3f}, expected 1.0 "
                f"(+/- {WEIGHT_SUM_TOLERANCE})"
            )
        for criterion in self.criteria:
            for level in criterion.levels:
                if not self.scale.min <= level <= self.scale.max:
                    raise RubricError(
                        f"criterion {criterion.id!r} declares level {level}, "
                        f"outside scale {self.scale.min}-{self.scale.max}"
                    )

    # -- lookup -------------------------------------------------------------

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.criteria)

    def criterion(self, criterion_id: str) -> RubricCriterion:
        for candidate in self.criteria:
            if candidate.id == criterion_id:
                return candidate
        raise RubricError(f"unknown criterion {criterion_id!r}; rubric has {list(self.ids)}")

    @property
    def all_probes(self) -> tuple[Probe, ...]:
        return tuple(probe for c in self.criteria for probe in c.probes)

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        header = f"RUBRIC: {self.name} (scored {self.scale.min}-{self.scale.max} per criterion)"
        if self.description:
            header += f"\n{self.description.strip()}"
        blocks = [c.render(self.scale) for c in self.criteria]
        return header + "\n\n" + "\n\n".join(blocks)

    def hints_for(self, criterion_ids: t.Sequence[str]) -> list[str]:
        out: list[str] = []
        for criterion_id in criterion_ids:
            criterion = self.criterion(criterion_id)
            out.extend(f"[{criterion.id}] {hint}" for hint in criterion.improvement_hints)
        return out

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: t.Mapping[str, t.Any], source_path: str = "") -> Rubric:
        try:
            raw_scale = data.get("scale") or {}
            scale = Scale(min=int(raw_scale.get("min", 1)), max=int(raw_scale.get("max", 5)))
            criteria = tuple(
                RubricCriterion(
                    id=str(item["id"]),
                    name=str(item.get("name", item["id"])),
                    weight=float(item["weight"]),
                    description=str(item.get("description", "")),
                    levels={int(k): str(v) for k, v in (item.get("levels") or {}).items()},
                    improvement_hints=tuple(str(h) for h in (item.get("improvement_hints") or ())),
                    probes=tuple(
                        Probe(
                            id=str(p["id"]),
                            pattern=str(p["pattern"]),
                            describe=str(p.get("describe", p["id"])),
                            expect=p.get("expect", "present"),
                            min_count=int(p.get("min_count", 1)),
                        )
                        for p in (item.get("probes") or ())
                    ),
                )
                for item in data["criteria"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            where = f" at {source_path}" if source_path else ""
            raise RubricError(f"malformed rubric{where}: {exc}") from exc

        target = data.get("target_score")
        return cls(
            id=str(data.get("id", "rubric")),
            name=str(data.get("name", data.get("id", "Rubric"))),
            criteria=criteria,
            scale=scale,
            domain=str(data.get("domain", "")),
            description=str(data.get("description", "")),
            target_score=float(target) if target is not None else None,
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Rubric:
        rubric_path = Path(path)
        if not rubric_path.exists():
            raise RubricError(f"rubric file not found: {rubric_path}")
        loaded = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RubricError(f"{rubric_path}: top level must be a mapping")
        return cls.from_dict(loaded, source_path=str(rubric_path))


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionScore:
    """One criterion's verdict, with the evidence that justifies it.

    ``evidence`` holds a short verbatim quote from the text. Requiring the judge
    to quote before it scores is chain-of-thought with a receipt: it makes an
    unjustifiable score visibly unjustifiable rather than merely wrong.
    """

    criterion_id: str
    score: float
    justification: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class ScoreCard:
    """A full set of criterion scores plus the maths the loop steers by.

    Carries its own weights and scale so it stays self-contained and
    serialisable -- a trace entry should be interpretable without also loading
    the rubric that produced it.
    """

    rubric_id: str
    scores: tuple[CriterionScore, ...]
    weights: dict[str, float]
    scale: Scale
    iteration: int = 0
    judge_notes: str = ""

    @classmethod
    def build(
        cls,
        rubric: Rubric,
        scores: t.Sequence[CriterionScore],
        *,
        iteration: int = 0,
        judge_notes: str = "",
    ) -> ScoreCard:
        """Construct from a rubric, filling in any criterion the judge skipped.

        A judge that omits a criterion gets it scored at the scale minimum
        rather than silently dropped, so a partial verdict can never inflate the
        percentage by shrinking the denominator.
        """
        by_id = {s.criterion_id: s for s in scores}
        unknown = set(by_id) - set(rubric.ids)
        if unknown:
            raise RubricError(f"judge returned unknown criteria: {sorted(unknown)}")
        complete = tuple(
            by_id.get(
                criterion.id,
                CriterionScore(
                    criterion_id=criterion.id,
                    score=float(rubric.scale.min),
                    justification="not scored by the judge; defaulted to the scale minimum",
                ),
            )
            for criterion in rubric.criteria
        )
        return cls(
            rubric_id=rubric.id,
            scores=complete,
            weights={c.id: c.weight for c in rubric.criteria},
            scale=rubric.scale,
            iteration=iteration,
            judge_notes=judge_notes,
        )

    # -- maths --------------------------------------------------------------

    def weighted_percent(self) -> float:
        """Weighted score normalised to 0-100 against the rubric's own scale."""
        return 100.0 * sum(
            self.weights.get(s.criterion_id, 0.0) * self.scale.normalise(s.score)
            for s in self.scores
        )

    def headroom(self, criterion_id: str) -> float:
        """Percentage points still available on this criterion.

        This is the quantity the agent should be maximising per edit.
        """
        score = self.score_for(criterion_id)
        weight = self.weights.get(criterion_id, 0.0)
        return 100.0 * weight * (1.0 - self.scale.normalise(score))

    def score_for(self, criterion_id: str) -> float:
        for entry in self.scores:
            if entry.criterion_id == criterion_id:
                return entry.score
        raise RubricError(f"scorecard has no entry for {criterion_id!r}")

    def ranked_by_headroom(self) -> list[CriterionScore]:
        """Criteria ordered by the value of fixing them, most valuable first."""
        return sorted(self.scores, key=lambda s: -self.headroom(s.criterion_id))

    def weakest(self, n: int = 1) -> list[CriterionScore]:
        return self.ranked_by_headroom()[: max(1, n)]

    def delta_from(self, previous: ScoreCard | None) -> float | None:
        if previous is None:
            return None
        return self.weighted_percent() - previous.weighted_percent()

    # -- presentation -------------------------------------------------------

    def render_table(self) -> str:
        rows = [
            f"{'criterion':<18} {'score':>7} {'weight':>7} {'headroom':>9}",
            "-" * 44,
        ]
        for entry in self.ranked_by_headroom():
            rows.append(
                f"{entry.criterion_id:<18} "
                f"{entry.score:>4.1f}/{self.scale.max:<2} "
                f"{self.weights.get(entry.criterion_id, 0.0):>6.0%} "
                f"{self.headroom(entry.criterion_id):>8.1f}pts"
            )
        rows.append("-" * 44)
        rows.append(f"{'WEIGHTED TOTAL':<18} {self.weighted_percent():>6.1f}%")
        return "\n".join(rows)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "rubric_id": self.rubric_id,
            "iteration": self.iteration,
            "weighted_percent": round(self.weighted_percent(), 2),
            "scale": {"min": self.scale.min, "max": self.scale.max},
            "scores": [
                {
                    "criterion_id": s.criterion_id,
                    "score": s.score,
                    "weight": self.weights.get(s.criterion_id, 0.0),
                    "headroom": round(self.headroom(s.criterion_id), 2),
                    "justification": s.justification,
                    "evidence": s.evidence,
                }
                for s in self.ranked_by_headroom()
            ],
            "judge_notes": self.judge_notes,
        }


__all__ = [
    "CriterionScore",
    "Probe",
    "ProbeResult",
    "Rubric",
    "RubricCriterion",
    "RubricError",
    "Scale",
    "ScoreCard",
]
