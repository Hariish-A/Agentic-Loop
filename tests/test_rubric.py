"""Rubric loading, validation, probes and scorecard maths."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rubric.core.rubric import (
    CriterionScore,
    Probe,
    Rubric,
    RubricError,
    Scale,
    ScoreCard,
)

ESSAY = "config/rubrics/essay_argumentative.yaml"
BUG = "config/rubrics/bug_report.yaml"


def minimal(**overrides: object) -> dict:
    data = {
        "id": "test",
        "name": "Test",
        "scale": {"min": 1, "max": 5},
        "criteria": [
            {"id": "a", "name": "A", "weight": 0.6, "description": "first"},
            {"id": "b", "name": "B", "weight": 0.4, "description": "second"},
        ],
    }
    data.update(overrides)
    return data


# --- loading and validation ------------------------------------------------


@pytest.mark.parametrize("path", [ESSAY, BUG])
def test_shipped_rubrics_load(path: str) -> None:
    rubric = Rubric.from_yaml(path)
    assert len(rubric.criteria) == 5
    assert abs(sum(c.weight for c in rubric.criteria) - 1.0) < 0.01
    assert rubric.target_score == 85.0
    assert rubric.all_probes


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(RubricError, match="weights sum to"):
        Rubric.from_dict(
            minimal(criteria=[{"id": "a", "name": "A", "weight": 0.6, "description": ""}])
        )


def test_duplicate_criterion_ids_are_rejected() -> None:
    with pytest.raises(RubricError, match="duplicate criterion ids"):
        Rubric.from_dict(
            minimal(
                criteria=[
                    {"id": "a", "name": "A", "weight": 0.5, "description": ""},
                    {"id": "a", "name": "A2", "weight": 0.5, "description": ""},
                ]
            )
        )


def test_level_outside_the_scale_is_rejected() -> None:
    with pytest.raises(RubricError, match="outside scale"):
        Rubric.from_dict(
            minimal(
                criteria=[
                    {"id": "a", "name": "A", "weight": 1.0, "description": "", "levels": {9: "x"}}
                ]
            )
        )


def test_missing_file_names_the_path() -> None:
    with pytest.raises(RubricError, match="not found"):
        Rubric.from_yaml("config/rubrics/nope.yaml")


def test_unknown_criterion_lookup_lists_the_real_ones() -> None:
    with pytest.raises(RubricError, match=r"\['a', 'b'\]"):
        Rubric.from_dict(minimal()).criterion("zzz")


# --- probes ----------------------------------------------------------------


def test_present_probe_respects_min_count() -> None:
    probe = Probe(id="p", pattern=r"\d+", describe="digits", expect="present", min_count=2)
    assert probe.run("one 1 and two 2").passed
    assert not probe.run("only 1").passed


def test_absent_probe_passes_only_on_zero_matches() -> None:
    probe = Probe(id="p", pattern=r"\bobviously\b", describe="no blame", expect="absent")
    assert probe.run("the service returns 500").passed
    assert not probe.run("obviously nobody tested this").passed


def test_invalid_pattern_is_a_rubric_error() -> None:
    with pytest.raises(RubricError, match="invalid pattern"):
        Probe(id="p", pattern="(unclosed", describe="broken").run("text")


def test_shipped_probes_all_fire_on_the_weak_samples() -> None:
    # The samples are deliberately bad; a probe that passes on them is not
    # measuring anything useful.
    samples = ((ESSAY, "samples/weak_essay.txt"), (BUG, "samples/weak_bug_report.txt"))
    for rubric_path, sample in samples:
        rubric = Rubric.from_yaml(rubric_path)
        text = Path(sample).read_text(encoding="utf-8")
        results = [probe.run(text) for probe in rubric.all_probes]
        assert results and not any(r.passed for r in results), rubric_path


# --- scorecard maths -------------------------------------------------------


def card(**scores: float) -> ScoreCard:
    rubric = Rubric.from_dict(minimal())
    return ScoreCard.build(
        rubric, [CriterionScore(criterion_id=k, score=v) for k, v in scores.items()]
    )


def test_scale_minimum_is_zero_percent_and_maximum_is_one_hundred() -> None:
    assert card(a=1, b=1).weighted_percent() == pytest.approx(0.0)
    assert card(a=5, b=5).weighted_percent() == pytest.approx(100.0)


def test_weighting_is_applied() -> None:
    # a is worth 60%, b 40%. a at max, b at min -> 60%.
    assert card(a=5, b=1).weighted_percent() == pytest.approx(60.0)


def test_headroom_is_weight_times_remaining_not_raw_score() -> None:
    # b scores lower (1 vs 2), but a is weighted 0.6 against b's 0.4, so a
    # carries more recoverable points. Ranking by raw score would send the
    # agent after the cheaper fix.
    result = card(a=2, b=1)
    assert result.headroom("a") == pytest.approx(45.0)
    assert result.headroom("b") == pytest.approx(40.0)
    assert result.weakest(1)[0].criterion_id == "a"


def test_missing_criteria_default_to_the_scale_minimum() -> None:
    # A partial verdict must not inflate the percentage by shrinking the
    # denominator, so anything the judge skipped scores the minimum.
    partial = card(a=5)
    assert partial.score_for("b") == 1
    assert partial.weighted_percent() == pytest.approx(60.0)
    assert "not scored" in dict((s.criterion_id, s.justification) for s in partial.scores)["b"]


def test_unknown_criteria_are_rejected_at_build_time() -> None:
    rubric = Rubric.from_dict(minimal())
    with pytest.raises(RubricError, match="unknown criteria"):
        ScoreCard.build(rubric, [CriterionScore(criterion_id="ghost", score=3)])


def test_delta_needs_a_previous_card() -> None:
    first = card(a=2, b=2)
    second = card(a=4, b=4)
    assert first.delta_from(None) is None
    assert second.delta_from(first) == pytest.approx(50.0)


def test_scale_rejects_an_inverted_range() -> None:
    with pytest.raises(RubricError, match="must exceed"):
        Scale(min=5, max=1)


def test_scorecard_serialises_with_headroom_ordering() -> None:
    payload = card(a=3, b=5).to_dict()
    assert payload["scores"][0]["criterion_id"] == "a"  # most headroom first
    assert payload["weighted_percent"] == pytest.approx(70.0)
