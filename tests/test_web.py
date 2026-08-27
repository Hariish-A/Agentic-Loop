"""The application server: payload assembly, the streaming run, and the seams.

The HTTP plumbing is stdlib and not worth testing; what matters is that the
browser is handed correct data and that a run produces the event sequence the UI
renders.

The server itself has **no simulated provider** — that was removed on purpose,
because a UI able to quietly show invented scores eventually will. These tests
substitute one at the `chain_factory` seam instead, which is the same
substitution every other test in this project makes, and is a parameter rather
than a mode: nothing the browser can send reaches it.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

from agentic_rubric.config import AppConfig, load_config
from agentic_rubric.core.state import Decision, Workspace
from agentic_rubric.harness.fallbacks import ProviderChain
from agentic_rubric.llm.base import LLMProvider
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder
from agentic_rubric.llm.mock import MockProvider
from agentic_rubric.llm.types import ProviderUnavailableError
from agentic_rubric.memory.base import MemoryRecord, NullMemory
from agentic_rubric.tools.definitions import build_registry
from agentic_rubric.tools.registry import ToolContext
from agentic_rubric.web import server

CONFIG = "config/config.yaml"
# Long enough to clear the essay rubric's admission floor (60 words, 3
# sentences) and still deliberately weak: hedged throughout, no figures, no
# attribution, no counterargument. The gate is about admissibility, not quality.
DRAFT = (
    "Remote work is somewhat complicated. It could be said that there are many "
    "views on the subject, and arguably both sides have points worth considering "
    "in various ways. Some companies have moved to remote work and some have not, "
    "which shows that there is not really a consensus. Workers seem to like it, "
    "and it is generally believed that productivity may be affected in various "
    "ways. Collaboration is a thing that some managers worry about, and company "
    "culture is often brought up too. In conclusion, organisations will need to "
    "decide for themselves what works best."
)


def offline_chain(rubric_id: str = "essay_argumentative", **responder_kwargs: t.Any):
    """A `chain_factory` that yields a deterministic provider instead of a live one."""

    def factory(
        config: AppConfig, requested: str
    ) -> tuple[LLMProvider, ProviderChain, list[str]]:
        rubric = server.load_rubrics()[rubric_id]
        responder = ScriptedAgentResponder(
            rubric=rubric, target_score=config.loop.target_score, **responder_kwargs
        )
        provider = MockProvider(responder=responder, name="offline")
        return provider, ProviderChain.of(provider), ["substituted provider for the test"]

    return factory


def collect(body: dict[str, t.Any], **kwargs: t.Any) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    kwargs.setdefault(
        "chain_factory", offline_chain(body.get("rubric_id") or "essay_argumentative")
    )
    server.run_loop_streaming(body, lambda name, data: events.append((name, data)), **kwargs)
    return events


def names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


def final(events: list[tuple[str, dict]]) -> dict:
    for name, data in reversed(events):
        if name == "complete":
            return data
    raise AssertionError(f"run did not complete; events were {names(events)}")


def base_body(tmp_path: Path, **overrides: t.Any) -> dict[str, t.Any]:
    body: dict[str, t.Any] = {
        "rubric_id": "essay_argumentative",
        "text": DRAFT,
        "target_score": 85,
        "max_iterations": 8,
        "memory_enabled": False,
        "session_id": "test",
    }
    body.update(overrides)
    return body


# ===========================================================================
# Bootstrap payloads
# ===========================================================================


def test_rubrics_load_with_criteria_and_probes() -> None:
    rubrics = server.load_rubrics()
    assert "essay_argumentative" in rubrics and "bug_report" in rubrics
    payload = server.rubric_payload(rubrics["essay_argumentative"])
    assert len(payload["criteria"]) == 5
    assert sum(c["weight"] for c in payload["criteria"]) == pytest.approx(1.0)
    assert any(c["probes"] for c in payload["criteria"])


def test_presets_are_kept_and_scoped_to_their_rubric() -> None:
    """The preset picker must never offer a bug report to the essay rubric."""
    samples = server.sample_payload()
    assert samples["essay_argumentative"] and samples["bug_report"]
    for entry in samples["essay_argumentative"]:
        assert "essay" in entry["id"]
        assert entry["text"].strip()


def test_the_provider_chain_is_reported_with_a_reason() -> None:
    rows = server.provider_payload(load_config(CONFIG))
    assert rows, "the chain should never be empty"
    assert all({"name", "model", "available", "reason", "position"} <= set(r) for r in rows)
    # The simulated provider is not offered by the application at all.
    assert all(row["name"] != "mock" for row in rows)


def test_tools_are_reported_as_the_model_sees_them() -> None:
    tools = server.tool_payload(server.load_rubrics()["essay_argumentative"])
    by_name = {t["name"]: t for t in tools}
    assert len(tools) == 5
    # Two of the five never call a model; the panel says which.
    assert by_name["analyze_text"]["uses_llm"] is False
    assert by_name["diff_drafts"]["uses_llm"] is False
    assert by_name["revise_text"]["uses_llm"] is True
    assert by_name["finalize"]["terminal"] is True
    assert "thought" not in by_name["revise_text"]["arguments"]


def test_build_chain_refuses_to_fall_back_to_a_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key anywhere must be an error, not a silent switch to fake scores."""
    # Every provider in the shipped chain, so a key present in the
    # developer's real environment cannot make this pass by accident.
    for variable in ("GEMINI_API_KEY", "GROQ_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    config = load_config(CONFIG, overrides={"llm.providers.ollama.requires_key": True})
    with pytest.raises(ProviderUnavailableError) as caught:
        server.build_chain(config, "")
    assert "GROQ_API_KEY" in str(caught.value)


# ===========================================================================
# The streaming run
# ===========================================================================


def test_a_run_emits_the_four_steps_for_every_iteration(tmp_path: Path) -> None:
    events = collect(base_body(tmp_path))
    result = final(events)
    order = names(events)

    assert order[0] == "note" or order[0] == "run_start"
    assert "run_start" in order and "complete" in order
    for step in ("perceive", "reason", "act", "reflect"):
        assert order.count(step) == result["iterations"], step
    assert result["iterations"] >= 3
    assert result["best_score"] > result["initial_score"]


def test_the_run_goes_through_the_harness_not_the_bare_loop(tmp_path: Path) -> None:
    """The application must report retries and traces, which only the Runner has."""
    result = final(collect(base_body(tmp_path)))
    assert "harness" in result
    assert set(result["harness"]) >= {"retries", "repairs", "failovers", "provider"}
    assert "guardrails" in result
    assert result["guardrails"]["tokens"]["budget"] > 0


def test_the_complete_payload_carries_everything_the_ui_renders(tmp_path: Path) -> None:
    result = final(collect(base_body(tmp_path)))
    for key in (
        "stages", "scorecards", "initial_draft", "best_draft", "final_draft",
        "score_trajectory", "iterations_detail", "harness", "guardrails", "memory",
    ):
        assert key in result, key


def test_the_bug_report_rubric_runs_through_the_same_code(tmp_path: Path) -> None:
    events = collect(base_body(
        tmp_path,
        rubric_id="bug_report",
        text=(
            "The export is broken again and it does not work at all. I tried it "
            "this morning and just got an error page instead of a file. This has "
            "happened before and nobody ever fixes it properly. Please fix it "
            "quickly, a lot of people are complaining about it."
        ),
    ))
    assert final(events)["rubric_id"] == "bug_report"


def test_an_empty_submission_is_refused_with_a_readable_message(tmp_path: Path) -> None:
    events = collect(base_body(tmp_path, text="   "))
    assert names(events) == ["error"]
    assert "text" in events[0][1]["message"].lower()


def test_an_unknown_rubric_is_refused(tmp_path: Path) -> None:
    events = collect(base_body(tmp_path, rubric_id="does_not_exist"))
    assert names(events) == ["error"]


# ===========================================================================
# The text transformation, which is the point of the application
# ===========================================================================


def test_every_stage_of_the_text_is_captured(tmp_path: Path) -> None:
    result = final(collect(base_body(tmp_path)))
    stages = result["stages"]

    assert len(stages) >= 3, "original plus at least two revisions"
    assert stages[0]["label"] == "Original"
    assert stages[0]["text"] == DRAFT
    # Each stage is a real change, in order, and carries what produced it.
    for earlier, later in zip(stages, stages[1:], strict=False):
        assert earlier["text"] != later["text"]
        assert later["stage"] == earlier["stage"] + 1
    assert any(s["action"] == "revise_text" for s in stages[1:])
    assert any(s["focus"] for s in stages[1:]), "a revision names the criteria it targeted"
    # The last stage is what the run actually hands back.
    assert stages[-1]["text"] == result["best_draft"]


def test_the_recorder_only_records_actual_changes(tmp_path: Path) -> None:
    """Scoring and analysis edit nothing; they must not appear as stages."""
    rubric = server.load_rubrics()["essay_argumentative"]
    recorder = server.DraftRecorder(build_registry(rubric))
    recorder.start(DRAFT)
    ctx = ToolContext(
        config=load_config(CONFIG),
        rubric=rubric,
        workspace=Workspace(draft=DRAFT),
        llm=MockProvider(responder=lambda call: ProviderUnavailableError("unused")),
    )
    recorder.dispatch(Decision(action="analyze_text"), ctx)
    assert len(recorder.stages) == 1  # still just the original


def test_stages_and_the_returned_draft_agree_with_the_scores(tmp_path: Path) -> None:
    result = final(collect(base_body(tmp_path)))
    assert result["best_draft"] == result["stages"][-1]["text"]
    assert len(result["initial_draft"]) < len(result["best_draft"])
    assert len(result["score_trajectory"]) == len(result["scorecards"])


# ===========================================================================
# Memory
# ===========================================================================


def test_recall_spy_reports_what_was_recalled_not_just_a_count(tmp_path: Path) -> None:
    class OneHit(NullMemory):
        def recall(self, query: str, **kwargs: t.Any) -> list:
            from agentic_rubric.core.state import MemoryHit

            return [MemoryHit(kind="lesson", content="attribute every figure", score=0.5)]

    spy = server.RecallSpy(OneHit())
    spy.recall("anything")
    assert spy.last and spy.last[0].content == "attribute every figure"

    spy.save(MemoryRecord(kind="lesson", content="something worth keeping", iteration=2))
    assert spy.writes == []  # NullMemory returns no id, so nothing was really stored


def test_memory_writes_are_reported_to_the_ui(tmp_path: Path) -> None:
    body = base_body(tmp_path, memory_enabled=True, session_id="web-stages")
    body["memory_backend"] = "sqlite_fts"
    result = final(collect(body))
    assert result["memory_writes"], "a run with memory on must write something"
    assert any(w["kind"] == "lesson" for w in result["memory_writes"])


def test_memory_off_recalls_nothing_and_writes_nothing(tmp_path: Path) -> None:
    result = final(collect(base_body(tmp_path, memory_enabled=False)))
    assert result["memory_writes"] == []


def test_stored_lessons_is_safe_on_a_store_that_cannot_search() -> None:
    assert server.stored_lessons(NullMemory()) == []


def test_unwrap_store_reaches_through_the_spy_and_the_policy_layer() -> None:
    from agentic_rubric.memory.manager import MemoryManager

    inner = NullMemory()
    wrapped = server.RecallSpy(MemoryManager(inner, load_config(CONFIG).memory))
    assert server.unwrap_store(wrapped) is inner


# ===========================================================================
# Harness, surfaced in the UI
# ===========================================================================


def test_an_injected_rate_limit_is_recovered_from_and_reported(tmp_path: Path) -> None:
    body = base_body(tmp_path, simulate_failure="rate_limit", fail_step="judge")
    result = final(collect(body))
    assert result["status"] == "target_reached"
    assert result["harness"]["retries"] >= 1


def test_an_injected_tool_fault_is_recovered_from(tmp_path: Path) -> None:
    body = base_body(tmp_path, simulate_failure="tool_error")
    result = final(collect(body))
    assert result["status"] == "target_reached"
    assert result["harness"]["tool_recoveries"] >= 1
    # The recorder still saw the revision, so the text timeline stays complete.
    assert any(s["action"] == "revise_text" for s in result["stages"])


def test_an_injected_memory_outage_still_finishes_the_run(tmp_path: Path) -> None:
    body = base_body(
        tmp_path, simulate_failure="memory_down", memory_enabled=True, session_id="web-down"
    )
    body["memory_backend"] = "sqlite_fts"
    result = final(collect(body))
    assert result["status"] == "target_reached"
    assert result["harness"]["degraded_memory"] is True


def test_the_token_budget_stops_the_run_and_keeps_the_best_draft(tmp_path: Path) -> None:
    result = final(collect(base_body(tmp_path, token_budget=900)))
    assert result["status"] == "budget_exhausted"
    assert result["best_draft"]
    assert result["guardrails"]["triggered"] == ["token_budget"]


def test_tree_of_thoughts_branch_reports_candidate_selection(tmp_path: Path) -> None:
    events = collect(base_body(tmp_path, revise_candidates=3))
    revisions = [
        data for name, data in events
        if name == "act" and data.get("action") == "revise_text" and data.get("ok")
    ]
    assert revisions
    assert any("candidates" in (r.get("summary") or "") for r in revisions)


# ===========================================================================
# Run history and traces
# ===========================================================================


def test_recent_runs_reads_the_summaries_on_disk(tmp_path: Path) -> None:
    import json

    config = load_config(CONFIG, overrides={"logging.trace_dir": str(tmp_path)})
    (tmp_path / "run_abc").mkdir()
    (tmp_path / "run_abc" / "summary.json").write_text(
        json.dumps({"run_id": "run_abc", "status": "target_reached", "iterations": 4}),
        encoding="utf-8",
    )
    (tmp_path / "not_a_run").mkdir()  # no summary.json; must be skipped

    rows = server.recent_runs(config)
    assert [r["run_id"] for r in rows] == ["run_abc"]
    assert rows[0]["iterations"] == 4


def test_recent_runs_is_empty_when_nothing_has_run(tmp_path: Path) -> None:
    config = load_config(CONFIG, overrides={"logging.trace_dir": str(tmp_path / "nope")})
    assert server.recent_runs(config) == []


def test_a_run_writes_a_trace_the_ui_can_link_to(tmp_path: Path) -> None:
    body = base_body(tmp_path)
    body["_"] = None
    result = final(collect(body))
    assert result["trace_path"].endswith("trace.jsonl")
    assert Path(result["trace_path"]).is_file()


def test_rubric_ids_used_by_the_preset_manifest_all_exist() -> None:
    """A typo here would silently give a rubric no presets."""
    rubrics = set(server.load_rubrics())
    assert set(server.SAMPLES) <= rubrics


def test_the_picker_offers_the_worked_example_first() -> None:
    """Filename order put bug_report first, so the essay tool opened on bug reports."""
    assert list(server.load_rubrics())[0] == "essay_argumentative"
