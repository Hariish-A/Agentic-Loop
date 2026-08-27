"""The console transcript.

Worth testing for a reason that is easy to dismiss: this is the *only* view of
the run a reviewer watching the demo actually sees. A renderer that silently
drops the harness events would make a run with fifteen absorbed rate limits
look identical to a run with none — the recovery would be real and invisible,
which for a demonstration is the same as not having it.

Pure formatting over a dict, so every test here is a string assertion against
a recorded event stream.
"""

from __future__ import annotations

import io
import typing as t

from agentic_rubric.observability.render import ConsoleRenderer, score_bar


def render(*events: tuple[str, dict[str, t.Any]], verbose: bool = False) -> str:
    stream = io.StringIO()
    renderer = ConsoleRenderer(stream, verbose=verbose)
    for name, payload in events:
        renderer(name, payload)
    return stream.getvalue()


RUN_START = (
    "run_start",
    {
        "run_id": "run_abc123",
        "session_id": "sess_xyz",
        "rubric_id": "essay_argumentative",
        "target_score": 85.0,
        "max_iterations": 6,
        "provider": "groq:openai/gpt-oss-120b",
        "memory": "SQLiteMemory[vector+keyword]",
    },
)


# ===========================================================================
# The score bar
# ===========================================================================


def test_the_bar_fills_with_the_score_and_marks_the_target() -> None:
    bar = score_bar(50.0, 85.0, width=20)
    assert bar.startswith("[") and bar.endswith("]")
    assert bar.count("#") == 10          # half of twenty
    assert "|" in bar                    # the target marker


def test_an_unscored_draft_reads_as_unknown_not_as_zero() -> None:
    """Reporting "0%" for "not measured yet" would be a lie the eye believes."""
    assert set(score_bar(None, 85.0, width=8)) == {"[", "?", "]"}


def test_the_bar_clamps_rather_than_overflowing() -> None:
    over = score_bar(140.0, 85.0, width=10)
    under = score_bar(-20.0, 85.0, width=10)
    assert len(over) == len(under) == 12          # ten cells plus the brackets
    assert "-" not in over                        # full, and the marker still drawn
    assert over.count("|") == 1
    assert under.count("#") == 0


# ===========================================================================
# The four steps
# ===========================================================================


def test_the_header_names_the_provider_and_the_limits() -> None:
    out = render(RUN_START)
    assert "RUN START" in out
    assert "run_abc123" in out and "sess_xyz" in out
    assert "groq:openai/gpt-oss-120b" in out
    assert "max iterations: 6" in out


def test_each_step_is_labelled_and_the_thought_is_shown() -> None:
    out = render(
        RUN_START,
        ("iteration_start", {"iteration": 2}),
        ("perceive", {"score": 66.2, "words": 243, "flesch": 47.3,
                      "failing_probes": ["has_figures"], "recalled": 3, "notes": []}),
        ("reason", {"action": "revise_text", "thought": "Evidence carries the most headroom."}),
        ("act", {"action": "revise_text", "ok": True, "summary": "243 -> 284 words",
                 "duration_ms": 12.0}),
        ("reflect", {"task_complete": False, "score_delta": 37.5, "next_focus": "reasoning"}),
    )
    assert "ITERATION 2" in out
    assert "PERCEIVE" in out and "66.2%" in out
    assert "REASON" in out and "revise_text" in out
    assert "Evidence carries the most headroom." in out
    assert "ACT" in out and "[ok]" in out
    assert "REFLECT" in out and "delta=+37.5pts" in out
    assert "next focus: reasoning" in out


def test_a_failed_tool_call_shows_the_error_not_the_summary() -> None:
    out = render(
        ("act", {"action": "revise_text", "ok": False, "summary": "",
                 "error": "the reviser returned an empty draft", "duration_ms": 3.0}),
    )
    assert "[FAILED]" in out
    assert "empty draft" in out


def test_a_degraded_decision_is_flagged_as_one() -> None:
    """A fallback action must never look like a choice the agent made."""
    out = render(("reason", {"action": "analyze_text", "thought": "[fallback] ...",
                             "degraded": True}))
    assert "DEGRADED FALLBACK" in out


def test_a_rule_based_reflection_is_flagged_as_one() -> None:
    out = render(("reflect", {"task_complete": False, "degraded": True}))
    assert "RULE-BASED" in out


def test_a_plateau_is_flagged() -> None:
    out = render(("reflect", {"task_complete": False, "plateau": True, "score_delta": 0.2}))
    assert "PLATEAU" in out


def test_a_lesson_is_always_shown_even_without_verbose() -> None:
    """The lesson is the Reflexion payload; it is the point of watching."""
    out = render(("reflect", {"task_complete": False,
                              "critique": "a long critique nobody asked for",
                              "lesson": "Attribute every figure."}))
    assert "LESSON: Attribute every figure." in out
    assert "nobody asked for" not in out          # critique is verbose-only


def test_verbose_adds_the_critique_and_the_perceive_detail() -> None:
    out = render(
        ("perceive", {"score": 50.0, "words": 200, "flesch": 40.0,
                      "failing_probes": [], "recalled": 2, "notes": []}),
        ("reflect", {"task_complete": False, "critique": "the revision helped"}),
        verbose=True,
    )
    assert "words=200" in out and "recalled=2" in out
    assert "the revision helped" in out


def test_perceive_notes_surface_without_verbose() -> None:
    """A degradation notice must not depend on the operator having asked."""
    out = render(("perceive", {"score": None, "notes": ["running without memory: breaker open"]}))
    assert "running without memory" in out


# ===========================================================================
# Harness events
# ===========================================================================


def test_a_retry_names_the_provider_the_delay_and_its_source() -> None:
    out = render(("retry", {"provider": "groq", "attempt": 1, "delay_s": 12.0,
                            "error_type": "RateLimitError", "honoured_retry_after": True}))
    assert "HARNESS" in out
    assert "retry 1 on groq after 12.0s (Retry-After): RateLimitError" in out


def test_computed_backoff_is_distinguished_from_an_honoured_hint() -> None:
    out = render(("retry", {"provider": "groq", "attempt": 2, "delay_s": 1.7,
                            "error_type": "TransientServerError",
                            "honoured_retry_after": False}))
    assert "(backoff)" in out


def test_local_salvage_and_a_repair_call_read_differently() -> None:
    """They cost very different amounts; the transcript should say which happened."""
    salvaged = render(("repair", {"provider": "groq", "method": "local_salvage"}))
    repaired = render(("repair", {"provider": "groq", "method": "repair_call",
                                  "error": "prose, no tool call"}))
    assert "salvaged the tool call locally" in salvaged
    assert "sent one repair prompt" in repaired


def test_a_failover_shows_both_ends_and_the_reason() -> None:
    out = render(("failover", {"from": "groq", "to": "ollama", "reason": "401 invalid key"}))
    assert "failover groq -> ollama" in out
    assert "401 invalid key" in out


def test_an_exhausted_chain_says_so_rather_than_printing_none() -> None:
    out = render(("failover", {"from": "ollama", "to": None, "reason": "connection refused"}))
    assert "nothing left in the chain" in out


def test_tool_recovery_distinguishes_a_retry_from_a_hand_back() -> None:
    retried = render(("tool_recovery", {"action": "revise_text", "method": "backoff_retry",
                                        "retry_as": "revise_text"}))
    handed = render(("tool_recovery", {"action": "revise_text",
                                       "method": "fed_back_as_observation", "retry_as": None}))
    assert "-> revise_text" in retried
    assert "handed back to the agent as an observation" in handed


def test_a_guardrail_trip_and_the_stop_are_both_visible() -> None:
    out = render(
        ("budget_warning", {"reason": "token budget 85% spent"}),
        ("guardrail_trip", {"guardrail": "token_budget", "detail": "1,080 of 900 spent"}),
        ("guardrail", {"status": "budget_exhausted", "reason": "finalising on the best draft"}),
    )
    assert "token budget 85% spent" in out
    assert "[token_budget] 1,080 of 900 spent" in out
    assert "STOP (budget_exhausted)" in out
    assert "finalising on the best draft" in out


# ===========================================================================
# The summary
# ===========================================================================


RUN_END = (
    "run_end",
    {
        "status": "target_reached",
        "iterations": 5,
        "score_trajectory": [15.0, 32.5, 87.5],
        "initial_score": 15.0,
        "best_score": 87.5,
        "tokens": {"input": 21552, "output": 8540},
        "elapsed_s": 169.67,
        "notes": ["a note worth reading"],
    },
)

RUN_SUMMARY = (
    "run_summary",
    {
        "harness": {
            "provider": "groq:openai/gpt-oss-120b",
            "retries": 15, "repairs": 0, "failovers": 0, "tool_recoveries": 0,
            "degraded_memory": False, "cost_est_usd": 0.0,
            "trace": "runs/run_abc123/trace.jsonl",
        },
        "guardrails": {
            "tokens": {"used": 30092, "budget": 200000, "fraction": 0.15},
            "triggered": [],
        },
        "notes": [],
    },
)


def test_the_summary_reports_the_trajectory_and_the_improvement() -> None:
    out = render(RUN_START, RUN_END)
    assert "RUN COMPLETE" in out
    assert "15.0% -> 32.5% -> 87.5%" in out
    assert "(+72.5 points)" in out
    assert "21,552 in / 8,540 out" in out
    assert "169.67s" in out
    assert "a note worth reading" in out


def test_a_never_scored_run_says_so_instead_of_printing_an_arrow() -> None:
    out = render(RUN_START, ("run_end", {"status": "error", "iterations": 0,
                                         "score_trajectory": [], "tokens": {}}))
    assert "never scored" in out


def test_the_harness_line_reports_what_recovery_cost() -> None:
    out = render(RUN_SUMMARY)
    assert "retries=15" in out and "repairs=0" in out and "failovers=0" in out
    assert "30,092 / 200,000 tokens (15%)" in out
    assert "runs/run_abc123/trace.jsonl" in out


def test_degraded_memory_is_called_out_in_the_summary() -> None:
    payload = dict(RUN_SUMMARY[1])
    payload["harness"] = {**payload["harness"], "degraded_memory": True}
    assert "memory DEGRADED" in render(("run_summary", payload))


def test_cost_is_only_printed_once_it_is_non_zero() -> None:
    """A "$0.0000" line implies a measurement; the default is an absence of one."""
    free = render(RUN_SUMMARY)
    assert "cost (est)" not in free

    payload = dict(RUN_SUMMARY[1])
    payload["harness"] = {**payload["harness"], "cost_est_usd": 0.0123}
    assert "$0.0123" in render(("run_summary", payload))


def test_triggered_guardrails_are_listed() -> None:
    payload = dict(RUN_SUMMARY[1])
    payload["guardrails"] = {**payload["guardrails"], "triggered": ["token_budget"]}
    assert "guardrails  : token_budget" in render(("run_summary", payload))


def test_an_unknown_event_is_ignored_rather_than_crashing() -> None:
    """New events must not be able to break a transcript mid-run."""
    assert render(("some_future_event", {"anything": 1})) == ""
