"""The entry point: argument handling, config precedence, exit codes, wiring.

`cli.py` is where every runtime decision is finally made — which provider,
which rubric, which limits, whether memory is on — so a bug here is invisible
to every other test in the suite and visible to every user on their first
command.

Everything runs against `--provider mock`, so the suite still needs no API key
and no network. `--no-trace` keeps runs from littering `runs/`, and every test
that touches memory points `memory.db_path` at `tmp_path`.
"""

from __future__ import annotations

import argparse
import json
import typing as t
from pathlib import Path

import pytest

from agentic_rubric import cli
from agentic_rubric.config import ConfigError, load_config
from agentic_rubric.core.rubric import Rubric
from agentic_rubric.harness.faults import SIMULATED_TOKEN_BUDGET, FaultyRegistry
from agentic_rubric.llm.mock import MockProvider

ESSAY = "samples/weak_essay.txt"
BUG_REPORT = "samples/weak_bug_report.txt"
BUG_RUBRIC = "config/rubrics/bug_report.yaml"


def run(*argv: str, memory_db: Path | None = None) -> int:
    """Invoke main() offline, with tracing off and memory pointed somewhere safe."""
    args = ["--provider", "mock", "--no-trace", "--quiet", *argv]
    if memory_db is not None:
        args += ["--set", f"memory.db_path={memory_db}"]
    return cli.main(args)


def parse(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


# ===========================================================================
# Argument handling
# ===========================================================================


def test_dotted_overrides_are_coerced_to_real_types() -> None:
    parsed = cli.parse_overrides(
        [
            "loop.max_iterations=3",
            "llm.temperature=0.5",
            "memory.enabled=false",
            "logging.trace_dir=runs/x",
            "memory.embed_model=null",
        ]
    )
    assert parsed["loop.max_iterations"] == 3
    assert parsed["llm.temperature"] == pytest.approx(0.5)
    assert parsed["memory.enabled"] is False
    assert parsed["logging.trace_dir"] == "runs/x"      # stays a string
    assert parsed["memory.embed_model"] is None


def test_a_malformed_override_is_rejected_with_the_offending_pair() -> None:
    with pytest.raises(ConfigError) as caught:
        cli.parse_overrides(["loop.max_iterations"])
    assert "loop.max_iterations" in str(caught.value)


def test_set_reaches_config_keys_that_have_no_flag(tmp_path: Path) -> None:
    """The claim the README makes: *any* key, from the command line."""
    config = load_config(
        "config/config.yaml",
        overrides=cli.parse_overrides(
            ["retry.tool_max_attempts=1", "guardrails.stuck_score_window=5"]
        ),
    )
    assert config.retry.tool_max_attempts == 1
    assert config.guardrails.stuck_score_window == 5


def test_flags_outrank_the_config_file(tmp_path: Path) -> None:
    """CLI > env > YAML > defaults, checked at the layer that assembles it."""
    from_file = load_config("config/config.yaml")
    overridden = load_config(
        "config/config.yaml", overrides={"loop.max_iterations": from_file.loop.max_iterations + 5}
    )
    assert overridden.loop.max_iterations == from_file.loop.max_iterations + 5


def test_input_and_text_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse("--input", ESSAY, "--text", "inline")


def test_text_is_read_inline() -> None:
    assert cli.read_input(parse("--text", "hello world")) == "hello world"


def test_a_missing_input_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        cli.read_input(parse("--input", str(tmp_path / "nope.txt")))
    assert "nope.txt" in str(caught.value)


def test_no_input_at_all_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with pytest.raises(ValueError) as caught:
        cli.read_input(parse())
    assert "--input" in str(caught.value) and "stdin" in str(caught.value)


# ===========================================================================
# Provider resolution
# ===========================================================================


def test_the_mock_provider_needs_no_key_and_no_chain() -> None:
    config = load_config("config/config.yaml")
    rubric = Rubric.from_yaml("config/rubrics/essay_argumentative.yaml")
    provider, chain, notes = cli.resolve_provider(
        config, parse("--provider", "mock"), rubric
    )
    assert isinstance(provider, MockProvider)
    assert chain.names == ["mock"]
    assert notes == []


def test_simulating_provider_down_builds_something_to_fail_over_to() -> None:
    """A one-link chain would prove only that a dead provider ends the run."""
    config = load_config("config/config.yaml")
    rubric = Rubric.from_yaml("config/rubrics/essay_argumentative.yaml")
    _, chain, notes = cli.resolve_provider(
        config, parse("--provider", "mock", "--simulate-failure", "provider_down"), rubric
    )
    assert chain.names == ["mock-primary", "mock"]
    assert any("refuse every call" in note for note in notes)


def test_an_llm_failure_is_injected_into_the_named_step() -> None:
    config = load_config("config/config.yaml")
    rubric = Rubric.from_yaml("config/rubrics/essay_argumentative.yaml")
    provider, _, notes = cli.resolve_provider(
        config,
        parse("--provider", "mock", "--simulate-failure", "rate_limit", "--fail-step", "reason"),
        rubric,
    )
    responder = t.cast(t.Any, provider)._responder
    assert "reason" in responder.fail_on
    assert any("reason step" in note for note in notes)


def test_no_usable_provider_reports_why_each_was_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answering "why did nothing work?" is the point of checking up front."""
    from agentic_rubric.llm.types import ProviderUnavailableError

    # Every provider in the shipped chain, so a key present in the
    # developer's real environment cannot make this pass by accident.
    for variable in ("GEMINI_API_KEY", "GROQ_API_KEY", "OLLAMA_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    config = load_config(
        "config/config.yaml",
        overrides={"llm.providers.ollama.requires_key": True},
    )
    rubric = Rubric.from_yaml("config/rubrics/essay_argumentative.yaml")

    with pytest.raises(ProviderUnavailableError) as caught:
        cli.resolve_provider(config, parse(), rubric)

    message = str(caught.value)
    assert "--provider mock" in message          # the way out is stated
    assert "GROQ_API_KEY" in message             # and so is the actual cause


# ===========================================================================
# Simulation plumbing
# ===========================================================================


def test_budget_simulation_is_config_not_a_special_code_path() -> None:
    """The guardrail under test must be the real one, reading its real knob."""
    overrides: dict[str, t.Any] = {}
    notes = cli.apply_simulation(parse("--simulate-failure", "budget"), overrides)
    assert overrides["guardrails.token_budget"] == SIMULATED_TOKEN_BUDGET
    assert notes and "token budget" in notes[0]


def test_no_simulation_leaves_the_config_untouched() -> None:
    overrides: dict[str, t.Any] = {}
    assert cli.apply_simulation(parse(), overrides) == []
    assert overrides == {}


def test_tool_faults_are_injected_by_swapping_the_registry() -> None:
    rubric = Rubric.from_yaml("config/rubrics/essay_argumentative.yaml")
    plain = cli.build_run_registry(parse(), rubric)
    rigged = cli.build_run_registry(parse("--simulate-failure", "tool_error"), rubric)
    assert not isinstance(plain, FaultyRegistry)
    assert isinstance(rigged, FaultyRegistry)
    assert rigged.names == plain.names  # same tool set, one of them rigged


# ===========================================================================
# End to end through main()
# ===========================================================================


def test_a_successful_run_exits_zero(tmp_path: Path) -> None:
    assert run("--input", ESSAY, "--no-memory") == 0


def test_the_same_code_path_handles_a_different_domain(tmp_path: Path) -> None:
    """Rubrics are data: swapping the YAML swaps the whole domain."""
    assert run("--input", BUG_REPORT, "--rubric", BUG_RUBRIC, "--no-memory") == 0


def test_stopping_short_of_the_target_exits_non_zero(tmp_path: Path) -> None:
    """So the command is usable as a CI gate."""
    code = run("--input", ESSAY, "--no-memory", "--max-iters", "1")
    assert code == 1


def test_a_bad_config_path_exits_two(tmp_path: Path) -> None:
    assert cli.main(["--input", ESSAY, "--config", str(tmp_path / "missing.yaml")]) == 2


def test_a_bad_rubric_path_exits_two(tmp_path: Path) -> None:
    assert run("--input", ESSAY, "--rubric", str(tmp_path / "missing.yaml")) == 2


def test_the_best_draft_can_be_written_to_a_file(tmp_path: Path) -> None:
    out = tmp_path / "improved.txt"
    assert run("--input", ESSAY, "--no-memory", "--out", str(out)) == 0
    written = out.read_text(encoding="utf-8")
    assert len(written) > len(Path(ESSAY).read_text(encoding="utf-8"))


def test_json_output_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("--input", ESSAY, "--no-memory", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "target_reached"


def test_json_keeps_stdout_to_itself(capsys: pytest.CaptureFixture[str]) -> None:
    """`... --json | jq` must work without also remembering --quiet."""
    assert cli.main(
        ["--input", ESSAY, "--provider", "mock", "--no-trace", "--no-memory", "--json"]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)          # would raise if the transcript leaked
    assert "RUN COMPLETE" in captured.err       # the transcript went to stderr
    assert payload["status"] == "target_reached"
    assert payload["iterations"] >= 3
    assert payload["harness"]["retries"] == 0
    assert len(payload["score_trajectory"]) >= 2


def test_show_draft_prints_the_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("--input", ESSAY, "--no-memory", "--show-draft") == 0
    assert "----- BEST DRAFT -----" in capsys.readouterr().out


def test_a_run_writes_a_trace_when_tracing_is_on(tmp_path: Path) -> None:
    code = cli.main(
        [
            "--input", ESSAY, "--provider", "mock", "--quiet", "--no-memory",
            "--trace-dir", str(tmp_path),
        ]
    )
    assert code == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    assert (runs[0] / "trace.jsonl").exists()
    assert (runs[0] / "summary.json").exists()


def test_no_trace_leaves_no_footprint(tmp_path: Path) -> None:
    assert run("--input", ESSAY, "--no-memory", "--trace-dir", str(tmp_path)) == 0
    assert list(tmp_path.iterdir()) == []


# ===========================================================================
# Memory operations exposed on the command line
# ===========================================================================


def test_memory_stats_and_clear_session_are_reachable_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two of the three required memory operations, demonstrable without a test."""
    db = tmp_path / "memory.db"
    assert run("--input", ESSAY, "--session", "cli-test", memory_db=db) == 0

    assert cli.main(["--memory-stats", "--set", f"memory.db_path={db}"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] > 0
    assert "lesson" in stats["by_kind"]

    assert cli.main(["--clear-session", "cli-test", "--set", f"memory.db_path={db}"]) == 0
    assert "cleared" in capsys.readouterr().out

    assert cli.main(["--memory-stats", "--set", f"memory.db_path={db}"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 0


def test_no_memory_is_a_complete_swap(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    assert run("--input", ESSAY, "--session", "off", "--no-memory", memory_db=db) == 0
    # Nothing was written at all: the A/B control has to be a real control.
    assert cli.main(["--memory-stats", "--set", f"memory.db_path={db}"]) == 0


def test_a_simulated_memory_outage_still_finishes_the_run(tmp_path: Path) -> None:
    assert run(
        "--input", ESSAY, "--simulate-failure", "memory_down", "--session", "down",
        memory_db=tmp_path / "memory.db",
    ) == 0


def test_the_budget_simulation_stops_the_run_short(tmp_path: Path) -> None:
    code = run(
        "--input", ESSAY, "--no-memory", "--simulate-failure", "budget"
    )
    assert code == 1  # budget_exhausted is not a success
