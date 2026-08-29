"""Command-line entry point.

Everything the run needs is assembled here and nowhere else: config is loaded,
overrides are layered, the rubric is read, the provider chain is built, and a
:class:`~.harness.runner.Runner` wraps the loop in retry, fallbacks, guardrails
and tracing. ``core/`` receives finished objects and never reaches back out for
any of it.

Examples::

    # offline, no API key, full run
    python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock

    # live, against whichever provider is configured first
    python -m agentic_rubric.cli --input samples/weak_essay.txt

    # different domain, same code path
    python -m agentic_rubric.cli --input samples/weak_bug_report.txt \\
        --rubric config/rubrics/bug_report.yaml

    # any config key, from the command line
    python -m agentic_rubric.cli --input samples/weak_essay.txt \\
        --set loop.revise_candidates=3 --set llm.temperature=0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import typing as t
from pathlib import Path

from .config import AppConfig, ConfigError, load_config
from .core.rubric import Rubric, RubricError
from .core.state import RunStatus
from .envfile import load_env_file
from .harness.fallbacks import ProviderChain
from .harness.faults import (
    FAILURE_KINDS,
    LLM_KINDS,
    SIMULATED_TOKEN_BUDGET,
    FaultyMemory,
    FaultyRegistry,
    llm_failure,
)
from .harness.runner import Runner
from .llm.base import LLMProvider
from .llm.demo_responder import ScriptedAgentResponder
from .llm.factory import available_chain, build_provider
from .llm.mock import MockProvider
from .llm.types import LLMError, ProviderUnavailableError
from .memory.base import MemoryStore
from .memory.factory import build_memory as build_memory_stack
from .observability.logger import configure_logging
from .observability.render import ConsoleRenderer
from .tools.definitions import build_registry
from .tools.registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_RUBRIC = PROJECT_ROOT / "config" / "rubrics" / "essay_argumentative.yaml"

FAILURE_STEPS = ("reason", "judge", "revise", "reflect")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rubric-forge",
        description="Score text against a rubric and iteratively improve it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", "-i", help="path to the text file to improve")
    source.add_argument("--text", help="text to improve, given inline")

    parser.add_argument("--rubric", "-r", default=str(DEFAULT_RUBRIC), help="rubric YAML file")
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG), help="config YAML file")
    parser.add_argument("--provider", "-p", help="override the primary LLM provider")
    parser.add_argument("--target", type=float, help="target score, 0-100")
    parser.add_argument("--max-iters", type=int, dest="max_iters", help="hard iteration cap")
    parser.add_argument("--session", help="session id, for memory continuity across runs")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="override any config key, e.g. --set loop.revise_candidates=3 (repeatable)",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="run without memory (A/B baseline)"
    )
    parser.add_argument(
        "--clear-session",
        metavar="SESSION_ID",
        help="delete every memory record for one session, then exit",
    )
    parser.add_argument(
        "--memory-stats", action="store_true", help="print memory statistics and exit"
    )
    parser.add_argument("--out", "-o", help="write the best draft to this file")
    parser.add_argument("--json", action="store_true", help="print the run result as JSON")
    parser.add_argument("--show-draft", action="store_true", help="print the final text")
    parser.add_argument("--verbose", "-v", action="store_true", help="show per-step detail")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress the transcript")
    parser.add_argument(
        "--simulate-failure",
        choices=FAILURE_KINDS,
        help="inject a failure to demonstrate recovery (requires --provider mock)",
    )
    parser.add_argument(
        "--fail-step",
        choices=FAILURE_STEPS,
        default="judge",
        help="which loop step an LLM-layer --simulate-failure should hit (default: judge)",
    )
    parser.add_argument(
        "--no-trace", action="store_true", help="do not write runs/<run_id>/"
    )
    parser.add_argument(
        "--trace-dir", help="where run traces are written (default: logging.trace_dir)"
    )
    return parser


def parse_overrides(pairs: t.Sequence[str]) -> dict[str, t.Any]:
    """Turn ``--set a.b=1`` into ``{"a.b": 1}``, with light literal coercion."""
    out: dict[str, t.Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = _literal(raw.strip())
    return out


def _literal(raw: str) -> t.Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def read_input(args: argparse.Namespace) -> str:
    """Read the draft from a file, an inline string, or stdin."""
    if args.text:
        return args.text
    if args.input:
        path = Path(args.input)
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")
        return path.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            return piped
    raise ValueError("no input: pass --input FILE, --text STRING, or pipe text on stdin")


def build_mock_chain(
    config: AppConfig, args: argparse.Namespace, rubric: Rubric
) -> tuple[ProviderChain, list[str]]:
    """The offline provider, plus a dead link in front of it when asked.

    ``--simulate-failure provider_down`` needs something to fail *over from*, so
    it prepends a provider that refuses every call. With only one link the demo
    would prove that an unavailable provider ends the run, which is the opposite
    of the point.
    """
    notes: list[str] = []
    responder = ScriptedAgentResponder(
        rubric=rubric,
        target_score=args.target if args.target is not None else config.loop.target_score,
    )
    healthy = MockProvider(responder=responder, name="mock")
    kind = args.simulate_failure

    if kind == "provider_down":
        dead = MockProvider(
            responder=lambda _call: llm_failure("provider_down"), name="mock-primary"
        )
        notes.append("primary provider 'mock-primary' will refuse every call")
        return (
            ProviderChain(links=[("mock-primary", lambda: dead), ("mock", lambda: healthy)]),
            notes,
        )

    if kind in LLM_KINDS:
        responder.fail_on[args.fail_step] = llm_failure(kind)
        notes.append(f"injected a {kind} failure into the {args.fail_step} step")

    return ProviderChain.of(healthy), notes


def resolve_provider(
    config: AppConfig, args: argparse.Namespace, rubric: Rubric
) -> tuple[LLMProvider, ProviderChain, list[str]]:
    """Build the provider chain the harness will walk.

    Availability is checked here rather than at the first API call, so an
    unusable provider costs nothing and is reported before the run starts
    instead of surfacing three retries deep as a 401. Unlike Milestone 2, the
    remaining links are *kept*: failover now happens inside the run, so a
    provider that dies mid-run has somewhere to go.
    """
    notes: list[str] = []
    requested = args.provider or config.llm.primary

    if requested == "mock":
        chain, mock_notes = build_mock_chain(config, args, rubric)
        return chain.get(0), chain, [*notes, *mock_notes]

    ordered = [requested, *[n for n in config.llm.chain if n != requested]]
    links: list[tuple[str, t.Callable[[], LLMProvider]]] = []
    primary: LLMProvider | None = None

    for name in ordered:
        try:
            provider = build_provider(config, name)
        except (ProviderUnavailableError, ConfigError) as exc:
            notes.append(f"provider {name!r} skipped: {exc}")
            continue
        if primary is None:
            primary = provider
            links.append((name, lambda p=provider: p))  # type: ignore[misc]
        else:
            # Backups stay lazy: a chain that eagerly opened every client would
            # hold sockets for providers it will most likely never call.
            provider.close()
            links.append((name, lambda n=name: build_provider(config, n)))  # type: ignore[misc]

    if primary is None:
        rows = "\n".join(
            f"    {n}: {reason}" for n, ok, reason in available_chain(config) if not ok
        )
        raise ProviderUnavailableError(
            "no usable LLM provider. Set a key in .env, or run with --provider mock.\n" + rows
        )

    if len(links) > 1:
        notes.append("failover chain: " + " -> ".join(name for name, _ in links))
    return primary, ProviderChain(links=links), notes


def build_memory(config: AppConfig, args: argparse.Namespace) -> tuple[MemoryStore, list[str]]:
    """Assemble the memory stack. ``--no-memory`` is the A/B baseline."""
    return build_memory_stack(config, enabled=not args.no_memory)


def wrap_faulty_memory(config: AppConfig, store: MemoryStore) -> MemoryStore:
    """Re-wrap the memory stack around a store whose reads always fail.

    The fault goes *under* the MemoryManager, not in place of it, so the run
    exercises the real circuit breaker rather than a stand-in for it.
    """
    from .memory.manager import MemoryManager

    inner = getattr(store, "store", store)
    return MemoryManager(FaultyMemory(inner), config.memory)


def run_memory_command(config: AppConfig, args: argparse.Namespace) -> int:
    """Handle the maintenance flags that exit before the loop runs.

    These expose two of the three required memory operations directly, so
    ``clear_session`` is demonstrable from the command line rather than only
    from a test.
    """
    store, notes = build_memory_stack(config, enabled=True, warm_embedder=False)
    try:
        for note in notes:
            print(f"note: {note}", file=sys.stderr)
        if args.clear_session:
            removed = store.clear_session(args.clear_session)
            print(f"cleared {removed} record(s) for session {args.clear_session}")
            return 0
        print(json.dumps(store.stats(), indent=2, default=str))
        return 0
    finally:
        store.close()


def apply_simulation(
    args: argparse.Namespace, overrides: dict[str, t.Any]
) -> list[str]:
    """Fold a ``--simulate-failure`` choice into the config overrides.

    Two of the seven kinds are configuration, not code: ``budget`` is a tiny
    token budget, and both are applied here so the guardrail under test is the
    real one reading its real knob, not a special path that only exists in demo
    mode.
    """
    notes: list[str] = []
    if args.simulate_failure == "budget":
        overrides["guardrails.token_budget"] = SIMULATED_TOKEN_BUDGET
        notes.append(
            f"token budget forced down to {SIMULATED_TOKEN_BUDGET} tokens for this run"
        )
    return notes


def build_run_registry(args: argparse.Namespace, rubric: Rubric) -> ToolRegistry:
    """The tool set, rigged to fail once when the demo asks for it."""
    registry = build_registry(rubric)
    if args.simulate_failure == "tool_error":
        return FaultyRegistry(registry)
    return registry


def render_result(result: t.Any, args: argparse.Namespace) -> None:
    """Optional outputs: file, JSON, and the text itself."""
    if args.out:
        Path(args.out).write_text(result.best_draft, encoding="utf-8")
        best = f"{result.best_score:.1f}%" if result.best_score is not None else "unscored"
        print(f"wrote best draft ({best}) to {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    if args.show_draft:
        print("\n----- BEST DRAFT -----")
        print(result.best_draft)
        print("----- END -----")


def main(argv: t.Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_report = load_env_file(PROJECT_ROOT / ".env")

    overrides = parse_overrides(args.overrides)
    if args.provider:
        overrides["llm.primary"] = args.provider
    if args.max_iters is not None:
        overrides["loop.max_iterations"] = args.max_iters
    if args.target is not None:
        overrides["loop.target_score"] = args.target
    if args.no_memory:
        overrides["memory.enabled"] = False
    if args.no_trace:
        overrides["logging.trace_enabled"] = False
    if args.trace_dir:
        overrides["logging.trace_dir"] = args.trace_dir
    simulation_notes = apply_simulation(args, overrides)

    try:
        config = load_config(args.config, overrides=overrides, project_root=PROJECT_ROOT)
        if args.clear_session or args.memory_stats:
            return run_memory_command(config, args)
        rubric = Rubric.from_yaml(args.rubric)
        draft = read_input(args)
        provider, chain, notes = resolve_provider(config, args, rubric)
    except (ConfigError, RubricError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    configure_logging(config.logging)
    notes.extend(env_report.notes)
    notes.extend(simulation_notes)

    memory, memory_notes = build_memory(config, args)
    notes.extend(memory_notes)
    if args.simulate_failure == "memory_down":
        memory = wrap_faulty_memory(config, memory)
        notes.append("memory reads will fail; the circuit breaker should open")

    for note in notes:
        print(f"note: {note}", file=sys.stderr)

    runner = Runner(
        config=config,
        rubric=rubric,
        provider=provider,
        chain=chain,
        memory=memory,
        registry=build_run_registry(args, rubric),
        console=None if args.quiet else ConsoleRenderer(
            # With --json, stdout belongs to the JSON document alone. Sending
            # the transcript there too would make `... --json | jq` fail unless
            # the caller also remembered --quiet, which is a trap rather than
            # an interface.
            stream=sys.stderr if args.json else sys.stdout,
            verbose=args.verbose,
        ),
    )

    try:
        report = runner.run(
            draft,
            session_id=args.session,
            target_score=args.target,
            max_iterations=args.max_iters,
        )
    except LLMError as exc:
        # Only reached when every rung of every ladder was spent -- the whole
        # provider chain is gone, or the request itself is malformed.
        print(f"error: the run could not complete: {exc}", file=sys.stderr)
        return 4
    finally:
        runner.close()

    # A refused submission is a verdict, not a run. Printing a transcript, a
    # trace path and an empty scorecard under it would bury the one line the
    # caller needs.
    if report.result.status is RunStatus.INPUT_REJECTED:
        for note in report.result.notes:
            print(f"refused: {note}", file=sys.stderr)
        if args.json:
            print(json.dumps(report.result.to_dict(), indent=2, default=str))
        return 2

    render_result(report.result, args)
    if report.trace_path and not args.quiet:
        print(f"trace: {report.trace_path}", file=sys.stderr)

    # Exit non-zero when the agent stopped without reaching the target, so the
    # command is usable in a script or a CI gate.
    return 0 if report.result.status.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
