"""Command-line entry point.

Everything the loop needs is assembled here and nowhere else: config is loaded,
overrides are layered, the rubric is read, a provider is built (with failover
down the configured chain), and the result is rendered. ``core/`` receives
finished objects and never reaches back out for any of it.

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

from dotenv import load_dotenv

from .config import AppConfig, ConfigError, load_config
from .core.loop import AgenticLoop
from .core.rubric import Rubric, RubricError
from .llm.base import LLMProvider
from .llm.demo_responder import ScriptedAgentResponder, make_failure
from .llm.factory import available_chain, build_provider
from .llm.mock import MockProvider
from .llm.types import LLMError, ProviderUnavailableError
from .memory.base import MemoryStore
from .memory.factory import build_memory as build_memory_stack
from .observability.render import ConsoleRenderer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_RUBRIC = PROJECT_ROOT / "config" / "rubrics" / "essay_argumentative.yaml"

FAILURE_KINDS = ("rate_limit", "bad_json", "server_error", "provider_down")
FAILURE_STEPS = ("reason", "judge", "revise", "reflect")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-rubric",
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
        help="which loop step --simulate-failure should hit (default: judge)",
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


def resolve_provider(
    config: AppConfig, args: argparse.Namespace, rubric: Rubric
) -> tuple[LLMProvider, list[str]]:
    """Build a provider, walking the failover chain. Returns it plus any notes.

    Trying the chain here rather than at the first API call means an unusable
    provider costs nothing and is reported before the run starts, instead of
    surfacing three retries deep as a 401.
    """
    notes: list[str] = []
    requested = args.provider or config.llm.primary

    if requested == "mock":
        responder = ScriptedAgentResponder(
            rubric=rubric,
            target_score=args.target if args.target is not None else config.loop.target_score,
        )
        if args.simulate_failure:
            responder.fail_on[args.fail_step] = make_failure(args.simulate_failure)
            notes.append(
                f"injected a {args.simulate_failure} failure into the {args.fail_step} step"
            )
        return MockProvider(responder=responder, name="mock"), notes

    chain = [requested, *[n for n in config.llm.chain if n != requested]]
    for index, name in enumerate(chain):
        try:
            provider = build_provider(config, name)
        except (ProviderUnavailableError, ConfigError) as exc:
            notes.append(f"provider {name!r} skipped: {exc}")
            continue
        if index:
            notes.append(f"failed over to provider {name!r}")
        return provider, notes

    rows = "\n".join(f"    {n}: {reason}" for n, ok, reason in available_chain(config) if not ok)
    raise ProviderUnavailableError(
        "no usable LLM provider. Set a key in .env, or run with --provider mock.\n" + rows
    )


def build_memory(config: AppConfig, args: argparse.Namespace) -> tuple[MemoryStore, list[str]]:
    """Assemble the memory stack. ``--no-memory`` is the A/B baseline."""
    return build_memory_stack(config, enabled=not args.no_memory)


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


def main(argv: t.Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")

    overrides = parse_overrides(args.overrides)
    if args.provider:
        overrides["llm.primary"] = args.provider
    if args.max_iters is not None:
        overrides["loop.max_iterations"] = args.max_iters
    if args.target is not None:
        overrides["loop.target_score"] = args.target
    if args.no_memory:
        overrides["memory.enabled"] = False

    try:
        config = load_config(args.config, overrides=overrides, project_root=PROJECT_ROOT)
        if args.clear_session or args.memory_stats:
            return run_memory_command(config, args)
        rubric = Rubric.from_yaml(args.rubric)
        draft = read_input(args)
        provider, notes = resolve_provider(config, args, rubric)
    except (ConfigError, RubricError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    memory, memory_notes = build_memory(config, args)
    notes.extend(memory_notes)

    renderer = None if args.quiet else ConsoleRenderer(verbose=args.verbose)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)

    loop = AgenticLoop(
        config=config,
        provider=provider,
        rubric=rubric,
        memory=memory,
        on_event=renderer,
    )

    try:
        result = loop.run(
            draft,
            session_id=args.session,
            target_score=args.target,
            max_iterations=args.max_iters,
        )
    except LLMError as exc:
        print(f"error: the run could not complete: {exc}", file=sys.stderr)
        return 4
    finally:
        provider.close()
        memory.close()

    if args.out:
        Path(args.out).write_text(result.best_draft, encoding="utf-8")
        print(f"wrote best draft ({result.best_score:.1f}%) to {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    if args.show_draft:
        print("\n----- BEST DRAFT -----")
        print(result.best_draft)
        print("----- END -----")

    # Exit non-zero when the agent stopped without reaching the target, so the
    # command is usable in a script or a CI gate.
    return 0 if result.status.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
