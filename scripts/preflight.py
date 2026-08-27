"""Phase 0 preflight check.

Answers the three questions that otherwise waste the first hour of every setup:
does the config parse, which providers are actually usable, and does a live call
succeed? Run it before anything else::

    python scripts/preflight.py                 # offline: config + availability
    python scripts/preflight.py --ping          # also make one real API call
    python scripts/preflight.py --ping --provider ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from agentic_rubric.config import ConfigError, load_config  # noqa: E402
from agentic_rubric.llm import (  # noqa: E402
    LLMError,
    available_chain,
    build_provider,
    system,
    user,
)

OK = "[ ok ]"
BAD = "[fail]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate config and provider reachability.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--ping", action="store_true", help="make one real completion call")
    parser.add_argument("--provider", default=None, help="force a specific provider for --ping")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    try:
        config = load_config(args.config, project_root=ROOT)
    except ConfigError as exc:
        print(f"{BAD} config: {exc}")
        return 2
    print(f"{OK} config parsed: {args.config}")

    print("\n--- runtime settings ---")
    print(f"  max_iterations   : {config.loop.max_iterations}")
    print(f"  target_score     : {config.loop.target_score}")
    print(f"  token_budget     : {config.guardrails.token_budget:,}")
    print(f"  retry attempts   : {config.retry.max_attempts} ({config.retry.jitter} jitter)")
    print(f"  memory backend   : {config.memory.backend} -> {config.path(config.memory.db_path)}")
    print(f"  trace dir        : {config.path(config.logging.trace_dir)}")

    print("\n--- provider failover chain ---")
    rows = available_chain(config)
    for position, (name, ok, reason) in enumerate(rows, start=1):
        settings = config.llm.providers.get(name)
        model = settings.model if settings else "?"
        marker = OK if ok else BAD
        print(f"  {position}. {marker} {name:<8} {model:<24} {reason}")

    usable = [name for name, ok, _ in rows if ok and name != "mock"]
    if not usable:
        print(
            "\n  No live provider is configured. Copy .env.example to .env and set at least\n"
            "  one key, or run the loop with --provider mock for the offline demo."
        )

    if not args.ping:
        return 0

    target = args.provider or (usable[0] if usable else None)
    if target is None:
        print(f"\n{BAD} --ping needs a usable provider")
        return 1

    print(f"\n--- live ping: {target} ---")
    try:
        provider = build_provider(config, target)
    except (ConfigError, LLMError) as exc:
        print(f"{BAD} could not build provider: {exc}")
        return 1

    try:
        response = provider.complete(
            [
                system("You are a terse assistant."),
                user("Reply with the single word: ready"),
            ],
            temperature=0.0,
            # Not 16: reasoning models spend the output budget on an internal
            # reasoning field before any content, so a tight cap makes a
            # perfectly healthy provider look like it replied with nothing.
            max_tokens=128,
        )
    except LLMError as exc:
        print(f"{BAD} {type(exc).__name__}: {exc}")
        return 1
    finally:
        provider.close()

    print(f"{OK} {provider.describe()} replied {response.text.strip()!r}")
    print(
        f"  tokens in/out    : {response.usage.input_tokens}/{response.usage.output_tokens}"
        f"{' (estimated)' if response.usage.estimated else ''}"
    )
    print(f"  latency          : {response.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
