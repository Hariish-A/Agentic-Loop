"""A/B demonstration that memory changes what the agent does.

Runs the same input three times and prints the comparison:

===========  =========================================================
run          condition
===========  =========================================================
**A**        memory enabled, **cold** store -- nothing learned yet
**B**        memory enabled, **warm** store, a brand-new session id
**C**        ``--no-memory`` -- the control
===========  =========================================================

A and C should behave identically, because a cold store and no store carry the
same information. B is the interesting one: a *different session*, reading
lessons that run A wrote, should need fewer iterations to reach the same target.

Run it::

    python scripts/memory_ab_demo.py                    # offline, no API key
    python scripts/memory_ab_demo.py --provider groq    # live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_rubric.config import load_config  # noqa: E402
from agentic_rubric.core.loop import AgenticLoop  # noqa: E402
from agentic_rubric.core.rubric import Rubric  # noqa: E402
from agentic_rubric.core.state import RunResult  # noqa: E402
from agentic_rubric.envfile import load_env_file  # noqa: E402
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder  # noqa: E402
from agentic_rubric.llm.factory import build_provider  # noqa: E402
from agentic_rubric.llm.mock import MockProvider  # noqa: E402
from agentic_rubric.memory.base import NullMemory  # noqa: E402
from agentic_rubric.memory.factory import build_memory  # noqa: E402

DB_PATH = "data/ab_demo_memory.db"


def make_provider(name: str, rubric: Rubric, config, target: float):  # noqa: ANN001
    if name == "mock":
        return MockProvider(
            responder=ScriptedAgentResponder(rubric=rubric, target_score=target), name="mock"
        )
    return build_provider(config, name)


def run(
    label: str,
    *,
    config,  # noqa: ANN001
    rubric: Rubric,
    draft: str,
    provider_name: str,
    session_id: str,
    target: float,
    use_memory: bool,
) -> tuple[RunResult, list[str]]:
    """Execute one arm of the comparison and capture what it recalled."""
    if use_memory:
        memory, _ = build_memory(config, enabled=True)
    else:
        memory = NullMemory()

    provider = make_provider(provider_name, rubric, config, target)
    loop = AgenticLoop(config=config, provider=provider, rubric=rubric, memory=memory)
    try:
        result = loop.run(draft, session_id=session_id, target_score=target)
    finally:
        provider.close()
        memory.close()

    # The same lesson surfaces on several iterations at slightly different
    # relevance scores, so dedupe on content rather than on the rendered line.
    seen: dict[str, str] = {}
    for record in result.records:
        for hit in record.observation.recalled:
            if hit.kind == "lesson":
                seen.setdefault(hit.content, hit.render())
    recalled = list(seen.values())
    print(f"  {label}: {result.status.value} in {result.iterations} iteration(s)")
    return result, recalled


def summarise(name: str, result: RunResult) -> str:
    actions = " -> ".join(r.decision.action.replace("_against_rubric", "") for r in result.records)
    return (
        f"{name:<28} {result.iterations:>2} iters   "
        f"{(result.best_score or 0):>5.1f}%   {result.status.value:<16} {actions}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--input", default=str(ROOT / "samples" / "weak_essay.txt"))
    parser.add_argument(
        "--rubric", default=str(ROOT / "config" / "rubrics" / "essay_argumentative.yaml")
    )
    parser.add_argument("--target", type=float, default=85.0)
    parser.add_argument("--keep-db", action="store_true", help="do not delete the demo database")
    args = parser.parse_args()

    env_report = load_env_file(ROOT / ".env")
    for note in env_report.notes:
        print(f"note: {note}")
    config = load_config(
        ROOT / "config" / "config.yaml",
        overrides={"memory.db_path": DB_PATH},
        project_root=ROOT,
    )
    rubric = Rubric.from_yaml(args.rubric)
    draft = Path(args.input).read_text(encoding="utf-8")

    # Guarantee a genuinely cold start; otherwise run A is not a baseline.
    db = config.path(DB_PATH)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db) + suffix)
        if candidate.exists():
            candidate.unlink()
    print(f"cleared {db}\n")

    print("running three arms...")
    common = {
        "config": config,
        "rubric": rubric,
        "draft": draft,
        "provider_name": args.provider,
        "target": args.target,
    }
    run_a, recalled_a = run("A cold memory ", session_id="ab-cold", use_memory=True, **common)
    run_b, recalled_b = run("B warm memory ", session_id="ab-warm", use_memory=True, **common)
    run_c, _ = run("C no memory   ", session_id="ab-none", use_memory=False, **common)

    print("\n" + "=" * 100)
    print(f"{'run':<28} {'iters':>5}   {'best':>6}   {'status':<16} actions")
    print("-" * 100)
    print(summarise("A  memory on, cold store", run_a))
    print(summarise("B  memory on, warm store", run_b))
    print(summarise("C  memory off (control)", run_c))
    print("=" * 100)

    saved = run_a.iterations - run_b.iterations
    print(
        f"\nA vs C  (cold store vs no store) : "
        f"{run_a.iterations} vs {run_c.iterations} iterations"
        "   <- expected to match; a cold store carries no information"
    )
    print(
        f"B vs A  (warm store vs cold store): "
        f"{run_b.iterations} vs {run_a.iterations} iterations"
    )
    if saved > 0:
        print(
            f"\n=> Memory saved {saved} iteration(s). Run B is a DIFFERENT session that read "
            f"lessons run A wrote."
        )
    else:
        print("\n=> No iteration saved in this configuration.")

    print(f"\nLessons run A recalled: {len(recalled_a)} (it wrote them, but only after the fact)")
    if recalled_b:
        print(f"Lessons run B recalled, written by run A: {len(recalled_b)}")
        for line in recalled_b:
            print(f"  {line}")

    first_thought_a = run_a.records[1].decision.thought if len(run_a.records) > 1 else ""
    first_thought_b = run_b.records[1].decision.thought if len(run_b.records) > 1 else ""
    print("\nIteration 2 decision, side by side:")
    print(f"  A ({run_a.records[1].decision.action}): {first_thought_a[:140]}")
    print(f"  B ({run_b.records[1].decision.action}): {first_thought_b[:140]}")

    if not args.keep_db:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db) + suffix)
            if candidate.exists():
                candidate.unlink()
        print(f"\nremoved {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
