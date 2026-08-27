"""PERCEIVE -- turn raw state into a structured Observation. No LLM.

This step is deliberately model-free. Three reasons:

* **It keeps the four steps genuinely distinct.** If Perceive called a model,
  it would be doing Reason's job with a different prompt, and the "four
  cognitive steps" would be four prompts wearing hats.
* **It costs nothing and cannot hallucinate.** Word counts, Flesch scores and
  regex probes are the same every run.
* **It makes Reason reproducible.** Reason sees an Observation and nothing
  else, so the same Observation always builds the same prompt.

Perceive is also where memory is *read* (Milestone 2). Reading here rather than
inside Reason means recall happens exactly once per iteration, is visible in the
Observation, and can fail without the reasoning step needing to know.
"""

from __future__ import annotations

import typing as t

from ..config import AppConfig
from ..memory.base import MemoryStore
from ..tools.handlers.scoring import run_probes
from ..tools.text_stats import compute_metrics
from .state import LoopState, MemoryHit, Observation

TRUNCATION_SUFFIX = "\n\n[... draft truncated for the prompt; the full text is still being edited]"


def build_recall_query(state: LoopState) -> str:
    """Compose the memory query for this iteration.

    Built from what the agent is about to work on -- the rubric, the criterion
    Reflect nominated, and the criteria with the most headroom -- rather than
    from the draft text. Querying with the essay itself retrieves memories about
    *similar essays*; querying with the problem retrieves memories about *how to
    solve this problem*, which is what the next decision needs.
    """
    parts = [state.rubric.name]

    reflection = state.last_reflection
    if reflection and reflection.next_focus:
        parts.append(state.rubric.criterion(reflection.next_focus).name)

    card = state.workspace.scorecard
    if card is not None:
        parts.extend(
            state.rubric.criterion(entry.criterion_id).name for entry in card.weakest(2)
        )
    elif not state.workspace.scorecard_history:
        parts.append("initial scoring")

    seen: set[str] = set()
    unique = [p for p in parts if not (p.lower() in seen or seen.add(p.lower()))]
    return " ".join(unique)


def recall_context(
    state: LoopState,
    config: AppConfig,
    memory: MemoryStore | None,
) -> tuple[tuple[MemoryHit, ...], list[str]]:
    """Read memory, returning hits and any notes about degraded behaviour.

    A memory failure is never fatal. The loop is *better* with memory, not
    dependent on it, so an unreadable store downgrades to a note in the
    Observation and the run continues.
    """
    if memory is None or not config.memory.enabled:
        return (), []

    try:
        hits = memory.recall(
            build_recall_query(state),
            session_id=state.session_id,
            rubric_id=state.rubric.id,
            limit=config.memory.recall_top_k,
            min_score=config.memory.recall_min_score,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the run
        return (), [f"memory recall failed ({type(exc).__name__}: {exc}); continuing without it"]

    if getattr(memory, "degraded", False):
        # The store's own circuit breaker has opened, so `recall` returned an
        # empty list without raising. Reported on every affected iteration
        # rather than once, because "the agent ran blind for four iterations"
        # and "the agent ran blind for one" are different facts. Silent
        # degradation is the exact failure Milestone 2 shipped and had to fix.
        reason = getattr(memory, "degraded_reason", "") or "memory is unavailable"
        return tuple(hits), [f"running without memory: {reason}"]

    return tuple(hits), []


def perceive(
    state: LoopState,
    config: AppConfig,
    memory: MemoryStore | None = None,
) -> Observation:
    """Assemble everything the agent can see at the start of this iteration."""
    notes: list[str] = []
    draft = state.workspace.draft

    # Metrics and probes run on the FULL draft; only the prompt view is capped.
    metrics = compute_metrics(draft)
    probes = run_probes(state.rubric, draft)

    limit = config.guardrails.max_input_chars
    if limit and len(draft) > limit:
        draft = draft[:limit] + TRUNCATION_SUFFIX
        notes.append(
            f"the draft exceeds the {limit}-character prompt limit and has been truncated "
            "for display; measurements above cover the whole text"
        )

    recalled, memory_notes = recall_context(state, config, memory)
    notes.extend(memory_notes)

    if state.last_reflection is None and state.iteration > 1:
        notes.append("no reflection from the previous iteration was available")

    return Observation(
        iteration=state.iteration,
        draft=draft,
        rubric=state.rubric,
        target_score=state.target_score,
        metrics=metrics,
        max_iterations=config.loop.max_iterations,
        latest_score=state.workspace.scorecard,
        score_history=state.score_history,
        probe_results=probes,
        recalled=recalled,
        scratchpad=tuple(state.scratchpad),
        last_reflection=state.last_reflection,
        best_score=state.best_score,
        notes=tuple(notes) + tuple(state.notes),
    )


__all__: t.Sequence[str] = ["build_recall_query", "perceive", "recall_context"]
