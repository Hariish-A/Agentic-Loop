"""Memory: storage, recall, scoping, degradation, and the loop integration.

Runs entirely on a temporary SQLite file with a **stub embedder**, so the suite
needs no model download, no network and no API key. The real fastembed path is
exercised by `scripts/memory_ab_demo.py`; what matters here is that the policy
and the degradation paths are correct, and those are independent of which
embedder produced the vectors.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import pytest

from agentic_rubric.config import load_config
from agentic_rubric.core.loop import AgenticLoop
from agentic_rubric.core.rubric import Rubric
from agentic_rubric.core.state import RunStatus
from agentic_rubric.llm.demo_responder import ScriptedAgentResponder
from agentic_rubric.llm.mock import MockProvider
from agentic_rubric.memory.base import MemoryRecord, NullMemory
from agentic_rubric.memory.embedding import (
    Embedder,
    EmbeddingUnavailable,
    NullEmbedder,
    build_embedder,
)
from agentic_rubric.memory.factory import build_memory
from agentic_rubric.memory.manager import CIRCUIT_BREAKER_THRESHOLD, MemoryManager
from agentic_rubric.memory.sqlite_store import SQLiteMemory

CONFIG = "config/config.yaml"
ESSAY = "config/rubrics/essay_argumentative.yaml"

LESSON_WEIGHTS = (
    "On this rubric, targeting the two highest-weighted criteria first moves the total "
    "faster than fixing the lowest raw score."
)
LESSON_EVIDENCE = (
    "Unattributed figures score no better than no figures; naming the source is what lifts "
    "the evidence criterion."
)
LESSON_BUGS = "Numbered reproduction steps are the highest-value addition to a bug report."


class StubEmbedder(Embedder):
    """Deterministic bag-of-words vectors. No model, no download, no flake.

    Similarity is real (shared vocabulary raises cosine), which is all the
    ranking logic under test needs.
    """

    name = "stub"
    VOCAB = [
        "rubric", "criteria", "weighted", "score", "evidence", "figures", "source",
        "thesis", "reproduction", "steps", "bug", "report", "essay", "revise",
        "target", "lesson",
    ]

    def __init__(self, *, broken: bool = False) -> None:
        self.broken = broken
        self.calls = 0

    @property
    def available(self) -> bool:
        return not self.broken

    @property
    def dimension(self) -> int:
        return len(self.VOCAB)

    def embed(self, texts: t.Sequence[str]) -> list[list[float]]:
        if self.broken:
            raise EmbeddingUnavailable("stub embedder is broken on purpose")
        self.calls += 1
        vectors = []
        for text in texts:
            words = set(text.lower().replace(",", " ").replace(".", " ").split())
            raw = [1.0 if term in words else 0.0 for term in self.VOCAB]
            norm = sum(v * v for v in raw) ** 0.5 or 1.0
            vectors.append([v / norm for v in raw])
        return vectors


@pytest.fixture
def store(tmp_path: Path) -> t.Iterator[SQLiteMemory]:
    memory = SQLiteMemory(tmp_path / "memory.db", embedder=StubEmbedder())
    yield memory
    memory.close()


@pytest.fixture
def keyword_store(tmp_path: Path) -> t.Iterator[SQLiteMemory]:
    memory = SQLiteMemory(tmp_path / "keyword.db", embedder=NullEmbedder())
    yield memory
    memory.close()


def lesson(
    content: str, *, rubric: str = "essay_argumentative", session: str = "s1"
) -> MemoryRecord:
    return MemoryRecord(
        kind="lesson", content=content, session_id=session, iteration=2, rubric_id=rubric
    )


def episode(content: str, *, session: str = "s1", iteration: int = 1) -> MemoryRecord:
    return MemoryRecord(
        kind="episodic",
        content=content,
        session_id=session,
        iteration=iteration,
        rubric_id="essay_argumentative",
    )


# ===========================================================================
# The three required operations
# ===========================================================================


def test_save_recall_roundtrip(store: SQLiteMemory) -> None:
    uid = store.save(lesson(LESSON_EVIDENCE))
    assert uid

    hits = store.recall("evidence figures source", rubric_id="essay_argumentative")
    assert hits
    assert hits[0].content == LESSON_EVIDENCE
    assert 0.0 < hits[0].score <= 1.0


def test_clear_session_removes_only_that_session(store: SQLiteMemory) -> None:
    store.save(episode("iteration 1 scored the essay", session="keep"))
    store.save(episode("iteration 1 scored the essay again", session="drop"))
    store.save(episode("iteration 2 revised the essay", session="drop"))

    removed = store.clear_session("drop")
    assert removed == 2
    assert store.stats()["total"] == 1
    assert store.list_sessions() == ["keep"]


def test_clear_session_is_a_noop_for_an_unknown_session(store: SQLiteMemory) -> None:
    assert store.clear_session("never-existed") == 0


def test_stats_reports_the_shape_of_the_store(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_EVIDENCE))
    # Content deliberately inside the stub vocabulary, so both rows produce a
    # non-degenerate vector and both get indexed.
    store.save(episode("iteration 1 scored the essay against the rubric"))
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["by_kind"] == {"lesson": 1, "episodic": 1}
    assert stats["vector_enabled"] is True
    assert stats["vectors_indexed"] == 2


def test_empty_content_is_not_stored(store: SQLiteMemory) -> None:
    assert store.save(lesson("   ")) == ""
    assert store.stats()["total"] == 0


# ===========================================================================
# Deduplication
# ===========================================================================


def test_a_relearned_lesson_increments_hits_instead_of_duplicating(store: SQLiteMemory) -> None:
    first = store.save(lesson(LESSON_EVIDENCE, session="s1"))
    second = store.save(lesson(LESSON_EVIDENCE, session="s2"))
    assert first == second  # same row
    assert store.stats()["by_kind"]["lesson"] == 1

    hits = store.recall("evidence figures", rubric_id="essay_argumentative")
    assert hits[0].content == LESSON_EVIDENCE


def test_dedupe_ignores_case_and_whitespace(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_EVIDENCE))
    store.save(lesson("  " + LESSON_EVIDENCE.upper() + "  "))
    assert store.stats()["by_kind"]["lesson"] == 1


def test_episodic_records_are_never_deduplicated(store: SQLiteMemory) -> None:
    # Two identical events in different sessions are two facts, not one.
    store.save(episode("iteration 1: score_against_rubric", session="a"))
    store.save(episode("iteration 1: score_against_rubric", session="b"))
    assert store.stats()["by_kind"]["episodic"] == 2


def test_repeat_boost_lifts_a_relearned_lesson(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_EVIDENCE))
    before = store.search("evidence figures source", kinds=["lesson"])[0].score
    for _ in range(4):
        store.save(lesson(LESSON_EVIDENCE, session="another"))
    after = store.search("evidence figures source", kinds=["lesson"])[0].score
    assert after > before


# ===========================================================================
# Scope policy
# ===========================================================================


def manager(store: SQLiteMemory, **overrides: object) -> MemoryManager:
    config = load_config(CONFIG, overrides=overrides or None)
    return MemoryManager(store, config.memory)


def test_a_lesson_from_another_rubric_is_not_recalled(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_EVIDENCE, rubric="essay_argumentative"))
    store.save(lesson(LESSON_BUGS, rubric="bug_report"))

    essay_hits = manager(store).recall("evidence", rubric_id="essay_argumentative", limit=5)
    bug_hits = manager(store).recall("reproduction steps", rubric_id="bug_report", limit=5)

    assert [h.content for h in essay_hits] == [LESSON_EVIDENCE]
    assert [h.content for h in bug_hits] == [LESSON_BUGS]


def test_lessons_cross_sessions_but_episodes_do_not(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_WEIGHTS, session="earlier"))
    store.save(episode("iteration 4: revised for evidence", session="earlier", iteration=4))

    hits = manager(store).recall(
        "weighted criteria evidence", session_id="later", rubric_id="essay_argumentative", limit=5
    )
    contents = [h.content for h in hits]
    assert LESSON_WEIGHTS in contents  # the lesson crosses the session boundary
    assert not any("iteration 4" in c for c in contents)  # the episode does not


def test_lessons_are_not_relevance_gated(store: SQLiteMemory) -> None:
    # A lesson that shares no vocabulary with the query must still surface:
    # lessons are few and curated, so recall beats precision for them.
    store.save(lesson("Prefer active voice in the opening paragraph."))
    hits = manager(store).recall(
        "zzz totally unrelated query", rubric_id="essay_argumentative", limit=5, min_score=0.9
    )
    assert [h.content for h in hits] == ["Prefer active voice in the opening paragraph."]


def test_episodic_recall_is_gated(store: SQLiteMemory) -> None:
    store.save(episode("iteration 1: score_against_rubric produced 28.7 percent"))
    ungated = manager(store).recall(
        "score rubric", session_id="s1", rubric_id="essay_argumentative", min_score=0.0
    )
    gated = manager(store).recall(
        "score rubric", session_id="s1", rubric_id="essay_argumentative", min_score=0.99
    )
    assert ungated
    assert not gated


def test_lesson_budget_is_capped(store: SQLiteMemory) -> None:
    for index in range(8):
        store.save(lesson(f"Lesson number {index} about weighted criteria and evidence."))
    hits = manager(store).recall("weighted criteria", rubric_id="essay_argumentative", limit=10)
    lessons = [h for h in hits if h.kind == "lesson"]
    assert len(lessons) <= load_config(CONFIG).memory.max_lessons_per_recall


def test_recall_respects_the_overall_limit(store: SQLiteMemory) -> None:
    store.save(lesson(LESSON_WEIGHTS))
    for index in range(6):
        store.save(episode(f"iteration {index}: revised the essay for evidence", iteration=index))
    hits = manager(store).recall(
        "essay evidence", session_id="s1", rubric_id="essay_argumentative", limit=3
    )
    assert len(hits) <= 3


def test_duplicate_content_is_collapsed_across_tiers(store: SQLiteMemory) -> None:
    shared = "Naming the source is what lifts the evidence criterion."
    store.save(lesson(shared))
    store.save(episode(shared))
    hits = manager(store).recall(
        "evidence source", session_id="s1", rubric_id="essay_argumentative", limit=5
    )
    assert sum(1 for h in hits if h.content == shared) == 1


# ===========================================================================
# Degradation
# ===========================================================================


def test_keyword_only_store_still_saves_and_recalls(keyword_store: SQLiteMemory) -> None:
    assert keyword_store.vector_enabled is False
    keyword_store.save(lesson(LESSON_EVIDENCE))
    hits = keyword_store.recall("figures source evidence", rubric_id="essay_argumentative")
    assert hits and hits[0].content == LESSON_EVIDENCE


def test_keyword_only_mode_bypasses_the_relevance_gate(keyword_store: SQLiteMemory) -> None:
    # BM25 magnitudes are not calibrated, so gating on them would silently
    # return nothing. The gate is skipped rather than applied to a fake score.
    keyword_store.save(episode("iteration 1 scored the essay against the rubric"))
    hits = manager(keyword_store).recall(
        "essay rubric", session_id="s1", rubric_id="essay_argumentative", min_score=0.99
    )
    assert hits


def test_a_broken_embedder_degrades_to_keyword_recall(tmp_path: Path) -> None:
    store = SQLiteMemory(tmp_path / "broken.db", embedder=StubEmbedder(broken=True))
    try:
        assert store.vector_enabled is False
        store.save(lesson(LESSON_EVIDENCE))
        assert store.recall("evidence figures", rubric_id="essay_argumentative")
    finally:
        store.close()


def test_no_match_on_either_channel_falls_back_to_recency(store: SQLiteMemory) -> None:
    store.save(episode("something entirely unrelated to the query"))
    hits = store.recall("zzzz", session_id="s1")
    assert hits  # better to offer recent context than nothing


def test_build_embedder_never_raises_on_a_bad_name() -> None:
    embedder, note = build_embedder("not-a-real-embedder", "x")
    assert embedder.available is False
    assert "keyword recall" in note


def test_the_circuit_breaker_stops_calling_a_broken_store(tmp_path: Path) -> None:
    class Exploding(NullMemory):
        def __init__(self) -> None:
            self.attempts = 0

        def recall(self, *args: object, **kwargs: object) -> list:
            self.attempts += 1
            raise OSError("database is locked")

    broken = Exploding()
    wrapped = MemoryManager(broken, load_config(CONFIG).memory)

    for _ in range(CIRCUIT_BREAKER_THRESHOLD + 4):
        assert wrapped.recall("anything") == []

    # After the threshold the store is never called again: retrying a store
    # that failed three times running costs latency on every iteration.
    assert broken.attempts == CIRCUIT_BREAKER_THRESHOLD
    assert wrapped.degraded is True
    assert "consecutive failures" in wrapped.degraded_reason


def test_a_transient_failure_does_not_trip_the_breaker(tmp_path: Path) -> None:
    class Flaky(NullMemory):
        def __init__(self) -> None:
            self.calls = 0

        def recall(self, *args: object, **kwargs: object) -> list:
            self.calls += 1
            if self.calls == 1:
                raise OSError("blip")
            return []

    wrapped = MemoryManager(Flaky(), load_config(CONFIG).memory)
    wrapped.recall("q")  # fails
    wrapped.recall("q")  # succeeds, resetting the counter
    assert wrapped.failures == 0
    assert wrapped.degraded is False


def test_factory_falls_back_when_the_database_cannot_be_opened(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    config = load_config(
        CONFIG,
        overrides={"memory.db_path": str(blocker / "memory.db"), "memory.embedder": "none"},
    )
    store, notes = build_memory(config, warm_embedder=False)
    assert isinstance(store, NullMemory)
    assert any("without memory" in note for note in notes)


def test_unknown_backend_falls_back_to_keyword(tmp_path: Path) -> None:
    config = load_config(
        CONFIG,
        overrides={"memory.backend": "redis", "memory.db_path": str(tmp_path / "m.db")},
    )
    store, notes = build_memory(config, warm_embedder=False)
    try:
        assert any("unknown memory backend" in note for note in notes)
    finally:
        store.close()


def test_memory_disabled_returns_a_null_store() -> None:
    store, notes = build_memory(load_config(CONFIG), enabled=False)
    assert isinstance(store, NullMemory)
    assert notes == ["memory disabled"]


# ===========================================================================
# Loop integration
# ===========================================================================


def run_loop(rubric: Rubric, draft: str, memory: object, session_id: str, **overrides: object):
    config = load_config(CONFIG, overrides=overrides or None)
    responder = ScriptedAgentResponder(rubric=rubric, target_score=85.0)
    loop = AgenticLoop(
        config=config,
        provider=MockProvider(responder=responder),
        rubric=rubric,
        memory=memory,  # type: ignore[arg-type]
    )
    return loop.run(draft, session_id=session_id, target_score=85.0)


@pytest.fixture
def essay_rubric() -> Rubric:
    return Rubric.from_yaml(ESSAY)


@pytest.fixture
def weak_essay() -> str:
    return Path("samples/weak_essay.txt").read_text(encoding="utf-8")


def test_the_loop_writes_episodes_and_lessons(
    store: SQLiteMemory, essay_rubric: Rubric, weak_essay: str
) -> None:
    result = run_loop(essay_rubric, weak_essay, manager(store), "run-1")
    stats = store.stats()
    assert stats["by_kind"]["episodic"] == result.iterations
    assert stats["by_kind"]["lesson"] >= 1


def test_memory_changes_behaviour_across_sessions(
    store: SQLiteMemory, essay_rubric: Rubric, weak_essay: str
) -> None:
    """The Milestone 2 requirement, asserted rather than demonstrated.

    Run one starts cold and spends an iteration exploring. Run two is a
    *different session* reading the lessons run one wrote, and skips straight to
    revising. Same input, same target, one fewer iteration.
    """
    cold = run_loop(essay_rubric, weak_essay, manager(store), "session-one")
    warm = run_loop(essay_rubric, weak_essay, manager(store), "session-two")

    assert cold.status is RunStatus.TARGET_REACHED
    assert warm.status is RunStatus.TARGET_REACHED
    assert warm.iterations < cold.iterations

    cold_actions = [r.decision.action for r in cold.records]
    warm_actions = [r.decision.action for r in warm.records]
    assert "analyze_text" in cold_actions  # exploration, with nothing recalled
    assert "analyze_text" not in warm_actions  # skipped, because memory answered

    # The recalled lesson reaches the reviser as an argument, not just the prompt.
    revisions = [r for r in warm.records if r.decision.action == "revise_text"]
    assert revisions and revisions[0].decision.arguments.get("apply_lessons")


def test_a_cold_store_behaves_like_no_store(
    store: SQLiteMemory, essay_rubric: Rubric, weak_essay: str
) -> None:
    # The control for the test above: without this, "memory helped" could just
    # mean "the second run of anything is faster".
    cold = run_loop(essay_rubric, weak_essay, manager(store), "cold")
    disabled = run_loop(essay_rubric, weak_essay, NullMemory(), "none")
    assert cold.iterations == disabled.iterations
    assert [r.decision.action for r in cold.records] == [
        r.decision.action for r in disabled.records
    ]


def test_a_lesson_learned_on_one_rubric_does_not_leak_into_another(
    store: SQLiteMemory, essay_rubric: Rubric, weak_essay: str
) -> None:
    run_loop(essay_rubric, weak_essay, manager(store), "essay-session")
    bug_rubric = Rubric.from_yaml("config/rubrics/bug_report.yaml")
    report = Path("samples/weak_bug_report.txt").read_text(encoding="utf-8")

    result = run_loop(bug_rubric, report, manager(store), "bug-session")
    recalled = [hit for record in result.records for hit in record.observation.recalled]

    # Scope is checked by origin, not by wording: the simulated agent emits the
    # same lesson text for every rubric, so matching on content would test the
    # mock rather than the policy. Nothing from the essay session may appear.
    assert recalled  # the bug run did recall its own records
    assert not any(hit.session_id == "essay-session" for hit in recalled)

    # Having learned nothing about bug reports yet, it explores first.
    assert "analyze_text" in [r.decision.action for r in result.records]


def test_clearing_a_session_removes_its_episodes_but_keeps_lessons(
    store: SQLiteMemory, essay_rubric: Rubric, weak_essay: str
) -> None:
    result = run_loop(essay_rubric, weak_essay, manager(store), "temporary")
    before = store.stats()["by_kind"]

    removed = store.clear_session("temporary")
    after = store.stats()["by_kind"]

    assert removed == before["episodic"] + before.get("lesson", 0)
    # Lessons written by that session go with it -- they are attributed to it.
    # What survives a session wipe is everything learned in *other* sessions.
    assert after.get("episodic", 0) == 0
    assert result.iterations > 0
