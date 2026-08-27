"""SQLite-backed memory: one file, two recall channels, no server.

Why SQLite rather than a vector database:

* **One file holds everything.** Records, the keyword index and the vectors all
  live in `data/memory.db`. One Docker volume, one backup, one thing to delete
  when a demo needs a clean slate. A separate vector service would mean two
  stores that can disagree about what the agent remembers.
* **It works offline.** No network at recall time, so the demo cannot fail
  because a free tier throttled.
* **`sqlite3` is stdlib.** The vector half (`sqlite-vec`) is a small loadable
  extension, and when it is missing everything still runs on FTS5.

Two recall channels, deliberately unequal:

``vector``
    `sqlite-vec` KNN with cosine distance. Calibrated: ``1 - distance`` is a
    real similarity, so ``recall_min_score`` means something against it.
``keyword``
    FTS5/BM25. Kept because it catches exact terms an embedding blurs -- a
    criterion id, a specific phrase. Its BM25 magnitudes are **not** calibrated
    across corpus sizes (on a ten-row table they are ~1e-6), so it contributes
    an *ordinal* score derived from rank, never a pretend-absolute one. See
    :data:`KEYWORD_BASE_SCORE` and docs/02_memory_design.md.

Both channels degrade independently. No extension, no vectors, keyword only. No
embedder, same. Neither available and the store still saves and returns recent
records by recency, because a memory that raises is worse than a memory that
forgets.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import typing as t
from pathlib import Path

from ..core.state import MemoryHit, new_id
from .base import MemoryRecord, MemoryStore
from .embedding import Embedder, EmbeddingUnavailable, NullEmbedder

SCHEMA_VERSION = 1

#: Best possible keyword score. Deliberately below 1.0: a keyword hit is
#: evidence of overlap, not of relevance, and should not outrank a strong
#: semantic match.
KEYWORD_BASE_SCORE = 0.6

#: Multiplier ceiling for a lesson that has been relearned in several sessions.
#: A finding independently rediscovered is better evidence than a one-off.
REPEAT_BOOST_PER_HIT = 0.05
REPEAT_BOOST_MAX_HITS = 4

#: Kinds that are deduplicated on save rather than appended.
DEDUPED_KINDS = ("lesson", "profile")

_WORD = re.compile(r"[A-Za-z0-9_]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY,
    uid          TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    session_id   TEXT    NOT NULL DEFAULT '',
    iteration    INTEGER NOT NULL DEFAULT 0,
    rubric_id    TEXT    NOT NULL DEFAULT '',
    criterion_id TEXT    NOT NULL DEFAULT '',
    score_delta  REAL,
    metadata     TEXT    NOT NULL DEFAULT '{}',
    created_at   TEXT    NOT NULL,
    last_seen_at TEXT    NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_memories_kind    ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_rubric  ON memories(rubric_id);
CREATE INDEX IF NOT EXISTS idx_memories_dedupe  ON memories(kind, rubric_id, content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class ScoredRecord(t.NamedTuple):
    record: MemoryRecord
    score: float
    channel: str
    row_id: int


class SQLiteMemory(MemoryStore):
    """Episodic, lesson and profile memory in a single SQLite file."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        embedder: Embedder | None = None,
        enable_vector: bool = True,
        vector_weight: float = 0.7,
    ) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or NullEmbedder()
        self.vector_weight = min(1.0, max(0.0, vector_weight))
        self.notes: list[str] = []

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

        self._vector_ready = False
        self._vector_dim = 0
        self._serialize: t.Callable[[t.Sequence[float]], bytes] | None = None
        if enable_vector and self.embedder.available:
            self._enable_vector()

    # -- vector setup -------------------------------------------------------

    def _enable_vector(self) -> None:
        """Load sqlite-vec and create the vector table. Degrades on any failure."""
        try:
            import sqlite_vec
        except ImportError as exc:
            self.notes.append(f"sqlite-vec is not installed ({exc}); keyword recall only")
            return
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception as exc:  # noqa: BLE001 - some builds forbid extensions
            self.notes.append(
                f"sqlite-vec could not be loaded ({type(exc).__name__}: {exc}); keyword recall only"
            )
            return

        dimension = self.embedder.dimension
        if not dimension:
            # Dimension unknown until the model runs once.
            probe = self.embedder.embed_one("dimension probe")
            dimension = len(probe) if probe else 0
        if not dimension:
            self.notes.append("embedder produced no vectors; keyword recall only")
            return

        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f"id INTEGER PRIMARY KEY, embedding float[{dimension}] distance_metric=cosine)"
            )
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            # Most likely an existing table built at a different dimension --
            # i.e. the embed model changed under an old database.
            self.notes.append(
                f"vector table unusable ({exc}); keyword recall only. "
                "If the embedding model changed, delete the database and re-run."
            )
            return

        self._serialize = sqlite_vec.serialize_float32
        self._vector_dim = dimension
        self._vector_ready = True

    @property
    def vector_enabled(self) -> bool:
        return self._vector_ready

    @property
    def describe(self) -> str:
        channel = (
            f"vector+keyword ({self.embedder.describe()})" if self._vector_ready else "keyword"
        )
        return f"SQLiteMemory[{channel}] {self.db_path.name}"

    # -- write --------------------------------------------------------------

    def save(self, record: MemoryRecord) -> str:
        """Persist a record, deduplicating lessons and profiles.

        A lesson relearned in a later session does not become a second row; its
        ``hits`` counter increments instead. That keeps the lesson set small
        enough to recall in full, and turns repetition into a usable ranking
        signal rather than duplicate noise in the prompt.
        """
        if not record.content:
            return ""

        content_hash = record.content_hash
        if record.kind in DEDUPED_KINDS:
            existing = self._conn.execute(
                "SELECT id, uid FROM memories "
                "WHERE kind = ? AND rubric_id = ? AND content_hash = ? LIMIT 1",
                (record.kind, record.rubric_id, content_hash),
            ).fetchone()
            if existing is not None:
                self._conn.execute(
                    "UPDATE memories SET hits = hits + 1, last_seen_at = ? WHERE id = ?",
                    (record.created_at, existing["id"]),
                )
                self._conn.commit()
                return str(existing["uid"])

        uid = new_id("mem")
        cursor = self._conn.execute(
            "INSERT INTO memories (uid, kind, content, content_hash, session_id, iteration, "
            "rubric_id, criterion_id, score_delta, metadata, created_at, last_seen_at, hits) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                uid,
                record.kind,
                record.content,
                content_hash,
                record.session_id,
                record.iteration,
                record.rubric_id,
                record.criterion_id,
                record.score_delta,
                json.dumps(record.metadata, default=str),
                record.created_at,
                record.created_at,
            ),
        )
        row_id = int(cursor.lastrowid or 0)
        self._index_vector(row_id, record.content)
        self._conn.commit()
        return uid

    def _index_vector(self, row_id: int, content: str) -> None:
        if not self._vector_ready or self._serialize is None:
            return
        try:
            vector = self.embedder.embed_one(content)
        except EmbeddingUnavailable:
            vector = None
        if not vector or not any(vector):
            # A degenerate vector would be unrankable at query time; leave the
            # row keyword-searchable instead of indexing something useless.
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories_vec(id, embedding) VALUES (?, ?)",
                (row_id, self._serialize(vector)),
            )
        except sqlite3.Error as exc:
            # A failed vector index must not lose the record itself.
            self.notes.append(f"vector index write failed for row {row_id}: {exc}")

    # -- read ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        kinds: t.Sequence[str] | None = None,
        session_id: str | None = None,
        rubric_id: str | None = None,
        limit: int = 10,
        pool: int = 60,
    ) -> list[ScoredRecord]:
        """Rank records for ``query`` under the given scope.

        KNN runs over the whole vector table and scoping is applied afterwards
        in Python, because `sqlite-vec` KNN does not accept arbitrary WHERE
        predicates. Correct and simple at this scale (hundreds to low thousands
        of records); it would need partition keys past that, and the ``pool``
        parameter is where that change would land.
        """
        allowed = self._scope_ids(kinds=kinds, session_id=session_id, rubric_id=rubric_id)
        if not allowed:
            return []

        vector_scores = self._vector_scores(query, pool) if self._vector_ready else {}
        keyword_scores = self._keyword_scores(query, pool)

        blended: dict[int, tuple[float, str]] = {}
        for row_id in set(vector_scores) | set(keyword_scores):
            if row_id not in allowed:
                continue
            vector = vector_scores.get(row_id)
            keyword = keyword_scores.get(row_id)
            if vector is not None and keyword is not None:
                score = self.vector_weight * vector + (1.0 - self.vector_weight) * keyword
                channel = "vector+keyword"
            elif vector is not None:
                score, channel = vector, "vector"
            else:
                score, channel = float(keyword or 0.0), "keyword"
            blended[row_id] = (score, channel)

        if not blended:
            # Neither channel matched. Recent records for this scope are a better
            # answer than nothing: at worst the agent ignores them.
            return self._recent(allowed, limit)

        rows = self._rows(list(blended))
        scored = [
            ScoredRecord(
                record=self._to_record(row),
                score=self._apply_repeat_boost(blended[row["id"]][0], int(row["hits"])),
                channel=blended[row["id"]][1],
                row_id=int(row["id"]),
            )
            for row in rows
        ]
        scored.sort(key=lambda item: (-item.score, -item.row_id))
        return scored[:limit]

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        rubric_id: str | None = None,
        kinds: t.Sequence[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        """The plain :class:`MemoryStore` contract: search, gate, convert.

        Scope *policy* (lessons cross-session, episodes session-local, lessons
        ungated) lives in :class:`~.manager.MemoryManager`. This method is the
        unopinionated version, used directly by tests and by anything that wants
        raw retrieval.
        """
        results = self.search(
            query, kinds=kinds, session_id=session_id, rubric_id=rubric_id, limit=limit
        )
        gate = min_score if self._vector_ready else 0.0
        return [self._to_hit(item) for item in results if item.score >= gate]

    # -- scoring channels ---------------------------------------------------

    def _vector_scores(self, query: str, pool: int) -> dict[int, float]:
        if not self._vector_ready or self._serialize is None or not query.strip():
            return {}
        try:
            vector = self.embedder.embed_one(query)
        except EmbeddingUnavailable:
            return {}
        if not vector or not any(vector):
            # Cosine distance against a zero vector is undefined; sqlite-vec
            # returns NULL for it. Skip the channel rather than feed the
            # ranking NULLs and let keyword recall answer instead.
            return {}
        try:
            rows = self._conn.execute(
                "SELECT id, distance FROM memories_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (self._serialize(vector), pool),
            ).fetchall()
        except sqlite3.Error as exc:
            self.notes.append(f"vector search failed ({exc}); keyword recall only for this query")
            return {}
        # Cosine distance in [0, 2]; similarity below zero is anti-correlation
        # and carries no useful signal here, so it clamps to zero. A NULL
        # distance means the *stored* vector was degenerate -- drop that row
        # rather than crash the whole recall on one bad record.
        scores: dict[int, float] = {}
        for row in rows:
            distance = row["distance"]
            if distance is None:
                continue
            scores[int(row["id"])] = max(0.0, 1.0 - float(distance))
        return scores

    def _keyword_scores(self, query: str, pool: int) -> dict[int, float]:
        """BM25 hits converted to an ordinal score. See the module docstring."""
        terms = _WORD.findall(query.lower())
        if not terms:
            return {}
        # FTS5 MATCH has its own syntax; feeding raw text in would be a parse
        # error on any punctuation. Tokenise and OR the terms explicitly.
        match = " OR ".join(sorted(set(terms)))
        try:
            rows = self._conn.execute(
                "SELECT rowid AS id FROM memories_fts WHERE memories_fts MATCH ? "
                "ORDER BY bm25(memories_fts) LIMIT ?",
                (match, pool),
            ).fetchall()
        except sqlite3.Error as exc:
            self.notes.append(f"keyword search failed ({exc})")
            return {}
        return {
            int(row["id"]): KEYWORD_BASE_SCORE / (1.0 + rank) for rank, row in enumerate(rows)
        }

    @staticmethod
    def _apply_repeat_boost(score: float, hits: int) -> float:
        boost = 1.0 + REPEAT_BOOST_PER_HIT * min(max(0, hits - 1), REPEAT_BOOST_MAX_HITS)
        return min(1.0, score * boost)

    # -- scoping and rows ---------------------------------------------------

    def _scope_ids(
        self,
        *,
        kinds: t.Sequence[str] | None,
        session_id: str | None,
        rubric_id: str | None,
    ) -> set[int]:
        clauses: list[str] = []
        params: list[t.Any] = []
        if kinds:
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if rubric_id is not None:
            # Records written before rubric tagging existed have an empty
            # rubric_id; treat them as global rather than orphaning them.
            clauses.append("(rubric_id = ? OR rubric_id = '')")
            params.append(rubric_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT id FROM memories{where}", params).fetchall()
        return {int(row["id"]) for row in rows}

    def _rows(self, ids: t.Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return list(
            self._conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", list(ids))
        )

    def _recent(self, allowed: set[int], limit: int) -> list[ScoredRecord]:
        rows = self._rows(sorted(allowed, reverse=True)[: limit * 3])
        rows.sort(key=lambda row: str(row["created_at"]), reverse=True)
        return [
            ScoredRecord(record=self._to_record(row), score=0.0, channel="recency",
                         row_id=int(row["id"]))
            for row in rows[:limit]
        ]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> MemoryRecord:
        record = MemoryRecord(
            kind=row["kind"],
            content=row["content"],
            session_id=row["session_id"],
            iteration=int(row["iteration"]),
            rubric_id=row["rubric_id"],
            criterion_id=row["criterion_id"],
            score_delta=row["score_delta"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
        )
        record.metadata.setdefault("uid", row["uid"])
        record.metadata.setdefault("hits", int(row["hits"]))
        return record

    @staticmethod
    def _to_hit(item: ScoredRecord) -> MemoryHit:
        return MemoryHit(
            kind=item.record.kind,
            content=item.record.content,
            score=round(item.score, 4),
            session_id=item.record.session_id,
            iteration=item.record.iteration,
        )

    # -- maintenance --------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Delete every record for one session, vectors included."""
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE session_id = ?", (session_id,)
        ).fetchall()
        ids = [int(row["id"]) for row in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        if self._vector_ready:
            self._conn.execute(f"DELETE FROM memories_vec WHERE id IN ({placeholders})", ids)
        # The FTS index is kept in sync by the AFTER DELETE trigger.
        self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
        self._conn.commit()
        return len(ids)

    def clear_all(self) -> int:
        """Wipe the store. Used by the A/B demo to guarantee a cold start."""
        total = int(self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
        if self._vector_ready:
            self._conn.execute("DELETE FROM memories_vec")
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return total

    def list_sessions(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT session_id, MAX(created_at) AS latest FROM memories "
            "WHERE session_id != '' GROUP BY session_id ORDER BY latest DESC"
        ).fetchall()
        return [row["session_id"] for row in rows]

    def stats(self) -> dict[str, t.Any]:
        by_kind = {
            row["kind"]: int(row["n"])
            for row in self._conn.execute("SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind")
        }
        total = sum(by_kind.values())
        vectors = 0
        if self._vector_ready:
            vectors = int(
                self._conn.execute("SELECT COUNT(*) AS n FROM memories_vec").fetchone()["n"]
            )
        return {
            "db_path": str(self.db_path),
            "total": total,
            "by_kind": by_kind,
            "sessions": len(self.list_sessions()),
            "vector_enabled": self._vector_ready,
            "vector_dimension": self._vector_dim,
            "vectors_indexed": vectors,
            "embedder": self.embedder.describe(),
            "notes": list(self.notes),
        }

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()


__all__ = [
    "DEDUPED_KINDS",
    "KEYWORD_BASE_SCORE",
    "SCHEMA_VERSION",
    "SQLiteMemory",
    "ScoredRecord",
]
