# Memory Design

> **Deliverable:** Milestone 2 — memory tool choice, structure, and a concrete
> example of memory changing the agent's output.

---

## 0. What memory is for here

The loop is a Reflexion agent (see [01_patterns_research.md](01_patterns_research.md) §6.2). Reflect
produces a verbal *lesson* from a numeric score; memory is what makes that lesson available to a
later decision. Without persistence, Reflexion collapses into ordinary self-critique — the agent
notices something useful and immediately forgets it.

So the design question is not "how do I store text and retrieve it". It is **which findings deserve
to outlive the run that produced them, and how do I get them in front of the next decision without
burying it in noise.**

---

## 1. Backend choice: SQLite + `sqlite-vec` + `fastembed`

| Layer | Choice | Why |
|---|---|---|
| Store | `sqlite3` (stdlib) | Zero infrastructure, transactional, one file |
| Keyword recall | FTS5 + BM25 | Ships inside SQLite; catches exact terms embeddings blur |
| Vector recall | `sqlite-vec` | KNN **inside the same database file** |
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` | 384d, ONNX, CPU-only, **no PyTorch** |

### Why not a vector database

I considered Chroma, Qdrant and LanceDB. All would work. SQLite won on three counts:

1. **One file is the whole memory.** Records, the keyword index and the vectors live together in
   `data/memory.db`. One Docker volume, one backup, one thing to delete for a clean demo. A separate
   vector service means two stores that can disagree about what the agent remembers — and the
   failure mode is silent: retrieval returns a vector id whose record was deleted.
2. **It works with the network unplugged.** No recall-time service call. A demo cannot fail because
   a free tier throttled.
3. **The failure story is simple.** `sqlite-vec` is a loadable extension. If it will not load, FTS5
   is still there, in the same file, over the same rows.

### Why `fastembed` and not `sentence-transformers`

Comparable quality at this scale, but `sentence-transformers` pulls PyTorch: the Docker image goes
from roughly 400 MB to roughly 2.5 GB for embeddings of a few hundred short strings. `fastembed`
runs ONNX Runtime and the `bge-small` model is about 130 MB.

Cost: **$0**. Nothing here needs a key, a service, or a network connection after first model
download.

---

## 2. Structure: three tiers with three different lifetimes

```
                    ┌──────────────────────────────────────────────┐
   Reflect ────────▶│  episodic   this session only                │
                    │  lesson     this RUBRIC, any session   ◀──── the Reflexion payload
                    │  profile    this rubric, standing rules      │
                    └──────────────────────────────────────────────┘
                                        │
                    Perceive ◀──────────┘  read once per iteration
```

| Tier | Scope | Written by | Example |
|---|---|---|---|
| `episodic` | one session | every Reflect | `iteration 2: revise_text -> Revised for thesis, evidence: 202 -> 243 words, similarity 0.95` |
| `lesson` | one rubric, all sessions | Reflect, when it found something transferable | `Unattributed figures score no better than no figures; naming the source is what lifts the evidence criterion.` |
| `profile` | one rubric | operator / future work | target score, tone constraints, banned edits |

### Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    uid TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,              -- episodic | lesson | profile
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,      -- normalised sha256 prefix, for dedupe
    session_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    rubric_id TEXT NOT NULL,         -- the scope key that stops cross-domain leakage
    criterion_id TEXT NOT NULL,
    score_delta REAL,                -- what this action was worth, in points
    metadata TEXT NOT NULL,          -- JSON: action, ok, thought
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 1  -- times this lesson has been relearned
);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', ...);
CREATE VIRTUAL TABLE memories_vec USING vec0(id INTEGER PRIMARY KEY,
                                             embedding float[384] distance_metric=cosine);
```

Triggers keep the FTS index in sync with inserts, updates and deletes, so the two indexes can never
drift from the table.

---

## 3. The three required operations

| Operation | Signature | CLI |
|---|---|---|
| **Save** | `save(record) -> uid` | automatic, after every Reflect |
| **Recall** | `recall(query, *, session_id, rubric_id, kinds, limit, min_score) -> [MemoryHit]` | automatic, in every Perceive |
| **Clear session** | `clear_session(session_id) -> int` | `--clear-session <id>` |

Plus `list_sessions()` and `stats()` for operability — `--memory-stats`:

```json
{ "total": 8, "by_kind": {"episodic": 6, "lesson": 2}, "sessions": 1,
  "vector_enabled": true, "vector_dimension": 384, "vectors_indexed": 8,
  "embedder": "fastembed (384d)", "degraded": false }
```

---

## 4. Four decisions worth defending

### 4.1 The recall query describes the *problem*, not the *text*

`build_recall_query` composes the query from the rubric name, the criterion Reflect nominated, and
the two criteria with the most headroom — never from the draft.

Querying with the essay retrieves memories about *similar essays*. Querying with the problem
retrieves memories about *how to solve this problem*, which is what the next decision actually
needs. There is a test asserting the draft's own words never appear in the query.

### 4.2 Lessons are rubric-scoped, not global

A finding about attributing statistics in essays is not evidence about bug reports. Recalling it
there is worse than recalling nothing — it spends prompt budget arguing for an irrelevant edit.
`rubric_id` is the scope key, and `test_a_lesson_learned_on_one_rubric_does_not_leak_into_another`
pins it.

### 4.3 Lessons are ranked but **not** relevance-gated

This is the most consequential policy choice in the module, and it is a deliberate trade of
precision for recall.

Episodic records are voluminous and noisy, so they must clear `recall_min_score`. Lessons are the
opposite: a handful per rubric, each already filtered by Reflect judging it worth keeping. Gating
them on cosine similarity throws away the entire point of Reflexion the first time a query happens
to be phrased differently from the lesson. They are ranked, capped at `max_lessons_per_recall: 3`,
and always offered.

**The risk this accepts:** an irrelevant lesson can reach the prompt. Mitigated by the rubric scope,
the cap of three, and the prompt labelling them as *prior experience, not instructions*. If the
lesson set ever grew past a few dozen per rubric, this would need revisiting.

### 4.4 The keyword channel is ordinal, not calibrated — and says so

The two channels are scored differently on purpose:

| Channel | Score | Calibrated? |
|---|---|---|
| vector | `1 - cosine_distance` | ✅ absolute; `min_score` is meaningful |
| keyword | `0.6 / (1 + rank)` | ❌ ordinal only |

FTS5's BM25 magnitudes depend on corpus size — on a ten-row table they come back around `1e-6`.
Squashing that into a pseudo-probability would produce a number that *looks* absolute and is not,
and `recall_min_score` would then silently reject everything. So the keyword channel contributes a
rank-derived score capped below 1.0 (a keyword hit is evidence of overlap, not of relevance), and
**in keyword-only mode the relevance gate is skipped entirely** rather than applied to a fake score.

When both channels hit: `0.7 × vector + 0.3 × keyword`, configurable via `memory.vector_weight`.

### 4.5 A relearned lesson increments a counter instead of duplicating

Lessons and profiles deduplicate on `(kind, rubric_id, content_hash)`, hashed case- and
whitespace-insensitively. A lesson rediscovered in a later session bumps `hits` and gets a small
ranking boost (up to +20%): a finding independently rediscovered is better evidence than a one-off.
Episodic records never deduplicate — two identical events in different sessions are two facts.

---

## 5. Degradation: four ways this can break, four ways it keeps running

| Failure | Response |
|---|---|
| `fastembed` not installed / model download blocked | `NullEmbedder`, keyword-only recall, note in the run |
| `sqlite-vec` will not load | keyword-only recall, same file, same rows |
| Embedding model **changed** under an existing DB | vector table dimension mismatch is caught; note says to delete the DB |
| Neither channel matches the query | fall back to **recent** records for the scope — better than nothing |
| Database unopenable (bad path, locked) | `NullMemory`, the run proceeds without memory |
| Store raises repeatedly | **circuit breaker**: after 3 consecutive failures the manager stops calling it entirely |

The circuit breaker (`MemoryManager`) is also the Milestone 3 "memory read failure" fallback. Once
tripped, `degraded: true` and the reason appear in `--memory-stats` and in the run's notes. Retrying
a store that has failed three times running costs latency on every iteration and almost never
succeeds.

**A bug this caught during development.** A zero-norm query vector makes `sqlite-vec` return `NULL`
for cosine distance. The resulting `TypeError` was swallowed by the manager's guard, the circuit
breaker tripped, and memory silently stopped working — the loop still completed, reporting nothing
wrong. Fixed by skipping the vector channel for degenerate query vectors and dropping `NULL`-distance
rows. Worth recording because the containment that kept the run alive is exactly what hid the bug;
the fix was found by a test asserting behaviour changed, not by anything crashing.

---

## 6. Concrete example: memory changing the output

Reproduce with `python scripts/memory_ab_demo.py` (offline, no API key). Full transcript in
[demos/m2_memory_ab.txt](demos/m2_memory_ab.txt).

Three arms, identical input, identical target, identical rubric:

| run | condition | iterations | best | actions |
|---|---|---|---|---|
| **A** | memory on, **cold** store | **6** | 96.2% | score → **analyze_text** → revise → score → revise → score |
| **B** | memory on, **warm** store, new session | **5** | 96.2% | score → **revise** → score → revise → score |
| **C** | `--no-memory` (control) | **6** | 96.2% | score → **analyze_text** → revise → score → revise → score |

**A ≡ C is the control that makes B meaningful.** A cold store carries the same information as no
store, so they must behave identically — and they do, action for action. Without that check,
"memory helped" could just mean "the second run of anything is faster".

### What run A wrote

```
[lesson] On this rubric, targeting the two highest-weighted criteria first moves the
         total faster than fixing the lowest raw score.
[lesson] Unattributed figures score no better than no figures; naming the source is
         what lifts the evidence criterion.
```

### What run B did with it — iteration 2, side by side

```
A (no prior experience):
   action  : analyze_text
   thought : "No prior experience with this rubric was recalled. Measure the draft
              directly before spending a revision on a guess."

B (lesson recalled from session ab-cold):
   action  : revise_text
   thought : "Memory says: On this rubric, targeting the two highest-weighted criteria
              first moves the total... Applying that directly to Thesis and Position,
              Evidence and Support rather than rediscovering it."
   args    : { focus_criteria: [thesis, evidence],
               apply_lessons: ["On this rubric, targeting the two highest-weighted..."] }
```

Two distinct effects, both visible in the trace:

1. **The exploration step is skipped.** With no prior experience the agent spends an iteration
   measuring the draft before committing to an edit. With the lesson recalled it goes straight to
   revising — one whole iteration saved.
2. **The lesson reaches the rewrite itself**, not just the reasoning prompt, via the
   `apply_lessons` argument on `revise_text`.

Run B is a **different session id**, reading records a previous session wrote. That is the
cross-session requirement, demonstrated rather than asserted.

### Honesty about the demo

The A/B runs use the offline simulated agent (`--provider mock`), so the *decision rule* linking
recall to behaviour is written by me, not emergent from a model. What is real: the storage, the
embedding, the retrieval, the scoping, the ranking, and the fact that **the only difference between
runs A and B is what appeared in the prompt** — the responder reads the recalled block out of the
prompt text exactly as a model would, and is never handed the memory object.

`python scripts/memory_ab_demo.py --provider groq` runs the identical comparison against a live
model, where the behavioural link is the model's own.

---

## 7. Limitations and what I would do differently

- **Lessons are never retired.** If a lesson turns out to be wrong, nothing removes or down-weights
  it. `hits` counts reinforcement but there is no counter-evidence signal. The right fix is to
  record the score delta of iterations where a lesson was applied and demote lessons that stop
  paying off — that data is already stored (`score_delta`), so it is a scoring change, not a schema
  change.
- **`clear_session` deletes that session's lessons too.** Lessons are attributed to the session that
  discovered them, so wiping a session takes them with it. Defensible (it makes "undo this run"
  complete) but surprising; a `--keep-lessons` flag would be a one-line addition.
- **KNN scans the whole vector table**, then scoping is applied in Python. Correct and fast at
  hundreds to low thousands of records; past that it needs `sqlite-vec` partition keys. The `pool`
  parameter in `search()` is where that change would land.
- **The episodic tier earns less than it costs.** In practice the ReAct scratchpad already tells the
  agent what it tried this session, so episodic recall is largely redundant within a run and useless
  across runs. If I were cutting scope, I would keep lessons and profiles and drop episodic
  persistence to in-memory only.
- **Recall quality is untested against a real corpus.** With two lessons per rubric, ranking barely
  matters. The blend weights and the keyword score curve are reasoned, not measured.

---

## 8. Configuration reference

```yaml
memory:
  enabled: true
  backend: sqlite_vec        # sqlite_vec | sqlite_fts | null
  db_path: data/memory.db
  embedder: fastembed        # fastembed | none
  embed_model: BAAI/bge-small-en-v1.5
  recall_top_k: 5
  recall_min_score: 0.25     # applies to episodic recall only
  lesson_scope: rubric       # rubric | global | session
  episodic_scope: session
  gate_lessons: false        # see §4.3
  max_lessons_per_recall: 3
  vector_weight: 0.7
```

Every value is overridable without touching code:

```bash
AGENTIC_MEMORY__BACKEND=sqlite_fts        # keyword-only, no model download
AGENTIC_MEMORY__EMBEDDER=none             # same, from the other direction
python -m agentic_rubric.cli --set memory.max_lessons_per_recall=5 ...
python -m agentic_rubric.cli --no-memory ...      # the A/B control
```
