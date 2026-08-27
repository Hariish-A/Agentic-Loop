# Progress Log

Companion to [Plan.md](Plan.md). Newest entry first.
If a session ends abruptly, read **▶ Resume here** at the top, diff it against `Plan.md`, and continue.

---

## ▶ Resume here

| | |
|---|---|
| **Last completed** | Milestone 2 — Memory Integration (M2-1 … M2-10), plus the provider switch to GroqCloud |
| **Next task** | **M3-1** — `harness/retry.py`: exponential backoff + jitter, `Retry-After`, retryable-vs-terminal classification |
| **Already done from M3** | The **memory circuit breaker** (`memory/manager.py`) and **provider failover** (`cli.resolve_provider`) landed early. M3 moves failover inside the run loop and adds retry, tracing, guardrails and Docker. |
| **Then** | M3-2 (fallbacks) → M3-3/4/5 (observability) → M3-6/7 (guardrails) → M3-8 (runner) → M3-10 (Docker) → M3-11 (doc) |
| **Blocked on** | Nothing offline. **Add `GROQ_API_KEY` to `.env`** for live runs, then `python scripts/preflight.py --ping`. |

**Verify the checkout is healthy before continuing:**

```bash
.venv/Scripts/python.exe -m pytest -q                      # expect: 166 passed
.venv/Scripts/python.exe -m ruff check src tests scripts   # expect: All checks passed!
.venv/Scripts/python.exe scripts/preflight.py              # config + provider chain
.venv/Scripts/python.exe scripts/memory_ab_demo.py         # memory A/B, offline
PYTHONPATH=src .venv/Scripts/python.exe -m agentic_rubric.cli \
    --input samples/weak_essay.txt --provider mock         # full offline run
```

---

## 2026-08-27 — Milestone 2: Memory Integration ✅

**Status:** complete · **Tasks:** M2-1 … M2-10 · **Tests:** 166 passing (31 new) · **Lint:** clean

### Provider switch (done first, as requested)

Gemini and Grok removed. **GroqCloud `openai/gpt-oss-120b` primary, Ollama secondary, mock for
tests.** Provider identifiers in code are `groq`, `ollama`, `mock`.

Two Groq specifics were handled explicitly rather than discovered in production:

- **`messages[].name` is unsupported on Groq.** Added a `supports_message_name` capability flag on
  the provider config; the client strips the field. No effect today — each loop step sends a fresh
  conversation, so no tool-result turns are replayed — but it would 400 the first time someone added
  conversation replay.
- **Groq converts `temperature=0` to `1e-8`.** The judge still runs at 0.0; it is *near*-deterministic
  there rather than exactly deterministic. Documented in `config.yaml` rather than left as a
  surprise, because the loop steers on differences between two scores.

Provider quirks live in config as capability flags. The client never branches on a provider name.

### Memory demo result (offline, `--provider mock`, no API key)

```
run                        iters   best     actions
A  memory on, cold store     6     96.2%    score -> analyze_text -> revise -> score -> revise -> score
B  memory on, warm store     5     96.2%    score -> revise -> score -> revise -> score
C  memory off (control)      6     96.2%    score -> analyze_text -> revise -> score -> revise -> score
```

**A ≡ C is the control that makes B meaningful.** A cold store carries the same information as no
store, so they must behave identically — and they do, action for action. Without that check, "memory
helped" could just mean "the second run of anything is faster". Run B is a *different session id*
reading lessons run A wrote.

Full transcripts: `docs/demos/m2_memory_ab.txt`, `m2_cold_session.txt`, `m2_warm_session.txt`.

### What was built

| Area | Files | Notes |
|---|---|---|
| Embeddings | `memory/embedding.py` | `fastembed` ONNX (384d, no PyTorch); lazy load; never raises |
| Store | `memory/sqlite_store.py` | One file: rows + FTS5 + `sqlite-vec`, kept in sync by triggers |
| Policy | `memory/manager.py` | Tier scoping, hybrid ranking, dedupe, circuit breaker |
| Assembly | `memory/factory.py` | Backend selection; every failure degrades with a reason |
| CLI | `cli.py` | `--no-memory`, `--clear-session`, `--memory-stats` |
| A/B harness | `scripts/memory_ab_demo.py` | Three arms, reproducible, offline |
| Doc | `docs/02_memory_design.md` | Choice, schema, policy, A/B transcript, limitations |

### Decisions made (and why)

- **SQLite + `sqlite-vec` beats a vector database here.** Records, keyword index and vectors live in
  one file: one Docker volume, one backup, one thing to delete for a clean demo. Two stores can
  disagree about what the agent remembers, and that failure mode is silent.
- **`fastembed` over `sentence-transformers`.** Comparable quality at this scale;
  `sentence-transformers` pulls PyTorch and takes the image from ~400 MB to ~2.5 GB for embeddings
  of a few hundred short strings.
- **Lessons are rubric-scoped, not global.** A finding about attributing statistics in essays is not
  evidence about bug reports; recalling it there is worse than recalling nothing.
- **Lessons are ranked but NOT relevance-gated; episodic recall is.** The most consequential policy
  call in the module. Lessons are few and already filtered by Reflect judging them worth keeping;
  gating them on cosine similarity discards the point of Reflexion the first time a query is phrased
  differently. Capped at three per recall, and the prompt labels them as prior experience, not
  instructions.
- **The keyword channel is ordinal, not calibrated — and says so.** FTS5 BM25 magnitudes depend on
  corpus size (~1e-6 on a ten-row table). Squashing that into a pseudo-probability produces a number
  that *looks* absolute and is not, and `recall_min_score` would then silently reject everything.
  Keyword hits score `0.6/(1+rank)`, and in keyword-only mode the gate is **skipped** rather than
  applied to a fake score.
- **The recall query describes the problem, not the text.** Built from the rubric and the criteria
  with the most headroom. Querying with the essay retrieves memories about similar essays; querying
  with the problem retrieves memories about how to solve it. A test asserts the draft's own words
  never reach the query.
- **A relearned lesson increments `hits` instead of duplicating**, with a small ranking boost: a
  finding independently rediscovered is better evidence than a one-off. Episodic records never
  deduplicate — two identical events in different sessions are two facts.
- **The offline agent reads recalled memory out of the prompt text**, exactly as a model would, and
  is never handed the memory object. That is what makes the A/B comparison meaningful: the only
  difference between runs A and B is what appeared in the prompt.

### Bug found and fixed during development

A zero-norm query vector makes `sqlite-vec` return `NULL` for cosine distance. The resulting
`TypeError` was swallowed by the manager's guard, the circuit breaker tripped after three
consecutive failures, and **memory silently stopped working while the run completed reporting
nothing wrong**.

Fixed by skipping the vector channel for degenerate query vectors and dropping `NULL`-distance rows.
Worth recording because the error containment that kept the run alive is exactly what hid the bug —
it was caught by a test asserting behaviour *changed*, not by anything crashing. An argument for
behavioural assertions over "it did not raise" assertions in agent code.

### Verification evidence

```
$ .venv/Scripts/python.exe -m pytest -q
166 passed
    test_config 12   test_llm_layer 40   test_loop 30
    test_memory 31   test_rubric 19      test_tools 34

$ .venv/Scripts/python.exe -m ruff check src tests scripts
All checks passed!

$ python -m agentic_rubric.cli --memory-stats
  total: 8   by_kind: {'episodic': 6, 'lesson': 2}   sessions: 1
  vector_enabled: True   vector_dimension: 384   vectors_indexed: 8
  embedder: fastembed (384d)   degraded: False

$ python -m agentic_rubric.cli --clear-session cli-demo
cleared 8 record(s) for session cli-demo          # total afterwards: 0

$ python scripts/memory_ab_demo.py
=> Memory saved 1 iteration(s). Run B is a DIFFERENT session that read lessons run A wrote.
```

### Open items carried forward

- **Lessons are never retired.** `hits` counts reinforcement but there is no counter-evidence
  signal. `score_delta` is already stored, so demoting lessons that stop paying off is a scoring
  change, not a schema change.
- **`clear_session` deletes that session's lessons too**, since lessons are attributed to the
  session that found them. Defensible but surprising; a `--keep-lessons` flag is a one-liner.
- **KNN scans the whole vector table**, then scoping is applied in Python. Fine at hundreds to low
  thousands of records; past that it needs `sqlite-vec` partition keys.
- **The episodic tier earns less than it costs.** The ReAct scratchpad already tells the agent what
  it tried this session, so episodic recall is largely redundant within a run and useless across
  runs. If cutting scope I would keep lessons and profiles and make episodic in-memory only.
- **Recall quality is untested against a real corpus.** With two lessons per rubric, ranking barely
  matters; the blend weights are reasoned, not measured.
- M1 demo transcripts were **regenerated with `--no-memory`**, because the offline agent now
  explores when nothing is recalled. They show the pure M1 loop.
- Groq model id `openai/gpt-oss-120b` remains unverified against a live account (no key set yet).

---

## 2026-08-27 — Milestone 1: The Core Agentic Loop ✅

**Status:** complete · **Tasks:** M1-1 … M1-15 · **Tests:** 135 passing · **Lint:** clean

### Demo result (offline, `--provider mock`, no API key)

```
status: target_reached · iterations: 5 · trajectory: 28.7% -> 66.2% -> 96.2%
actions: score -> revise -> score -> revise -> score
```

Same code, different rubric: `--rubric config/rubrics/bug_report.yaml` → `target_reached`, 28.7% → 92.5%.

### What was built

| Area | Files |
|---|---|
| Domain model | `core/rubric.py` — `Rubric`, `Probe`, `ScoreCard`, weight validation, headroom maths |
| State | `core/state.py` — typed hand-offs; `LoopState.advance()` is the feedback edge |
| Tools | `tools/` — registry, rubric-derived schemas, 5 tools across 4 handler modules |
| The four steps | `core/{perceive,reason,act,reflect}.py` |
| The loop | `core/loop.py` |
| Prompts | `prompts/{reason,score,revise,reflect}.py` |
| Offline agent | `llm/demo_responder.py` |
| CLI + console | `cli.py`, `observability/render.py` |
| Memory seam | `memory/base.py` — ABC + `NullMemory` |
| Doc | `docs/01_patterns_research.md` |

### Decisions made (and why)

- **Perceive and Act use no LLM.** Only Reason and Reflect call a model. If Perceive prompted, it
  would be doing Reason's job with a different template and the "four cognitive steps" would be four
  prompts wearing hats. It also makes Reason reproducible: it sees an `Observation` and nothing else.
- **"Weakest" means most weighted headroom** (`weight × remaining points`), not lowest raw score.
  Ranking by raw score chases cheap points on low-weight criteria.
- **Tools take a *reference* to the working draft**, never the text as an argument. Passing the
  document through the model costs thousands of tokens per call and invites corruption in transit.
- **Tool schemas are generated from the rubric**, so criterion arguments carry an `enum` of real ids.
- **`thought` is a required property on every tool schema.** Several providers return empty content
  alongside a tool call; scraping the Thought from free text would silently reduce ReAct to plain
  action-selection whenever the provider did not cooperate.
- **Completion is decided by rules, not the model.** A `finalize` below target with the score still
  climbing is declined, the flag cleared, and the agent told why.
- **Plateau is measured between consecutive scorecards, not iterations.** A revision turn takes no
  measurement; per-iteration deltas would flag a plateau on every revise.
- **Runs return the best-scoring draft ever seen**, not the last one produced.
- **Two of the five tools never call an LLM.** An agent whose every tool is another prompt has no way
  to check itself.
- **A revision that changed nothing is a failure**, not a success.
- **The judge runs at temperature 0 and its schema orders `evidence → justification → score`**,
  because models fill JSON in schema order.

### Bug found and fixed during development

The registry stripped `thought` before dispatch but validated against the schema that still required
it, so **every** tool call failed. Two contracts were conflated: the schema the model must satisfy,
and the schema the handler's arguments are checked against. `dispatch_schema()` is now the explicit
conversion, with a test pinning the distinction.

The loop did not crash — it ran all six iterations, failed identically each time, and returned
cleanly. That is the error containment working, and a live preview of what the M3 stuck-detector
needs to catch.

---

## 2026-08-27 — Phase 0: Foundation ✅

**Status:** complete · **Tasks:** P0-1 … P0-17 · **Tests:** 52 passing · **Lint:** clean

### What was built

Scaffold and packaging; `.gitignore` / `.dockerignore` / `.env.example` / `.gitattributes`; typed
layered config (`config.py` + `config/config.yaml`); the LLM layer (`llm/{types,parsing,base,
openai_compatible,mock,factory}.py`); two rubric YAMLs; deliberately weak samples;
`scripts/preflight.py`.

### Decisions made (and why)

- **Raw `httpx`, not a vendor SDK.** This layer's job is mapping provider HTTP outcomes onto one
  error taxonomy; several SDKs would mean re-deriving that mapping from several exception trees.
- **Error taxonomy split by caller action**, not status code: `RetryableLLMError` /
  `TerminalLLMError` / `LLMParseError` / `ProviderUnavailableError`. Keeps the M3 retry decorator to
  a dozen lines and makes "back off vs fail over" a type-level decision.
- **`requires_key` declared explicitly per provider** rather than inferred from a localhost URL, so
  "why was this provider skipped?" is answerable from `config.yaml` alone.
- **Retryable HTTP statuses come from config**, proven by a test that flips 418 from terminal to
  retryable via config alone.
- **`MockProvider` accepts a stateful `responder`**, so the offline demo reacts to conversation state
  instead of replaying a script that ignores its input.
- **All M2/M3 config knobs declared up front**, before the code reading them existed. Forces
  "configurable without touching core loop code" to be designed in rather than bolted on.

---

## Template for future entries

```
## YYYY-MM-DD — <Milestone / Phase>: <title>  ✅ | 🚧 | ⛔
**Status:** … · **Tasks:** … · **Tests:** … · **Lint:** …
### Demo result
### What was built
### Decisions made (and why)
### Bug found and fixed during development
### Verification evidence
### Open items carried forward
```
