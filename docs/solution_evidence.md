# Evidence pack for the Solution PDF

> **What this file is.** Every verified number, file reference and transcript you might need while
> writing the Solution PDF, gathered in one place so you are not hunting through six documents.
>
> **What this file is deliberately not.** It contains no drafted paragraphs, no framing and no
> argument. The brief is explicit:
>
> > *AI Usage: You may use AI tools to help write and debug code. You may **not** use AI to write
> > your solution PDF — it must reflect your own thinking.*
>
> So each section below gives you (a) the questions the brief asks that section to answer, and
> (b) the raw facts, verified against the repository. **The thinking, the framing, the argument and
> every sentence are yours.** Do not paste from here — the numbers are here so you do not have to
> re-derive them, not so the prose writes itself.
>
> Render whatever you write with `python scripts/make_pdf.py docs/solution.md`, then Ctrl+P →
> Save as PDF. The script warns if you are outside the 4–8 page range.

---

## Section 1 — Use case and design rationale

**Questions to answer:** What is the use case? Why is it a good fit for an agentic loop? What were
the significant design decisions and what were the alternatives?

**Verified facts**

- Use case: *score text against a rubric and improve it*.
- Two shipped rubrics, both 5 weighted criteria: `config/rubrics/essay_argumentative.yaml`,
  `config/rubrics/bug_report.yaml`. Swapping them swaps the domain with **zero code change** —
  the tool schemas are generated from the rubric, so criterion arguments carry an `enum` of that
  rubric's real ids.
- Rubric YAML declares: weighted criteria, level descriptors, improvement hints, and deterministic
  regex `probes`.
- Perceive and Act make **no LLM calls**. Only Reason (1 call) and Reflect (1 call) do; the judge
  and reviser are LLM-backed *tools* invoked from Act.
- 5 tools: `score_against_rubric`, `revise_text`, `analyze_text`, `diff_drafts`, `finalize`.
  **Two of the five never call a model** (`analyze_text`, `diff_drafts`).
- "Weakest criterion" is defined as **most weighted headroom** (`weight × remaining points`), not
  lowest raw score. See `core/rubric.py`.
- Tools take a *reference* to the working draft, never the text as an argument (`core/state.py`,
  `Workspace`).
- Completion is decided by **rules in Python**, not by the model. A `finalize` below target with the
  score still climbing is declined and the flag cleared (`core/reflect.py::_resolve_completion`).
  The model's vote is recorded as `model_votes_done` and only ever corroborates the plateau rule.
- Plateau is measured **between consecutive scorecards**, not between iterations.
- A run returns the **best-scoring draft ever seen**, not the last one (`LoopState._track_best`).
- Codebase: 10,051 lines in `src/`, of which `core/` is 1,934 and `harness/` is 1,630.
  Tests: 3,920 lines.

**Alternatives you actually rejected** (so you can discuss trade-offs honestly)

| Rejected | In favour of | Where it is recorded |
|---|---|---|
| Vendor SDKs (OpenAI/Anthropic/Google) | one `httpx` client + one error taxonomy | README "LLM choice", `llm/openai_compatible.py` docstring |
| A vector database (Chroma/Qdrant/FAISS) | SQLite + `sqlite-vec` in one file | `docs/02_memory_design.md` |
| `sentence-transformers` | `fastembed` (ONNX, no PyTorch) | image size ~400 MB vs ~2.5 GB |
| LATS | ReAct + Reflexion + shallow ToT | `docs/01_patterns_research.md` |
| Prompting Perceive/Act | keeping them LLM-free | `core/perceive.py` docstring |

---

## Section 2 — Patterns applied, and why

**Questions to answer:** Which of ReAct, Reflexion, CoT, Tree of Thoughts and LATS did you apply?
Why do those fit *this* use case? Why not the others?

**Source document:** `docs/01_patterns_research.md` — already contains mechanism, loop shape, cost
profile, failure modes and paper citation for all five. Re-read it; do not re-derive it.

**Verified facts about what is actually implemented**

- **ReAct** — the Thought/Action/Observation scratchpad is `LoopState.scratchpad`
  (`core/state.py::ReactStep`), rendered into the Reason prompt.
- `thought` is a **required property on every tool schema** (`tools/registry.py::THOUGHT_PROPERTY`),
  not scraped from free text. Reason for this: several providers return empty `content` alongside a
  tool call, which would silently reduce ReAct to plain action-selection.
- **Reflexion** — the verbal self-critique in `core/reflect.py` produces a `lesson`, which is
  written to memory and replayed into later reasoning. This is the mechanism the A/B in Section 3
  measures.
- **Chain of Thought** — the judge's schema orders `evidence → justification → score`
  (`prompts/score.py`), because models fill JSON in schema order. Judge runs at
  `JUDGE_TEMPERATURE = 0.0`.
- **Tree of Thoughts (shallow)** — `loop.revise_candidates > 1` generates N candidate revisions at
  spread temperatures (`CANDIDATE_TEMPERATURES = (0.4, 0.7, 0.9, 1.0)`), scores each with the judge,
  keeps the winner. Default is 1, i.e. **off**.
- **LATS** — explicitly rejected, with reasoning, in `docs/01_patterns_research.md`.

---

## Section 3 — Memory structure, with a concrete example

**Questions to answer:** What backend and why? How is memory structured? Show a concrete example of
memory changing the agent's output.

**Source document:** `docs/02_memory_design.md`.

**Verified facts**

- Backend: **SQLite + `sqlite-vec` + `fastembed`** — records, FTS5 keyword index and vectors all in
  **one file**, kept in sync by triggers. Embedding model `BAAI/bge-small-en-v1.5`, 384 dimensions,
  ONNX, CPU-only, no PyTorch.
- Three tiers, three lifetimes:

  | Tier | Scope | Lifetime |
  |---|---|---|
  | `episodic` | this session | one run |
  | `lesson` | this **rubric**, any session | across sessions — the Reflexion payload |
  | `profile` | this rubric | standing constraints |

- Three required operations: `save`, `recall(query)`, `clear_session` — plus `list_sessions` and
  `stats`. All on the `MemoryStore` ABC (`memory/base.py`). Reachable from the CLI:
  `--memory-stats`, `--clear-session <id>`, `--no-memory`.
- Read at the start of **every** Perceive; written after **every** Reflect.
- **Lessons are ranked but NOT relevance-gated; episodic recall is.** Capped at
  `max_lessons_per_recall: 3`. This is the most consequential policy call in `memory/manager.py`.
- The recall query describes **the problem, not the text** — built from the rubric and the criteria
  with the most headroom. A test asserts the draft's own words never reach the query.
- Keyword-channel scores are **ordinal, not calibrated** (FTS5 BM25 magnitudes depend on corpus
  size). In keyword-only mode the relevance gate is *skipped* rather than applied to a fake score.
- A relearned lesson increments `hits` instead of duplicating; episodic records never deduplicate.

**The concrete example — verified, reproducible**

```bash
rm -f data/memory.db
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --session demo-a
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --session demo-b
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --no-memory
```

| Run | Iterations | Action sequence |
|---|---|---|
| A · memory on, **cold** store | 6 | score → **analyze_text** → revise → score → revise → score |
| B · memory on, **warm** store, *different session* | 5 | score → **revise** → score → revise → score |
| C · memory **off** (control) | 6 | score → **analyze_text** → revise → score → revise → score |

All three reach `target_reached` at 96.2%; trajectory 28.7% → 66.2% → 96.2%.

Iteration 2, verbatim from the transcripts:

- **A (cold):** `analyze_text` — *"No prior experience with this rubric was recalled. Measure the
  draft directly before spending a revision on a guess."*
- **B (warm):** `revise_text` — *"Memory says: On this rubric, targeting the two highest-weighted
  criteria first moves the total faster than fixing the lowest raw score... Applying that directly
  to Thesis and Position rather than rediscovering it."*

The recalled record itself:

> `[lesson | session demo-a, iter 3 | relevance 0.43] On this rubric, targeting the two
> highest-weighted criteria first moves the total faster than fixing the lowest raw score.`

**Why A ≡ C matters** (worth making explicit in your write-up): a cold store carries the same
information as no store, so they must behave identically — and they do, action for action. Without
that control, "memory helped" could just mean "the second run of anything is faster."

**How the effect is transmitted** (two mechanisms, both visible in the transcript): the lesson
appears in the Reason prompt under a `RECALLED FROM MEMORY` block, and it is passed through to the
reviser as an `apply_lessons` argument so it reaches the rewrite itself.

Transcripts: `docs/demos/m2_memory_ab.txt`, `m2_cold_session.txt`, `m2_warm_session.txt`.

---

## Section 4 — Failure modes the harness defends against

**Questions to answer:** What can go wrong, and what does the harness do about each?

**Source document:** `docs/03_harness_design.md` — has the full ladder table and per-decision
rationale.

**Verified facts**

- **The strongest single piece of evidence:** a live run on Groq's free tier returned **HTTP 429
  fifteen times** in one five-iteration run. Every `Retry-After` was honoured (1s to 24s, 139s of
  backoff out of 170s wall clock). The run completed: `target_reached`, 15.0% → 32.5% → 87.5%,
  `retries=15 repairs=0 failovers=0`. An earlier run absorbed **16** and finished at 100.0%.
  Artifacts: `docs/demos/m3_live_groq_run.txt`, `m3_live_groq_trace.jsonl`, `m3_live_groq_summary.json`.
- Retry uses exponential backoff with **full** jitter by default (`equal` and `none` also
  available); `Retry-After` beats computed backoff but is capped at 60s.
- **Separate policies** for transport (4 attempts, 1.0s base) and tools (2 attempts, 0.25s base).
- Not retried: `TerminalLLMError`, `ProviderUnavailableError`, `LLMParseError` — each has a
  different answer instead.
- A **400 is raised; a 404 fails over.** Failover is **sticky**.
- Seven injectable failure kinds, each injected at the layer where the real thing occurs:
  `rate_limit`, `server_error`, `bad_json`, `provider_down`, `tool_error`, `memory_down`, `budget`.
  Transcripts for all of them in `docs/demos/m3_failure_*.txt`.
- Guardrails: hard iteration cap (in Python, never in a prompt), token budget with an 80% warning,
  wall-clock deadline, ingestion cap. **Every stop is graceful** — status, reason, best draft, full
  trace; never an exception.
- Stuck detection, three signals: repeated `(action, args)` *consecutively*, draft A→B→A cycle,
  frozen score across a window.
- Observability: `runs/<run_id>/trace.jsonl` (one envelope per event, flushed per line) +
  `summary.json`. Redaction happens **in the formatter**, matching both key names and value patterns.
- The precise scope claim: **no retry, backoff, jitter, failover, provider-chain, budget or timeout
  logic anywhere in `core/`.** `core/` does contain seven `except` clauses — one in `loop.py`
  (guarding a memory *write*), and six single-clause degradations that predate the harness. The
  table is in `docs/03_harness_design.md`.

---

## Section 5 — Honest reflections: what didn't work, what you'd do differently

**Questions to answer:** What went wrong? What would you change?

> This section carries disproportionate weight — the brief says *"we are looking for clear thinking,
> honest engineering decisions"*. It is also the section where pasted text would be most obvious.
> **Write it from memory of actually doing the work.**

**Bugs that were found and fixed** (all recorded in `progress.md` with dates)

1. **The registry stripped `thought` before dispatch but validated against the schema that still
   required it** — so *every* tool call failed. The loop did not crash: it ran all six iterations,
   failed identically each time, and returned cleanly. Two contracts had been conflated.
   *(Milestone 1)*
2. **A zero-norm query vector made `sqlite-vec` return `NULL` for cosine distance.** The resulting
   `TypeError` was swallowed by the manager's guard, the circuit breaker tripped, and **memory
   silently stopped working while the run reported nothing wrong**. *(Milestone 2)*
3. **The memory circuit breaker could never open.** "Consecutive failures" was one shared counter
   across all operations, so successful writes reset the failing read's streak. With a half-broken
   store (reads fail, writes work) the breaker never fired and every iteration paid for a failing
   read, forever, while reporting itself healthy. *(Milestone 3, found by `--simulate-failure
   memory_down`)*
4. **Reflect's token usage was never counted** toward the run total, so the budget guardrail would
   have been enforced against a number roughly a third too low. *(Milestone 3)*
5. **A reasoning model can return HTTP 200 with an empty body.** `gpt-oss-120b` spends output tokens
   on an internal `reasoning` field before emitting content; a tight `max_tokens` returns a
   successful, empty completion — which would have let `revise_text` replace the user's draft with
   `""`. *(Milestone 3, found by running `preflight --ping` live)*
6. **`diff_drafts` ran a second full `SequenceMatcher` pass** over the whole document to produce a
   `reverse_similarity` value that is provably identical to one it already had, consumed by nothing.
   *(Milestone 4)*

**A pattern worth noticing across 1, 2 and 3:** the error containment that kept each run alive is
exactly what hid the bug. All three were caught by tests asserting that behaviour **changed**, not
by anything crashing.

**Known limitations, already documented** (`progress.md` "Open items carried forward")

- **The episodic tier earns less than it costs.** The ReAct scratchpad already tells the agent what
  it tried this session, so episodic recall is largely redundant within a run and useless across
  runs. If cutting scope, keep lessons and profiles; make episodic in-memory only.
- **Lessons are never retired.** `hits` counts reinforcement but there is no counter-evidence signal.
- **`clear_session` deletes that session's lessons too.** Defensible but surprising.
- **KNN scans the whole vector table**, then scopes in Python. Fine at hundreds of records.
- **Recall quality is untested against a real corpus.** With two lessons per rubric, ranking barely
  matters; the blend weights are reasoned, not measured.
- **The stuck detector's `score_plateau` signal barely earns its place** — Reflect's own rule fires
  first in every realistic configuration.
- **The token budget is enforced between iterations, not within one.**
- **Wall-clock is checked at iteration boundaries only** — one hung call can overshoot by the
  provider timeout.
- **`cost_est` is per-event and priced at the *active* provider**, so a run that fails over mid-way
  prices its early events at the new rate.
- **Repair could be smarter and is not** — it states the error and re-asks; it does not narrow the
  tool set, lower the temperature, or reduce `max_tokens`.
- **The judge is a simulation in every offline demo.** `ScriptedAgentResponder` produces scores by a
  rule, not by reading the text. It faithfully reproduces the *shape* of a run, and it says so.
- **The browser demo calls `AgenticLoop` directly**, not the `Runner`, so it displays no harness
  events.
- **Docker has never been built locally** — `docker compose config` validates and the CI `docker`
  job builds it, but no image has been produced on this machine.

---

## Numbers you may want, in one block

| Fact | Value |
|---|---|
| Tests / coverage | 303 passing, 91% line coverage, ~9s, no key or network |
| Lint / types | `ruff` zero findings; `mypy` clean under `disallow_untyped_defs` |
| Source size | 10,051 lines in `src/` (core 1,934 · harness 1,630); 3,920 lines of tests |
| Tools | 5, of which 2 never call an LLM |
| Memory tiers | 3 (episodic, lesson, profile) in 1 SQLite file |
| Rubrics shipped | 2, both 5 weighted criteria, swappable with no code change |
| Failure kinds injectable | 7 |
| Live run | 15 real 429s absorbed; 15.0% → 87.5%; 170s (139s backoff) |
| Offline A/B | cold 6 iters · warm 5 iters · no-memory 6 iters (control) |
| Running cost | $0 |
| Python | 3.10 and 3.12, both in CI |

---

## Before you submit

- [ ] The PDF is 4–8 pages (`scripts/make_pdf.py` estimates and warns)
- [ ] Every sentence is yours
- [ ] Section 5 is substantial — it is weighted like the others and is the easiest to under-write
- [ ] No API key appears in any screenshot or code block
- [ ] Numbers in the PDF match this file (which matches the repository)
