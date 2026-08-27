# Score Text Against a Rubric and Improve It

**An agentic loop with persistent memory and a production harness**

Hariish A · AI Systems Engineering challenge · Milestones 1–3

---

> **HOW TO USE THIS FILE — delete this block before rendering.**
>
> Your brief says: *"You may not use AI to write your solution PDF — it must reflect your own
> thinking."* So this file contains **no argument and no drafted prose**. What it does contain:
>
> * the **section structure** the brief asks for, in the order it asks for it;
> * every **table, figure and number**, already verified against the repository and laid out — these
>   are facts about your own code, not writing;
> * a `>` **prompt block** in each section listing exactly what you need to argue there, and roughly
>   how many words that is at the target page count.
>
> Replace every `>` prompt block with your own paragraphs. Keep the tables. Then run:
>
> ```
> python scripts/make_pdf.py docs/solution.md
> ```
>
> and Ctrl+P → Save as PDF. The script estimates page count and warns outside 4–8 pages.
> Budget: **~2,600–3,200 words of your prose** lands in the 6–7 page range with these tables.

---

## 1. The use case and why a loop fits it

> **Argue here (~450 words).** What the agent does, in two sentences. Then the point that decides
> everything else: a rubric score is a *real, external, numeric* observation, so the loop is
> verifiable — you can check whether iteration 3 actually beat iteration 2 instead of taking the
> model's word for it. Say why you chose a use case with that property. Then your significant design
> decisions from the table below — pick the three you most want to defend and say what you traded
> away. Mention that swapping the rubric swaps the domain with zero code change, and why you built it
> that way rather than hardcoding an essay grader.

**The loop.**

```
        ┌──────────────────── Reflection feeds forward ─────────────────────┐
        │                                                                   │
        ▼                                                                   │
  PERCEIVE ──▶ REASON ──▶ ACT ──▶ REFLECT ──▶ complete? ──no───────────────┘
  (no LLM)     (1 call)   (dispatch)  (rules + 1 call)   │
                                                        yes
                                                         ▼
                                            best draft seen + full trace
```

| Step | LLM? | Responsibility |
|---|---|---|
| **Perceive** | No | Load rubric, normalise draft, compute metrics and regex probes, recall memory, fold in last Reflection |
| **Reason** | Yes (1 call) | Choose exactly one next action, via forced tool use |
| **Act** | No | Validate arguments against the schema, dispatch, capture result or typed error |
| **Reflect** | Yes (1 call) + rules | Score delta, plateau check, self-critique, extract lesson, decide completion |

**Design decisions worth defending.**

| Decision | Alternative rejected |
|---|---|
| Perceive and Act make no LLM calls | Prompting all four steps |
| Tool schemas generated **from the rubric** (criterion args carry an `enum` of real ids) | Hand-written static schemas |
| "Weakest" = **most weighted headroom** (`weight × remaining`), not lowest raw score | Ranking by raw score |
| Tools take a *reference* to the working draft, never the text as an argument | Passing the draft through the model each call |
| Completion decided by **rules in Python**; `finalize` below target is declined | Letting the model declare itself done |
| Return the **best-scoring draft ever seen**, not the last one | Returning the final draft |
| One `httpx` client + one error taxonomy | Vendor SDKs |

**Scale.** 10,051 lines in `src/` (core 1,934 · harness 1,630); 3,920 lines of tests; 303 tests at
91% line coverage, running in ~9s with no key and no network. Two rubrics, five tools, three memory
tiers, seven injectable failure kinds. Running cost: **$0**.

---

## 2. Patterns applied, and why

> **Argue here (~550 words).** Which patterns you applied and — the part that carries the marks —
> *why they fit a rubric-scoring task specifically*. The strongest argument available to you: patterns
> split into those that need an external signal (ReAct, Reflexion, LATS) and those that substitute
> reasoning for one (CoT, ToT); you had a real cheap signal, so you used the first kind and were
> suspicious of the second. Then explain the LATS rejection in your own words — the three reasons are
> in the table; the reasoning is yours. Close on the shallow-ToT branch and why it defaults to off.

| Pattern | Used | Where, and the implementation detail that matters |
|---|---|---|
| **ReAct** | ✅ backbone | `LoopState.scratchpad` replays Thought/Action/Observation. `thought` is a **required property on every tool schema**, not scraped from free text — several providers return empty `content` alongside a tool call, which would silently reduce ReAct to plain action-selection |
| **Reflexion** | ✅ learning | Reflect's verbal critique produces a `lesson`, written to memory and replayed into later reasoning. The rubric percentage is the scalar evaluator the pattern assumes |
| **CoT** | ✅ narrowly | Judge schema orders `evidence → justification → score`, because models fill JSON in schema order — the argument is generated before the number. Judge runs at temperature 0.0 |
| **Tree of Thoughts** | ⚠️ one level, off by default | `revise_candidates > 1` generates N revisions at spread temperatures, judges each, keeps the winner. Real generator + evaluator + pruning; no backtracking, no lookahead |
| **LATS** | ❌ rejected | (1) The document is mutated in place — there is no cheap `reset()` for MCTS to backtrack to. (2) Branching factor is 1–2 sensible actions at any point. (3) Tens to hundreds of calls per problem, against a free tier |

---

## 3. How memory is structured, with a concrete example

> **Argue here (~550 words).** Why SQLite + `sqlite-vec` beat a vector database *for this project*
> (one file, one volume, one thing to delete; two stores can disagree silently). Why `fastembed` over
> `sentence-transformers`. Then the policy decision you should spend the most words on: **lessons are
> ranked but not relevance-gated, while episodic recall is** — say what that trades away and why you
> made that call. Finish by walking the reader through the A/B table below, and be explicit about why
> A ≡ C is the control that makes B mean anything.

**Three tiers, three lifetimes, one SQLite file** (rows + FTS5 keyword index + `sqlite-vec` vectors,
kept in sync by triggers; `BAAI/bge-small-en-v1.5`, 384-d, ONNX, no PyTorch).

| Tier | Scope | Lifetime |
|---|---|---|
| `episodic` | this session | one run |
| `lesson` | this **rubric**, any session | across sessions — the Reflexion payload |
| `profile` | this rubric | standing constraints |

Three required operations — `save`, `recall(query)`, `clear_session` — plus `list_sessions` and
`stats`, all on the `MemoryStore` ABC and all reachable from the CLI. Read at the start of **every**
Perceive; written after **every** Reflect.

**The A/B result.** Identical input, identical target, three arms:

| Run | Iterations | Action sequence |
|---|---|---|
| A · memory on, **cold** store | 6 | score → **analyze_text** → revise → score → revise → score |
| B · memory on, **warm** store, *different session* | **5** | score → **revise** → score → revise → score |
| C · memory **off** (control) | 6 | score → **analyze_text** → revise → score → revise → score |

All three reach `target_reached` at 96.2%; trajectory 28.7% → 66.2% → 96.2%.

**Iteration 2, verbatim.**

- **A (cold):** `analyze_text` — *"No prior experience with this rubric was recalled. Measure the
  draft directly before spending a revision on a guess."*
- **B (warm):** `revise_text` — *"Memory says: On this rubric, targeting the two highest-weighted
  criteria first moves the total faster than fixing the lowest raw score… Applying that directly
  rather than rediscovering it."*

The recalled record:

```
[lesson | session demo-a, iter 3 | relevance 0.43] On this rubric, targeting the two
highest-weighted criteria first moves the total faster than fixing the lowest raw score.
```

The effect reaches the run by two paths: a `RECALLED FROM MEMORY` block in the Reason prompt, and an
`apply_lessons` argument passed through to the reviser so it reaches the rewrite itself.

---

## 4. Failure modes the harness defends against

> **Argue here (~600 words).** Lead with the live evidence — 15 real 429s absorbed in one run — and
> what that told you that a mocked test could not. Then walk the four areas the brief names (retry,
> fallbacks, observability, guardrails) using the table, spending most words on the *classification*
> idea: the error taxonomy is split by what the caller should do, not by status code, which is why
> the retry decorator stays small and why "back off" and "fail over" are different code paths. Make
> the scope claim precisely: no retry/backoff/jitter/failover/budget/timeout logic anywhere in
> `core/`. Say what that discipline cost you.

**Live evidence.** One five-iteration run on Groq's free tier absorbed **HTTP 429 fifteen times**.
Every `Retry-After` was honoured — 139 s of backoff inside 170 s of wall clock. The run completed:
`target_reached`, 15.0% → 32.5% → 87.5%, `retries=15 repairs=0 failovers=0`.

| Failure | Response |
|---|---|
| Rate limit / 5xx / timeout | Exponential backoff, **full jitter**; `Retry-After` beats computed delay, capped at 60 s. Separate policies for transport (4 attempts) and tools (2 attempts) |
| Unparseable model output | Forced tool schema → local JSON salvage → one repair call → safe read-only default action, marked `degraded` |
| Failed tool call | Returned as `ActionResult(ok=False)` — an observation the agent reacts to, never an exception |
| 400 vs 404 | A 400 is **raised**; a 404 **fails over**. Failover is sticky |
| Provider unavailable | Walk the chain `groq → ollama` |
| Iteration cap | Enforced in Python, never in a prompt; returns the best draft seen |
| Token budget | 80% warning, then graceful finalize |
| Stuck loop | Three signals: repeated `(action, args)`, draft A→B→A cycle, frozen score |
| Memory read failure | Circuit breaker → `NullMemory`, run continues `degraded_memory=true` |

**Observability.** `runs/<run_id>/trace.jsonl` (one envelope per event, flushed per line) plus
`summary.json`. Redaction happens in the formatter, matching both key names and value patterns.

**Seven injectable failure kinds**, each injected at the layer where the real thing occurs:
`rate_limit`, `server_error`, `bad_json`, `provider_down`, `tool_error`, `memory_down`, `budget`.

---

## 5. Honest reflections

> **Argue here (~700 words — the longest section).** The brief weights this like the others and it is
> the easiest to under-write. Write it from memory of actually doing the work, not from the list.
>
> The strongest thing you have is the **pattern across bugs 1, 2 and 3**: in all three, the error
> containment that kept the run alive is exactly what hid the bug, and all three were caught by tests
> asserting behaviour *changed* rather than by anything crashing. That is a genuine, non-obvious
> engineering lesson — make it the spine of this section.
>
> Then pick three or four limitations you actually care about and say what you would do instead.
> Do not list all twelve; depth beats coverage here.

**Bugs found and fixed.**

| # | Bug | Why it mattered |
|---|---|---|
| 1 | The registry stripped `thought` before dispatch but validated against the schema that still required it | *Every* tool call failed — and the loop did not crash. It ran all six iterations, failed identically each time, and returned cleanly |
| 2 | A zero-norm query vector made `sqlite-vec` return `NULL` cosine distance | The `TypeError` was swallowed by the manager's guard, the circuit breaker tripped, and memory **silently stopped working while the run reported nothing wrong** |
| 3 | The memory circuit breaker could never open | "Consecutive failures" was one shared counter, so successful writes reset a failing read's streak. A half-broken store failed forever while reporting itself healthy |
| 4 | Reflect's token usage was never counted | The budget guardrail would have been enforced against a number a third too low |
| 5 | A reasoning model can return HTTP 200 with an empty body | `gpt-oss-120b` spends output tokens on an internal `reasoning` field; a tight `max_tokens` returns a successful, empty completion — which would have let `revise_text` replace the draft with `""` |

**Limitations, honestly.**

- **The episodic tier earns less than it costs.** The ReAct scratchpad already tells the agent what
  it tried this session, so episodic recall is largely redundant within a run and useless across runs.
- **Lessons are never retired.** `hits` counts reinforcement; there is no counter-evidence signal.
- **The judge is the weak link in the whole design.** Every pattern here inherits its quality, and
  self-consistency (score *n* times, take the median) was cut on token cost.
- **Trivial input is not detected.** A three-word greeting is scored 0%, then rewritten into a
  170-word essay with 0.00 similarity to the input, and reported as `target_reached` at 85%.
- **The token budget is enforced between iterations, not within one.**
- **Docker has never been built locally** — CI builds it; this machine has not.

---

## Appendix — reproducing every claim

```bash
python -m pytest -q                          # 303 passed, 91% coverage, ~9s, no key
python scripts/memory_ab_demo.py             # the A/B table in §3
python -m agentic_rubric.cli --input samples/weak_essay.txt \
    --simulate-failure rate_limit --fail-step judge     # §4
python demo.py                               # browser UI
```

Transcripts: `docs/demos/`. Design documents: `docs/01_patterns_research.md`,
`02_memory_design.md`, `03_harness_design.md`.
