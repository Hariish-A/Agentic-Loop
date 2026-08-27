# Progress Log

Companion to [Plan.md](Plan.md). Newest entry first.
If a session ends abruptly, read **▶ Resume here** at the top, diff it against `Plan.md`, and continue.

---

## ▶ Resume here

| | |
|---|---|
| **Last completed** | Milestone 4 — submission pass (M4-1 … M4-5, M4-7). CI, types, coverage, docs. |
| **Next task** | **M4-5 (you)** — write `docs/solution.md`. The brief forbids an AI-written solution PDF, so the prose is yours; `docs/solution_evidence.md` holds every verified number so you are not re-deriving them. Render with `python scripts/make_pdf.py docs/solution.md`. |
| **Then** | **M4-4** record the video from `docs/04_demo_script.md` · **M4-6** push to the private remote and add the reviewer accounts |
| **Blocked on** | Nothing technical. **Not pushed yet** — `gh` is not installed here, so the remote's visibility could not be verified, and the brief disqualifies a public repository. Confirm `github.com/Hariish-A/Agentic-Loop` is private, then `git push -u origin main --follow-tags`. |
| **Known gap** | Docker has still never been built locally (engine down). `docker compose config` validates and the CI `docker` job builds and runs it, but no image has been produced on this machine. |

**Verify the checkout is healthy before continuing:**

```bash
.venv/Scripts/python.exe -m pytest -q                      # expect: 303 passed
.venv/Scripts/python.exe -m pytest -q --cov=agentic_rubric # expect: 91%
.venv/Scripts/python.exe -m ruff check src tests scripts   # expect: All checks passed!
.venv/Scripts/python.exe -m mypy                           # expect: no issues in 53 files
.venv/Scripts/python.exe scripts/preflight.py --ping       # one real API call
.venv/Scripts/python.exe scripts/memory_ab_demo.py         # memory A/B, offline
docker compose config --quiet                              # compose is valid
```

---

## 2026-08-27 — Milestone 4: Submission pass ✅

**Status:** M4-1 … M4-3, M4-5 (tooling + evidence), M4-7 complete · M4-4 and M4-6 need you
**Tests:** 303 passing (85 new) · **Coverage:** 91% · **Lint:** clean · **Types:** clean

### What was built

| Task | Delivered |
|---|---|
| M4-1 | README: LLM choice and rationale as its own section, whole-system architecture diagram, a map of where each graded area lives, testing section with real numbers |
| M4-2 | `ruff` clean, `mypy` clean under `disallow_untyped_defs`, coverage 80% → **91%** |
| M4-3 | `.github/workflows/ci.yml` — five jobs, no secrets |
| M4-4 | `docs/04_demo_script.md` — shot-by-shot, timed to 4:15 of 5:00 |
| M4-5 | `docs/solution_evidence.md` + `scripts/make_pdf.py`. **The PDF prose is not written and will not be** — see below |
| M4-7 | Submission checklist run; two overclaims found and corrected |

### The solution PDF is deliberately not written

The brief says: *"You may not use AI to write your solution PDF — it must reflect your own
thinking."* So `docs/solution_evidence.md` contains the questions each required section must answer
and every verified fact, and **no drafted sentences**. `scripts/make_pdf.py` renders whatever gets
written to a print-ready page — browser Ctrl+P rather than shipping a GTK toolchain for WeasyPrint
or hand-laying paragraphs in ReportLab. It estimates page count and warns outside 4–8.

### Two overclaims, found and corrected

Both had been repeated across the README, `docs/03_harness_design.md` and this log.

1. **"`core/` contains no `try`/`except` at all."** False — there are seven. `core/loop.py` has
   exactly one, and it guards a memory *write*; the other six are single-clause degradations that
   predate the harness (memory read, two Reason parse fallbacks, Reflect's rule-based critique, and
   two rubric-YAML loaders). The true and still meaningful claim, now stated everywhere instead:
   **no retry, backoff, jitter, failover, provider-chain, budget or timeout logic exists anywhere
   in `core/`.** Verified by grep.
2. **"eleven lines in `core/loop.py`."** Invented — the actual M3 diff was 88 insertions. Measured
   properly: the seams cost **five lines inside `run()`** (a stop check at each boundary, and
   `act(...)` becoming `self._act(...)`), plus three short helpers and a `Protocol` declaration.

Worth recording because both were plausible, quotable and wrong, and both would have been read by a
reviewer as claims about the code rather than as slogans.

### Bugs and defects found during the pass

- **`diff_drafts` computed the reverse diff for nothing.** A second full `SequenceMatcher` pass over
  the whole document produced `reverse_similarity` — a value `difflib`'s symmetric ratio makes
  provably equal to one already in hand, consumed by nothing. Verified symmetric over 300 random
  pairs before removing it.
- **`--json` was unusable in a pipe.** The console transcript went to stdout alongside the JSON, so
  `... --json | jq` failed unless the caller also remembered `--quiet`. That is a trap, not an
  interface; the transcript now goes to stderr under `--json`.
- **`result` in `loop.run()` named both an `ActionResult` and the final `RunResult`.** It compiled,
  and made the last twenty lines read as though a tool call had become the run. Caught by mypy.
- **`float(entry.get("score"))` relied on `TypeError` to handle a missing score.** Correct at
  runtime, invisible to a reader; a missing score and an unparseable one are now different branches.
- **Two `set.add()`-inside-a-comprehension dedupe idioms.** Both worked, both flagged by mypy for
  using a `None`-returning call as a value. One became `dict.fromkeys`; the other, a case-insensitive
  dedupe, became an explicit loop.
- **mypy could not run at all.** `python_version = "3.10"` made it parse numpy's 3.12-syntax stubs as
  3.10 and abort before checking anything. The pin is gone; 3.10 compatibility is verified where it
  actually counts — the CI matrix runs the whole suite on it.

### Coverage: where it went and why

80% → **91%**. Two modules were at **0%**:

- **`cli.py`** (216 statements) — where every runtime decision is finally made, so a bug there is
  invisible to every other test and visible to every user on their first command. Now 91%, 30 tests.
- **`render.py`** (141 statements) — the *only* view of a run a reviewer watching the demo sees. A
  renderer that silently dropped harness events would make a run with fifteen absorbed rate limits
  look identical to one with none. Now 97%, 26 tests.

Plus five handler-contract tests for `analyze_text` and `diff_drafts` branches the happy path never
reaches (97%, up from 65%).

### CI design

Five jobs, **no secrets configured anywhere** — every test runs against `MockProvider`, so a fork,
an outside contributor's PR, and a day when the Groq free tier is exhausted all behave identically.

- `quality` — ruff + mypy
- `test` — the suite on Python **3.10 and 3.12**, `--cov-fail-under=88`
- `test-vector` — the same suite with `sqlite-vec` + `fastembed`, model cached
- `demo` — preflight, a full offline run asserting the trajectory climbs, the other rubric, the
  memory A/B, all seven injected failures, and the budget guardrail's exit code. Uploads traces.
- `docker` — builds the image and runs the loop inside it

`test` runs with `AGENTIC_MEMORY__EMBEDDER=none` on purpose. That is not a shortcut past the
embedding model: it exercises the FTS5/BM25 degradation path on **every commit**, which is the same
path the harness falls back to when the embedder is unavailable in production. `test-vector` covers
the other half. Every step was rehearsed locally before being written down.

### Verification evidence

```
$ pytest -q                                  303 passed in 8.9s
$ pytest --cov=agentic_rubric                TOTAL 3946 stmts, 372 miss, 91%
$ ruff check src tests scripts               All checks passed!
$ mypy                                       Success: no issues found in 53 source files
$ scripts/preflight.py --ping                [ ok ] groq:openai/gpt-oss-120b replied 'ready'
$ docker compose config --quiet              valid
                                             all internal doc links resolve
```

A/B numbers re-verified against the code at this commit, not quoted from memory:

```
demo-a  (cold)     6 iters  target_reached  96.2%  score -> analyze -> revise -> score -> revise -> score
demo-b  (warm)     5 iters  target_reached  96.2%  score -> revise -> score -> revise -> score
control (no mem)   6 iters  target_reached  96.2%  score -> analyze -> revise -> score -> revise -> score
```

### Open items carried forward

- **The solution PDF is yours to write.** `docs/solution_evidence.md`, then `scripts/make_pdf.py`.
- **The demo video is unrecorded.** `docs/04_demo_script.md` is the script.
- **Nothing has been pushed.** `gh` is not installed here, so the remote's visibility could not be
  checked, and a public repository is disqualifying. Confirm private, then push with
  `--follow-tags`, then add the reviewer accounts as collaborators.
- **Docker has never been built on this machine.** The CI `docker` job will be the first real build.
- **`web/server.py` is at 60% coverage** and still calls `AgenticLoop` directly rather than the
  `Runner`, so the browser demo displays no harness events. Switching it is a small change and would
  make the demo strictly better.
- **The `markdown` package is a new dev dependency**, used only by `scripts/make_pdf.py`.

---

## 2026-08-27 — Milestone 3: Harness Engineering ✅

**Status:** complete · **Tasks:** M3-1 … M3-13 · **Tests:** 226 passing at the tag (58 new here); 242 with the browser demo's tests in the working tree · **Lint:** clean

### Headline: the harness was proved without simulating anything

A live run against GroqCloud's free tier hit **fifteen genuine HTTP 429s** in five iterations and
completed anyway:

```
status      : target_reached
trajectory  : 15.0% -> 32.5% -> 87.5%      (+72.5 points)
elapsed     : 169.67s                       (~30s of work, 139s of honoured backoff)
harness     : provider=groq:openai/gpt-oss-120b (fallbacks: ollama)
              retries=15 repairs=0 failovers=0 tool_recoveries=0
budget      : 30,092 / 200,000 tokens (15%)
```

All fifteen `Retry-After` headers were honoured in preference to computed backoff. An earlier run of
the same command absorbed **16** (211s of backoff) and finished at 100.0%. Two runs, both completed
— reproducible, not a lucky sample. Without `harness/retry.py` either run dies on its second call.

Artifacts: `docs/demos/m3_live_groq_run.txt`, `m3_live_groq_trace.jsonl`, `m3_live_groq_summary.json`.

### The design decision everything else follows from

**No retry, backoff, failover, budget or timeout logic exists anywhere in `core/`.** The harness
attaches through two new seams on `AgenticLoop`, both defaulting to "no harness" --
**five lines inside `run()`** -- a stop check at each iteration boundary, and `act(...)` becoming `self._act(...)` -- plus three short helpers and a `Protocol` declaration:

| Collaborator | Default | What the runner substitutes |
|---|---|---|
| `provider` | an `LLMProvider` | `ResilientProvider` — retry, salvage, repair, sticky failover |
| `act_fn` | `core.act.act` | `ToolRecovery` — same signature, recovery ladder attached |
| `controller` | `None` | `Guardrails` — budget, clock, cap, stuck detection |
| `on_event` | renderer or nothing | tracer + renderer + logger, fanned out |

`LoopController` can *only stop the run*. It cannot choose an action, edit the draft or change a
score — a guardrail that could redirect the agent would be a fifth cognitive step hiding in the
harness. Two tests pin that: the loop still runs with no harness at all, and a controller that tries
to steer can only halt.

### What was built

| Area | Files | Notes |
|---|---|---|
| Retry | `harness/retry.py` | Backoff + full/equal jitter, `Retry-After` honoured and capped, separate tool policy |
| Fallbacks | `harness/fallbacks.py` | Provider chain, repair ladder, typed tool recovery |
| Guardrails | `harness/guardrails.py` | Budget (+80% warning), wall clock, iteration cap, ingestion cap |
| Stuck | `harness/loop_detect.py` | Repeated action, draft A→B→A cycle, frozen score |
| Injection | `harness/faults.py` | Seven failure kinds, each at the layer where the real thing occurs |
| Runner | `harness/runner.py` | Composes all of it; annotates the RunResult afterwards |
| Logging | `observability/logger.py` | JSON formatter, key- **and** pattern-based redaction |
| Tracing | `observability/trace.py` | `runs/<run_id>/trace.jsonl` + `summary.json`, one envelope per event |
| Console | `observability/render.py` | Harness events rendered inline with the step that provoked them |
| Container | `Dockerfile`, `docker-compose.yml`, `scripts/warm_models.py` | Model baked in at build time |
| Doc | `docs/03_harness_design.md` | Every decision paired with its failure mode |

### Decisions made (and why)

- **Two retry policies, not one.** Tools get 2 attempts and a 0.25s base; the transport gets 4 and
  1.0s. A failing tool is usually failing *deterministically* — bad arguments, an empty diff — and
  two of the five tools spend tokens on every attempt. The one extra attempt exists for the case
  that genuinely is transient: the reviser's own model call getting rate-limited.
- **`Retry-After` wins over computed backoff, but is capped at 60s.** The provider knows when its
  window resets. A header asking for twenty minutes should trigger failover, not a run that looks
  hung.
- **A 400 is raised; a 404 fails over.** 404 means "this model id does not exist here", and a
  deprecated model is exactly what a backup provider is for. Any other terminal error means *our*
  request is wrong, and failing over would burn the whole chain to reproduce our own bug — replacing
  a precise message with "every provider failed".
- **Failover is sticky.** An agent loop makes ~3 calls per iteration; re-probing a backend that just
  exhausted its retry budget would pay the full backoff on every one of them.
- **Exactly one repair call.** A model that cannot produce valid JSON twice will not manage it on
  the third try. Local salvage is tried first, and costs nothing.
- **Tool recovery is typed, not string-matched.** The registry contains every exception to keep the
  loop alive, which destroys the type — so it classifies at the point of containment
  (`ErrorKind.VALIDATION / UNKNOWN_TOOL / TRANSIENT / RECOVERABLE / TERMINAL`). Otherwise the
  harness would be pattern-matching on error strings.
- **Sanitising only ever *removes* arguments.** Guessing a replacement would put words in the
  agent's mouth and hide the mistake from the trace.
- **A substituted tool is always read-only.** A degraded decision is one we do not fully understand,
  and the wrong response to not understanding the situation is to start rewriting the user's text.
- **The "do nothing" branch still emits an event.** So that "the harness did nothing" and "the
  harness chose to do nothing" are distinguishable afterwards.
- **JSONL, flushed per line, not one JSON document.** A trace is most useful when the run *did not*
  finish, and a file that only becomes valid on its closing brace is useless in exactly that case.
- **Redaction lives in the formatter, not at call sites.** A rule that depends on every caller
  remembering it is not a rule. Key names *and* value patterns (`Bearer …`, `sk-…`, `gsk_…`) are
  both matched.
- **`cost_est` defaults to 0.0 and says so.** Every provider in the chain has a free path;
  fabricating a price would be worse than an honest zero. `cost_per_1k_*` are config knobs.
- **The loop's `run_end` omits the harness block.** It never learns that a retry happened, so
  reporting zeroes would be a confident lie. The runner emits `run_summary` with the real numbers.
- **Stuck detection counts *consecutive* repeats.** A healthy agent alternating score/revise calls
  `score_against_rubric()` with identical empty arguments many times; a total count would flag it.

### Bugs found and fixed during development

**1. The memory circuit breaker could never open.** `--simulate-failure memory_down` fails *reads
only* — a corrupt index or a locked reader, the realistic half-outage. The first run showed the
breaker never tripping. `MemoryManager` counted "consecutive failures" in **one shared integer
across every operation**, and since the loop writes after every Reflect, each successful write reset
the failing read's streak. The run would have paid for a failing read on every iteration, forever,
while reporting itself healthy. Fixed by counting per operation.

This is the **second** time in this project an error-containment mechanism has hidden a bug from
itself, and both were caught by a test asserting behaviour *changed* rather than "it did not raise".

**2. Reflect's token usage was never counted.** Wiring the budget guardrail required a running
total, which exposed that `RunResult` summed Reason's usage and the tools' but silently dropped
Reflect's — roughly a third of the spend. The budget would have been enforced against a number a
third too low. `test_token_accounting_includes_every_call` now counts against the provider's own
call log so the assertion cannot drift with the loop's shape.

**3. A reasoning model can return HTTP 200 with nothing in it.** `preflight --ping` reported Groq
replying `''`. `openai/gpt-oss-120b` spends output tokens on an internal `reasoning` field *before*
emitting content, so a `max_tokens` that looks generous returns a successful, empty response —
which would have let `revise_text` replace the user's draft with an empty string. Two fixes: the
client now raises `LLMParseError` when content is empty **and** `finish_reason=length`, naming
`llm.max_tokens` as the cure; and `llm.max_tokens` went from 2048 to 4096 with the reason written
next to it in the config. Neither was anticipated — both came from running it live.

### Verification evidence

```
$ .venv/Scripts/python.exe -m pytest -q     # at tag milestone-3
226 passed in 5.3s
    test_config 12   test_harness 58   test_llm_layer 42
    test_loop 30     test_memory 31    test_rubric 19   test_tools 34
    (test_harness and two new test_llm_layer cases are this milestone's. The
     working tree also holds test_web's 16, from the browser demo, which is
     not part of this commit -- 242 in the tree, 226 at the tag.)

$ .venv/Scripts/python.exe -m ruff check src tests scripts --exclude src/agentic_rubric/web
All checks passed!

$ .venv/Scripts/python.exe scripts/preflight.py --ping
[ ok ] groq:openai/gpt-oss-120b replied 'ready'    # model id confirmed against a live account
  tokens in/out : 87/16      latency : 722 ms

$ docker compose config --quiet                     # valid
```

All seven injected failures, each recovering (transcripts in `docs/demos/m3_failure_*.txt`):

```
rate_limit     retry 1 on mock after 0.05s (Retry-After)      -> target_reached
server_error   retry 1 on mock after 0.843s (backoff)         -> target_reached
bad_json       unusable reply: sent one repair prompt         -> target_reached
provider_down  failover mock-primary -> mock                  -> target_reached
tool_error     revise_text failed; backoff_retry              -> target_reached
memory_down    breaker opens after 3; "running without memory"-> target_reached
budget         [token_budget] 1,080 of 900 spent -> STOP      -> budget_exhausted, best draft kept
```

### Open items carried forward

- **`Dockerfile` and `docker-compose.yml` are unbuilt.** `docker compose config` validates, but
  Docker Desktop's engine was not running on this machine, so `docker build` has never executed.
  The layer ordering and the `warm_models.py` step are reasoned, not observed. **Build it before
  the submission.**
- **`mypy` is not installed**, so the M3 code is unchecked against the project's
  `disallow_untyped_defs`. M4-2.
- **The browser demo (`src/agentic_rubric/web/`) still calls `AgenticLoop` directly.** It was added
  alongside this work and predates the `Runner`, so it shows no retry, repair or guardrail events.
  Switching it to `Runner` is a small change and would make the demo strictly better.
- **The stuck detector's `score_plateau` signal barely earns its place.** Reflect's own
  `min_improvement` rule fires first in every realistic configuration. Kept because the two answer
  to different config knobs, but it is the first thing I would cut.
- **The token budget is enforced between iterations, not within one.** An iteration that revises a
  20,000-character draft can overshoot by a few thousand tokens. Enforcing mid-iteration means
  threading the guardrail into the tool context, putting budget logic inside tool handlers — a worse
  trade than a bounded overshoot.
- **Wall-clock is checked at iteration boundaries only.** One hung call can exceed the limit by the
  provider timeout (90s on Groq). The real fix is a deadline threaded into the `httpx` client, which
  would give the timeout two owners.
- **`cost_est` is per-event and priced at the *active* provider**, so a run that fails over mid-way
  prices its early events at the new rate. The `summary.json` total is computed once at the end and
  is correct; the per-event column is indicative.
- **Repair could be smarter and is not.** It states the error and re-asks. It does not narrow the
  tool set, lower the temperature, or reduce `max_tokens` to make truncation less likely. All three
  are cheap; I had no evidence to choose between them and did not want three unmeasured knobs.

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
