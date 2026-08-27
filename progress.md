# Progress Log

Companion to [Plan.md](Plan.md). Newest entry first.
If a session ends abruptly, read **▶ Resume here** at the top, diff it against `Plan.md`, and continue.

---

## ▶ Resume here

| | |
|---|---|
| **Last completed** | Milestone 1 — The Core Agentic Loop (M1-1 … M1-15) |
| **Next task** | **M2-3** — `memory/sqlite_store.py`: SQLite schema, migrations, FTS5 keyword recall |
| **Already done from M2** | **M2-1** and **M2-2** were built early — `memory/base.py` has the `MemoryStore` ABC (`save` / `recall` / `clear_session` / `list_sessions` / `stats`), `MemoryRecord`, and `NullMemory`. The loop already reads memory in Perceive and writes episodic + lesson records after Reflect. |
| **Then** | M2-4 (sqlite-vec + fastembed) → M2-5 (manager/scoping) → M2-6 (swap `cli.build_memory`) → M2-7 (A/B demo) → M2-8 (doc) → M2-9 (tests) |
| **Blocked on** | Nothing. `.env` has placeholders; add a `GEMINI_API_KEY` for live runs. The mock path needs no key. |

**First action for M2:** install the deferred deps — `pip install sqlite-vec fastembed` — then implement
`memory/sqlite_store.py`. The only wiring change needed afterwards is
[`cli.build_memory`](src/agentic_rubric/cli.py), which currently returns `NullMemory()` in both branches
with a comment marking the swap point.

**Verify the checkout is healthy before continuing:**

```bash
.venv/Scripts/python.exe -m pytest -q                      # expect: 135 passed
.venv/Scripts/python.exe -m ruff check src tests scripts   # expect: All checks passed!
.venv/Scripts/python.exe scripts/preflight.py              # config + provider chain
PYTHONPATH=src .venv/Scripts/python.exe -m agentic_rubric.cli \
    --input samples/weak_essay.txt --provider mock         # full offline run
```

---

## 2026-08-27 — Milestone 1: The Core Agentic Loop ✅

**Status:** complete · **Tasks:** M1-1 … M1-15 · **Tests:** 135 passing (83 new) · **Lint:** clean

### Demo result (offline, `--provider mock`, no API key)

```
status     : target_reached
iterations : 5
trajectory : 28.7% -> 66.2% -> 96.2%
score      : 28.7% -> 96.2%  (+67.5 points)
actions    : score -> revise -> score -> revise -> score
```

Transcripts committed under [docs/demos/](docs/demos/): the essay run, the bug-report run (same code,
different rubric), and two injected-failure runs.

### What was built

| Area | Files | Notes |
|---|---|---|
| Domain model | `core/rubric.py` | `Rubric`, `Probe`, `ScoreCard`; weight validation; headroom maths |
| State | `core/state.py` | Typed hand-offs between the four steps; `LoopState.advance()` is the feedback edge |
| Tools | `tools/` (registry, definitions, text_stats, 4 handler modules) | 5 tools, schemas built **from the rubric** |
| The four steps | `core/{perceive,reason,act,reflect}.py` | Perceive and Act are model-free by design |
| The loop | `core/loop.py` | Wires the four; enforces the cap; tracks the best draft |
| Prompts | `prompts/{reason,score,revise,reflect}.py` | One module per LLM-touching step |
| Offline agent | `llm/demo_responder.py` | Stateful simulation for tests, offline demo and fault injection |
| CLI | `cli.py` | Input/rubric/provider/target/`--set` overrides/failure injection |
| Console | `observability/render.py` | Same event stream the M3 JSONL tracer will attach to |
| Memory seam | `memory/base.py` | ABC + `NullMemory`, so M2 is a swap not a rewrite |
| Docs | `docs/01_patterns_research.md` | ReAct, Reflexion, CoT, ToT, LATS + applied/rejected rationale |

### Decisions made (and why)

- **Perceive and Act use no LLM.** Only Reason and Reflect call a model. If Perceive prompted, it
  would be doing Reason's job with a different template and the "four cognitive steps" would be four
  prompts wearing hats. It also makes Reason reproducible: it sees an `Observation` and nothing else.
- **"Weakest criterion" means most weighted headroom, not lowest raw score.** `weight × remaining
  points`. Ranking by raw score sends the agent after cheap points on a low-weight criterion; ranking
  by headroom sends it to the highest-value edit available. There is a test for exactly this
  (`test_headroom_is_weight_times_remaining_not_raw_score`).
- **Tools take a *reference* to the working draft, never the text as an argument.** Passing the draft
  through the model would cost thousands of tokens per call and give it a standing opportunity to
  corrupt the document in transit.
- **Tool schemas are generated from the rubric.** Every criterion argument carries an `enum` of that
  rubric's real ids, so "improve the vibes criterion" is structurally impossible rather than caught
  by validation. Same code produces a correct, different tool set for the bug-report rubric.
- **`thought` is a required property on every tool schema.** Several providers return empty content
  alongside a tool call; scraping the Thought from free text would silently reduce ReAct to plain
  action-selection whenever the provider did not cooperate.
- **Completion is decided by rules, not by the model.** `finalize` is advisory: a request made below
  target while the score is still climbing is declined, the flag is cleared, and the agent is told
  why. The model's own opinion is recorded as `model_votes_done` and only ever corroborates the
  deterministic plateau rule.
- **Plateau is measured between consecutive *scorecards*, not consecutive iterations.** A revision
  turn produces no measurement; treating every iteration as a data point would flag a plateau on
  every revision and stop the loop after two turns.
- **The run returns the best-scoring draft ever seen, not the last one.** A revision can make things
  worse. An improvement agent that hands back text worse than its input has failed at its one job.
- **Two of the five tools never call an LLM.** An agent whose every tool is another prompt has no way
  to check itself. `analyze_text` and `diff_drafts` are the only things in the loop that can
  contradict the agent's self-report.
- **A revision that changed nothing is a failure, not a success.** Without the similarity guard the
  agent can loop forever reporting work it did not do.
- **The judge runs at temperature 0.0, and its schema orders `evidence → justification → score`.**
  Models fill JSON in schema order, so the argument is generated before the number. The loop steers
  on the *difference* between two scores; a judge that drifts is indistinguishable from progress.
- **Every tool failure is an `ActionResult(ok=False)`, never an exception.** A broken tool call is
  data the agent reacts to. Verified by the injected-failure demos: a 429 on the judge is retried by
  the agent on the next iteration and the run still reaches target.

### Bug found and fixed during development

The registry stripped `thought` from the arguments before dispatch, then validated against the
unmodified schema — which still listed `thought` as required. **Every** tool call failed with
"missing required argument 'thought'". Two contracts were being conflated: the schema the model is
asked to satisfy, and the schema the handler's arguments are checked against. `dispatch_schema()` is
now the explicit conversion between them, with a test pinning the distinction.

Worth noting how it surfaced: the loop did not crash. It ran all six iterations, failed the same way
each time, and returned cleanly — which is the error containment working, and also a live preview of
exactly what the M3 stuck-detector needs to catch.

### Verification evidence

```
$ .venv/Scripts/python.exe -m pytest -q
135 passed
    tests/test_config.py    12      tests/test_rubric.py   19
    tests/test_llm_layer.py 40      tests/test_tools.py    34
    tests/test_loop.py      30

$ .venv/Scripts/python.exe -m ruff check src tests scripts
All checks passed!

$ python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock
target_reached in 5 iterations, 28.7% -> 96.2%

$ python -m agentic_rubric.cli --input samples/weak_bug_report.txt \
      --rubric config/rubrics/bug_report.yaml --provider mock
target_reached in 5 iterations, 28.7% -> 92.5%   # zero code change

$ python -m agentic_rubric.cli ... --simulate-failure rate_limit --fail-step judge
iteration 1 tool FAILED (429), agent re-scored on iteration 2, run reached target

$ python -m agentic_rubric.cli ... --simulate-failure bad_json --fail-step reason
REASON -> score_against_rubric [DEGRADED FALLBACK], run completed normally
```

### Open items carried forward

- `sqlite-vec` and `fastembed` are declared in `requirements.txt` but still **not installed** —
  install at the start of M2.
- The **shallow Tree-of-Thoughts branch** (`loop.revise_candidates > 1`) is implemented and unit
  tested for correct selection, but never benchmarked against `N=1` on real text. Documented as
  unverified in the patterns doc rather than claimed as a quality win.
- **The judge is the weak link** in the whole design. Self-consistency (score *n* times, take the
  median) would reduce its variance and was left out purely on token cost. First thing to add with
  more budget.
- The ReAct scratchpad is **truncated to the last four steps** to bound prompt growth; on long runs
  the agent can forget an early failed strategy.
- `cli.build_memory` currently returns `NullMemory()` on both branches. This is the single wiring
  point M2 needs to change.
- Model ids in `config.yaml` remain unverified against a live account (no key set yet).

---

## 2026-08-27 — Phase 0: Foundation ✅

**Status:** complete · **Tasks:** P0-1 … P0-17 · **Tests:** 52 passing · **Lint:** clean

### What was built

| Area | Files | Notes |
|---|---|---|
| Scaffold | `src/agentic_rubric/{core,llm,tools,memory,harness,observability,prompts}/` | src layout, git repo on `main` |
| Packaging | `pyproject.toml`, `requirements*.txt` | ruff + mypy + pytest configured |
| Hygiene | `.gitignore`, `.dockerignore`, `.env.example`, `.gitattributes` | `.env`, `data/*.db`, `runs/*` never committed |
| Config | `config/config.yaml`, `config.py` | frozen dataclasses; CLI > env > YAML > defaults; unknown keys rejected |
| LLM layer | `llm/{types,parsing,base,openai_compatible,mock,factory}.py` | one httpx client for all OpenAI-compatible providers |
| Rubrics | `config/rubrics/*.yaml` | 5 weighted criteria each, weights verified to sum to 1.00 |
| Samples | `samples/weak_*.txt` | deliberately weak, so iterations have headroom |
| Preflight | `scripts/preflight.py` | config validation + provider availability + optional live ping |

### Decisions made (and why)

- **Raw `httpx`, not a vendor SDK.** This layer's job is mapping provider HTTP outcomes onto one
  error taxonomy; four SDKs would mean re-deriving that mapping from four exception trees.
- **Error taxonomy split by caller action**, not status code: `RetryableLLMError` /
  `TerminalLLMError` / `LLMParseError` / `ProviderUnavailableError`. Keeps the M3 retry decorator to
  a dozen lines and makes "back off vs fail over" a type-level decision.
- **`requires_key` declared explicitly per provider** rather than inferred from a localhost URL, so
  "why was this provider skipped?" is answerable from `config.yaml` alone.
- **Retryable HTTP statuses come from config**, proven by a test that flips 418 from terminal to
  retryable via config alone.
- **`MockProvider` accepts a stateful `responder`**, not just a fixed tape — so the offline demo can
  react to conversation state instead of replaying a script that ignores its input.
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
### Verification evidence
### Open items carried forward
```
