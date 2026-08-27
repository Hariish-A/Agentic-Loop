# Agentic Rubric Loop

**Score text against a rubric, then improve it** — a hand-rolled Perceive → Reason → Act → Reflect
agentic loop with no agent framework in the core path.

The agent scores a draft against a weighted rubric, decides which criterion is the highest-value
lever, revises the text targeting that criterion, re-scores, and repeats until it hits the target
score, plateaus, or trips a guardrail.

[![CI](https://github.com/Hariish-A/Agentic-Loop/actions/workflows/ci.yml/badge.svg)](https://github.com/Hariish-A/Agentic-Loop/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-303%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)
![Cost](https://img.shields.io/badge/running%20cost-%240-brightgreen)

> **Status:** all four milestones complete. [Plan.md](Plan.md) is the task list;
> [progress.md](progress.md) is the running log of what was built, what broke, and why each
> decision was made.

---

## Why this use case

A rubric score is a real, external, numeric observation. That makes the loop *verifiable*: you can
see whether iteration 3 actually beat iteration 2, rather than taking the model's word for it. It
also makes Reflexion a natural fit — the scalar reward that pattern assumes is just sitting there.

### Where each evaluation area lives

| Area | Where to look | One-line claim |
|---|---|---|
| **Loop correctness** | [`core/`](src/agentic_rubric/core/) — one file per step | Perceive and Act use **no LLM at all**, so the four steps cannot collapse into four prompts. `LoopState.advance()` is the feedback edge |
| **Memory integration** | [`memory/`](src/agentic_rubric/memory/), [docs/02](docs/02_memory_design.md) | An A/B with a **cold-store control** proves a lesson written in session A changes session B's second decision |
| **Harness engineering** | [`harness/`](src/agentic_rubric/harness/), [docs/03](docs/03_harness_design.md) | **No retry, failover or budget logic anywhere in `core/`**; proved live by absorbing 15 real rate limits |
| **Patterns understanding** | [docs/01](docs/01_patterns_research.md) | ReAct + Reflexion + CoT applied, shallow ToT optional, **LATS explicitly rejected** with reasons |
| **Tool design** | [`tools/`](src/agentic_rubric/tools/) | Schemas generated **from the rubric**; two of five tools never call a model |
| **Code quality** | `pytest`, `ruff`, `mypy` | 303 tests, 91% coverage, zero lint findings, `disallow_untyped_defs` clean |

---

## Architecture

```
                        ┌──────────── HARNESS (harness/) ────────────┐
                        │  retry · fallbacks · guardrails · tracing   │
                        │  substituted for the loop's collaborators   │
                        └──┬───────────────┬──────────────┬───────────┘
                           │               │              │
      ┌────────────────────┼───────────────┼──────────────┼──────────────────────┐
      │  ┌──── Reflection feeds forward ───┼──────────────┼───────────────────┐  │
      │  │                                 │              │                   │  │
      │  ▼                                 ▼              ▼                   │  │
 in ──┼▶ PERCEIVE ──────▶ REASON ───────▶ ACT ───────▶ REFLECT ──▶ done? ──no──┘  │
      │  no LLM          1 LLM call      dispatch      rules            │         │
      │  metrics+probes  forced tools    only          + 1 LLM call    yes        │
      │     │                │             │              │             │         │
      └─────┼────────────────┼─────────────┼──────────────┼─────────────┼─────────┘
            │ recall         │ prompt      │ 5 tools      │ lesson      ▼
            ▼                ▼             ▼              ▼        best draft
       ┌─────────────────────────┐   ┌──────────────┐  ┌────────┐  + trace.jsonl
       │  MEMORY (one .db file)  │   │ score_rubric │  │ MEMORY │  + summary.json
       │  episodic │ lesson │    │◀──│ revise_text  │  │ write  │
       │  profile  │            │   │ analyze_text │  └────────┘
       │  sqlite-vec + FTS5     │   │ diff_drafts  │
       └─────────────────────────┘   │ finalize     │
                                     └──────────────┘
                                      2 of 5 use no LLM
```

Read it in three layers. The **middle row** is the loop the challenge asks for. **Below** it are the
two things the loop reaches for: memory (read in Perceive, written after Reflect) and the tool set
(dispatched by Act). **Above** it is the harness, which the loop never sees — it substitutes
decorated versions of the loop's own collaborators before the run starts.

| Step | LLM? | Responsibility |
|---|---|---|
| **Perceive** | No | Load rubric, normalise draft, compute deterministic metrics, recall memory, fold in last reflection |
| **Reason** | Yes | Choose exactly one next action with arguments, via forced tool-use |
| **Act** | No | Validate arguments against the schema, dispatch to a handler, capture result or typed error |
| **Reflect** | Yes + rules | Score delta, plateau check, self-critique, extract a reusable lesson, decide completion |

Making Perceive and Act LLM-free is deliberate: it keeps the four steps genuinely distinct rather
than four prompts wearing different hats.

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| LLM | Any OpenAI-compatible endpoint | One ~220-line httpx client covers Groq, Ollama, OpenAI, OpenRouter |
| Primary provider | **GroqCloud**, `openai/gpt-oss-120b` | Fast, cheap, supports tool calling and structured outputs |
| Fallback chain | `groq` → `ollama` → `mock` | Provider failover is itself a demoable failure mode |
| Memory | SQLite + `sqlite-vec` + `fastembed` | One file, no server, works offline; FTS5/BM25 auto-fallback |
| HTTP | `httpx` | Direct control over status→error mapping; retries owned by our harness |
| Config | YAML + env + CLI, typed dataclasses | Every runtime knob swappable without touching loop code |

Provider quirks are handled as **capability flags in config**, never by branching on a provider name.
Groq, for example, rejects `messages[].name` (declared as `supports_message_name: false`) and coerces
`temperature=0` to `1e-8`.

**Total running cost: $0.** Every dependency is OSS and every provider option has a free path.

### LLM choice and rationale

**GroqCloud running `openai/gpt-oss-120b`**, with local Ollama as the failover and a deterministic
mock for tests and offline demos.

*Why an OpenAI-compatible endpoint rather than a vendor SDK.* This project's LLM layer exists to map
provider HTTP outcomes onto **one** error taxonomy — retryable / terminal / parse / unavailable —
because that taxonomy is what keeps the retry decorator a dozen lines. Going through three vendor
SDKs would mean re-deriving that mapping from three different exception trees. One ~270-line `httpx`
client covers Groq, Ollama, OpenAI and OpenRouter, and swapping between them is a config edit.

*Why Groq specifically.*

- **A genuinely free tier that supports tool calling.** The whole loop rests on forced tool use; a
  provider without it would need prompt-and-parse, and half the reliability work would become
  parsing work.
- **Fast.** A rubric loop makes ~3 calls per iteration and 5–6 iterations per run. Latency compounds,
  and a demo that takes eight minutes is a demo nobody watches.
- **`gpt-oss-120b` is a large open-weights model.** If Groq disappears, the same model runs on
  Ollama, Together or a local box — the config already names the fallback. Betting the project on a
  proprietary model would make the provider chain decorative.

*What that choice actually cost, measured.* Groq's free tier rate-limits hard. A single live
five-iteration run returned **HTTP 429 fifteen times**, and the run took 170 seconds of which 139
were honoured `Retry-After` backoff. That is not a complaint — it is why
[`harness/retry.py`](src/agentic_rubric/harness/retry.py) exists and how it got tested for real
rather than only against injected faults.

*Two provider quirks, handled as config rather than as code.*

| Quirk | How it is handled |
|---|---|
| Groq rejects `messages[].name` | `supports_message_name: false` — the client strips the field |
| Groq converts `temperature=0` to `1e-8` | Documented in `config.yaml`; the judge is *near*-deterministic, not deterministic, and the loop steers on differences between two scores |
| `gpt-oss` spends output tokens on internal reasoning **before** emitting content | Found live: a tight `max_tokens` returns HTTP 200 with an empty body. The client now raises a typed parse error naming `llm.max_tokens`; the default was raised to 4096 |

The client never branches on a provider name. Every difference is a declared capability flag.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
# source .venv/bin/activate && pip install -r requirements-dev.txt  # macOS/Linux

cp .env.example .env          # then add GROQ_API_KEY (optional for the mock demo)
python scripts/preflight.py   # validates config, shows which providers are usable
python scripts/preflight.py --ping   # makes one real API call
```

Run the loop with no API key at all:

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock
```

Run the tests — no API key or network required:

```bash
python -m pytest -q                        # passes in seconds, no key needed
python -m ruff check src tests scripts
```

---

## Configuration

Everything lives in [`config/config.yaml`](config/config.yaml). Precedence, highest first:

```
CLI flags  >  environment variables  >  config.yaml  >  dataclass defaults
```

Environment overrides use `AGENTIC_<SECTION>__<FIELD>`, nesting arbitrarily deep:

```bash
AGENTIC_LLM__PRIMARY=ollama
AGENTIC_LOOP__MAX_ITERATIONS=8
AGENTIC_LLM__PROVIDERS__GROQ__MODEL=openai/gpt-oss-20b
AGENTIC_MEMORY__BACKEND=sqlite_fts
```

Or from the command line, for any key at all:

```bash
python -m agentic_rubric.cli --set loop.revise_candidates=3 --set memory.max_lessons_per_recall=5 ...
```

API keys are never stored in the config file — each provider names the variable it reads via
`api_key_env`, and the value comes from `.env`.

---

## Rubrics are data, not code

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt \
    --rubric config/rubrics/essay_argumentative.yaml

python -m agentic_rubric.cli --input samples/weak_bug_report.txt \
    --rubric config/rubrics/bug_report.yaml
```

Two shipped rubrics — an argumentative essay and an engineering bug report — swap the entire domain
with no code change. Each declares weighted criteria, level descriptors, improvement hints and
deterministic regex `probes` in YAML.

---

## Memory

Three tiers with three lifetimes, all in one SQLite file:

| Tier | Scope | Example |
|---|---|---|
| `episodic` | this session | `iteration 2: revise_text -> 202 -> 243 words, similarity 0.95` |
| `lesson` | this **rubric**, any session | `Unattributed figures score no better than no figures.` |
| `profile` | this rubric | standing constraints (tone, target, banned edits) |

Read at the start of every Perceive, written after every Reflect. Semantic recall via `sqlite-vec` +
`fastembed`, with FTS5/BM25 as the automatic fallback when the embedder is unavailable.

```bash
python scripts/memory_ab_demo.py        # proves memory changes behaviour, offline
python -m agentic_rubric.cli --memory-stats
python -m agentic_rubric.cli --clear-session <session-id>
python -m agentic_rubric.cli --no-memory ...    # the A/B control
```

The A/B result — identical input, identical target, three arms:

```
A  memory on, cold store     6 iters   score -> analyze_text -> revise -> score -> revise -> score
B  memory on, warm store     5 iters   score -> revise -> score -> revise -> score
C  memory off (control)      6 iters   score -> analyze_text -> revise -> score -> revise -> score
```

Run B is a *different session* reading lessons run A wrote. A ≡ C is the control that makes B
meaningful. Full write-up in [docs/02_memory_design.md](docs/02_memory_design.md).

---

## Browser demo

A single command, no install step, no web framework, no API key:

```bash
python demo.py            # opens http://127.0.0.1:8000
```

Built on `http.server` from the standard library. A demo whose first step is
"pip install a web framework" is a demo that fails on the reviewer's machine.

The page drives the real loop and streams every step as it happens:

| Panel | Shows |
|---|---|
| **Rubric** | Criteria, weights and probe counts, loaded from YAML — swap rubrics from the dropdown |
| **Live transcript** | One card per iteration with all four steps: PERCEIVE (score bar, metrics, failing probes, **recalled memory**), REASON (chosen tool + thought + arguments), ACT (result or typed error), REFLECT (delta, plateau, **lesson stored**) |
| **Summary** | Status, trajectory, action sequence, tokens, elapsed |
| **Scorecards** | Per-criterion score, weight, headroom, and the judge's quoted evidence |
| **Before / after** | Input vs the best-scoring draft ever seen |
| **Memory** | Store stats, what this run wrote, and every stored lesson |

### Reproducing each concept from the demo

| Concept | How to see it |
|---|---|
| The loop iterates | Run any sample — the trajectory climbs across three scorings |
| Rubrics are data | Switch to **Engineering Bug Report**; the tool schemas change with it |
| Headroom, not raw score | Scorecards tab — the agent targets weight × remaining points |
| Two LLM-free tools | Watch `analyze_text` and `diff_drafts` in the transcript |
| Shallow Tree of Thoughts | Set **Revision candidates** to 3; ACT reports "chose the best of 3 candidates" |
| Failure recovery | Inject `rate_limit` into `judge` — the tool fails, the agent re-scores next turn |
| Degraded reasoning fallback | Inject `bad_json` into `reason` — REASON shows a **degraded fallback** badge |
| **Cross-session memory** | Run with session `demo-a`, then change the session id to `demo-b` and run again |
| Memory control | Untick **Memory enabled** — the run reverts to the cold behaviour |
| The three memory ops | **Clear session** / **Wipe memory** buttons, plus the Memory tab's stats |

The memory demonstration in one sequence:

1. Press **Wipe memory**, then run with session `demo-a` → **6 iterations**, and
   iteration 2 spends a turn on `analyze_text` because nothing was recalled.
2. Change the session id to `demo-b` and run the same text again → **5 iterations**.
   Iteration 2 now recalls a lesson written by `demo-a` (highlighted in purple) and
   goes straight to `revise_text`, passing the lesson through as `apply_lessons`.
3. Untick **Memory enabled** and run again → back to **6 iterations**, which is the
   control proving the difference came from memory and not from run order.

## Harness

Retry, fallbacks, guardrails and observability wrap the loop **without touching it**: the harness
substitutes decorated versions of the loop's own collaborators before it starts. **No retry,
backoff, jitter, failover, budget or timeout logic exists anywhere in `core/`** — `core/loop.py`
has exactly one `except`, and it guards a memory write.

| Area | Defence |
|---|---|
| **Retry** | Exponential backoff, full/equal jitter, `Retry-After` honoured (and capped); separate, tighter policy for tool calls |
| **Fallbacks** | One ladder per failure mode: forced schema → local JSON salvage → one repair call → safe default; sticky provider failover; typed tool recovery |
| **Guardrails** | Hard iteration cap in Python, token budget with an 80% warning, wall-clock deadline, ingestion cap |
| **Stuck detection** | Repeated `(action, args)`, draft A→B→A cycle, frozen score → `status=stuck` |
| **Observability** | `runs/<run_id>/trace.jsonl` + `summary.json`, one envelope per event; structured JSON logs with key- *and* pattern-based redaction |

Every stop is graceful: the run returns the **best draft seen**, a status, a reason and a complete
trace — never an exception.

### Proving it, on demand

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock \
    --simulate-failure rate_limit      # or server_error bad_json provider_down
                                       #    tool_error memory_down budget
```

Each kind is injected at the layer where the real thing occurs — a 429 inside the provider, a fault
inside a real tool handler, an outage inside the memory store — not by short-circuiting the harness
into pretending. Transcripts for all seven are in [docs/demos/](docs/demos/).

### And it was proved without simulation

A live run against Groq's free tier hit **fifteen genuine rate limits** and completed anyway:

```
status  : target_reached      trajectory : 15.0% -> 32.5% -> 87.5%
harness : retries=15 repairs=0 failovers=0 tool_recoveries=0
budget  : 30,092 / 200,000 tokens (15%)
```

Full write-up in [docs/03_harness_design.md](docs/03_harness_design.md).

---

## Testing and quality

```bash
pytest -q                                        # 303 tests, ~9s, no key, no network
pytest --cov=agentic_rubric --cov-report=term    # 91% line coverage
ruff check src tests scripts                     # zero findings
mypy                                             # disallow_untyped_defs, clean
```

| Suite | Tests | Covers |
|---|---|---|
| `test_harness` | 58 | Backoff bounds, jitter spread, each fallback rung, budget, stuck detection |
| `test_llm_layer` | 42 | Status→error mapping, JSON salvage, truncated completions |
| `test_tools` | 39 | Schema generation, validation, dispatch containment, handler contracts |
| `test_memory` | 31 | Round-trip, session isolation, `clear_session`, BM25 fallback, cross-session lessons |
| `test_loop` | 30 | Each step in isolation, the feedback edge, end-to-end with a rising score |
| `test_cli` | 30 | Override precedence, exit codes, memory commands, trace output |
| `test_render` | 26 | Every event the demo transcript shows, including harness recovery |
| `test_config` | 12 | Four-layer precedence, unknown-key rejection, type coercion |
| `test_rubric` | 19 | Weight validation, headroom maths, YAML loading |
| `test_web` | 16 | The browser demo's endpoints |

Every test runs against `MockProvider` with injected sleeps, so backoff bounds are asserted in
milliseconds rather than waited through. [CI](.github/workflows/ci.yml) runs lint, types, the suite
on Python 3.10 **and** 3.12, an end-to-end demo including all seven injected failures, and a Docker
build — with **no secrets configured**.

---

## Docker

```bash
docker compose run --rm preflight                                        # will it work here?
docker compose run --rm agent --input samples/weak_essay.txt --provider mock
docker compose run --rm agent --input samples/weak_essay.txt             # live
```

`python:3.12-slim` (not alpine — `onnxruntime` has no musl wheel), non-root, dependency layer cached
ahead of the source, and the embedding model **baked in at build time** so the first run needs no
network. Two named volumes: `agent-data` for the memory database, `agent-runs` for traces.

---

## Repository layout

```
config/          config.yaml + rubrics/*.yaml
docs/            patterns research, memory design, harness design, solution write-up
samples/         deliberately weak inputs
scripts/         preflight and demo helpers
demo.py          one-command launcher for the browser demo
src/agentic_rubric/
  config.py      typed layered configuration
  llm/           provider ABC, httpx client, mock provider, JSON salvage
  core/          perceive / reason / act / reflect / loop   (Milestone 1)
  tools/         schemas, registry, handlers                (Milestone 1)
  memory/        episodic + lesson + profile stores         (Milestone 2)
  web/           stdlib demo server + single-page UI
  harness/       retry, fallbacks, guardrails, faults,
                 loop detection, runner                     (Milestone 3)
  observability/ structured logging, JSONL traces, console  (Milestone 3)
tests/           mock-driven, no key or network required
runs/            per-run traces and summaries (gitignored)
Dockerfile       python:3.12-slim, non-root, model baked in
docker-compose.yml
```

---

## Documents

| Document | Covers |
|---|---|
| [Plan.md](Plan.md) | Full milestone-by-milestone task list with stable IDs |
| [progress.md](progress.md) | Live status, decisions with rationale, resume pointer |
| [docs/01_patterns_research.md](docs/01_patterns_research.md) | ReAct, Reflexion, CoT, Tree of Thoughts, LATS — mechanisms, costs, and which are applied here |
| [docs/02_memory_design.md](docs/02_memory_design.md) | Backend choice, schema, scope policy, and the A/B transcript |
| [docs/03_harness_design.md](docs/03_harness_design.md) | Each engineering decision paired with the failure mode it defends against |
| [docs/04_demo_script.md](docs/04_demo_script.md) | Shot-by-shot script for the demo video, with commands and timings |
| [docs/solution_evidence.md](docs/solution_evidence.md) | Every verified number and reference, gathered for the solution write-up |
| [docs/demos/](docs/demos/) | Captured run transcripts, including injected failures |

---

## License

MIT. Private repository until challenge results are announced.
