# Agentic Rubric Loop

**Score text against a rubric, then improve it** — a hand-rolled Perceive → Reason → Act → Reflect
agentic loop with no agent framework in the core path.

The agent scores a draft against a weighted rubric, decides which criterion is the highest-value
lever, revises the text targeting that criterion, re-scores, and repeats until it hits the target
score, plateaus, or trips a guardrail.

> **Status:** Milestones 1, 2 and 3 complete. See [Plan.md](Plan.md) for the task list and
> [progress.md](progress.md) for what is done and what is next.

---

## Why this use case

A rubric score is a real, external, numeric observation. That makes the loop *verifiable*: you can
see whether iteration 3 actually beat iteration 2, rather than taking the model's word for it. It
also makes Reflexion a natural fit — the scalar reward that pattern assumes is just sitting there.

---

## Architecture

```
                 ┌───────────────────────── Reflection feeds forward ──────────────────────────┐
                 │                                                                             │
                 ▼                                                                             │
   input ──▶ PERCEIVE ──▶ REASON ──▶ ACT ──▶ REFLECT ──▶ done? ──yes──▶ best draft + trace      │
             (no LLM)     (1 LLM     (pure    (LLM +        │                                   │
                           call,     dispatch  determin-    └──no──────────────────────────────┘
                           forced    only)     istic)
                           tools)
```

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
python -m pytest -q                        # 242 passed, in seconds
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

Retry, fallbacks, guardrails and observability wrap the loop **without touching it**. `core/`
contains no `try`/`except` at all: the harness substitutes decorated versions of the loop's own
collaborators before it starts.

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
| [docs/demos/](docs/demos/) | Captured run transcripts, including injected failures |

---

## License

MIT. Private repository until challenge results are announced.
