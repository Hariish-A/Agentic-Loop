# Agentic Rubric Loop

**Score text against a rubric, then improve it** — a hand-rolled Perceive → Reason → Act → Reflect
agentic loop with no agent framework in the core path.

The agent scores a draft against a weighted rubric, decides which criterion is the highest-value
lever, revises the text targeting that criterion, re-scores, and repeats until it hits the target
score, plateaus, or trips a guardrail.

> **Status:** Phase 0 complete (foundation). See [Plan.md](Plan.md) for the task list and
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
| LLM | Any OpenAI-compatible endpoint | One ~200-line httpx client covers Gemini, Grok, Ollama, OpenAI, OpenRouter |
| Primary provider | **Google Gemini** (free tier) | No card required; generous enough for development |
| Fallback chain | Grok → Ollama → mock | Provider failover is itself a demoable failure mode |
| Memory | SQLite + `sqlite-vec` + `fastembed` | One file, no server, works offline; BM25/FTS5 auto-fallback |
| HTTP | `httpx` | Direct control over status→error mapping; retries owned by our harness |
| Config | YAML + env + CLI, typed dataclasses | Every runtime knob swappable without touching loop code |

**Total running cost: $0.** Every dependency is OSS and every provider option has a free path.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
# source .venv/bin/activate && pip install -r requirements-dev.txt  # macOS/Linux

cp .env.example .env          # then add at least one API key (optional for the mock demo)
python scripts/preflight.py   # validates config, shows which providers are usable
python scripts/preflight.py --ping   # makes one real API call
```

Run the tests — no API key or network required:

```bash
python -m pytest -q
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
AGENTIC_LLM__PRIMARY=grok
AGENTIC_LOOP__MAX_ITERATIONS=8
AGENTIC_LLM__PROVIDERS__GROK__MODEL=grok-4.6
AGENTIC_MEMORY__BACKEND=sqlite_fts
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
with no code change. Each declares weighted criteria, level descriptors and improvement hints in
YAML.

---

## Repository layout

```
config/          config.yaml + rubrics/*.yaml
docs/            patterns research, memory design, harness design, solution write-up
samples/         deliberately weak inputs
scripts/         preflight and demo helpers
src/agentic_rubric/
  config.py      typed layered configuration
  llm/           provider ABC, httpx client, mock provider, JSON salvage
  core/          perceive / reason / act / reflect / loop   (Milestone 1)
  tools/         schemas, registry, handlers                (Milestone 1)
  memory/        episodic + lesson + profile stores         (Milestone 2)
  harness/       retry, fallbacks, guardrails, runner       (Milestone 3)
  observability/ structured logging and run traces          (Milestone 3)
tests/           mock-driven, no key or network required
```

---

## Documents

| Document | Covers |
|---|---|
| [Plan.md](Plan.md) | Full milestone-by-milestone task list with stable IDs |
| [progress.md](progress.md) | Live status, decisions with rationale, resume pointer |
| `docs/01_patterns_research.md` | ReAct, Reflexion, CoT, Tree of Thoughts, LATS — and which are applied here |
| `docs/02_memory_design.md` | Backend choice, schema, and a concrete before/after example |
| `docs/03_harness_design.md` | Each engineering decision paired with the failure mode it defends |

---

## License

MIT. Private repository until challenge results are announced.
