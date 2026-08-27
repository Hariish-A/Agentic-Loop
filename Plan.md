# Plan — Agentic Rubric Loop

> **Use case:** Score text against a rubric and improve it.
> **Constraint:** No agent frameworks in the core loop. Python 3.10+. Everything free-tier.

This file is the **static** task list. Live status lives in [progress.md](progress.md).
Every task has a stable ID (`P0-3`, `M2-5`, …) so progress entries can reference it.

**Task states:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped

---

## Legend: how this maps to the evaluation criteria

| Criterion | Weight | Delivered by |
|---|---|---|
| Loop Correctness | 25% | M1-4 … M1-9 |
| Memory Integration | 20% | M2-1 … M2-8 |
| Harness Engineering | 20% | M3-1 … M3-13 |
| Patterns Understanding | 15% | M1-10, M1-11 |
| Tool Design | 10% | M1-3 |
| Code Quality | 10% | P0-2, M4-1, M4-2 (typing, tests, lint, docstrings) |

---

## Phase 0 — Foundation

*Goal: a repo where `pytest` and a preflight check both pass before any agent code exists.*

- [x] **P0-1** Repo scaffold + `git init` (src layout, `config/`, `docs/`, `tests/`, `samples/`, `scripts/`)
- [x] **P0-2** Project metadata: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, ruff + mypy + pytest config
- [x] **P0-3** Hygiene files: `.gitignore`, `.dockerignore`, `.env.example`
- [x] **P0-4** `config/config.yaml` — every runtime knob for all three milestones declared up front
- [x] **P0-5** `config.py` — typed, frozen dataclasses; 4-layer precedence (CLI > env > YAML > defaults); unknown-key validation
- [x] **P0-6** `llm/types.py` — `Message`, `ToolSpec`, `ToolCall`, `Usage`, `LLMResponse` + error taxonomy split by *caller action* (retryable / terminal / parse / unavailable)
- [x] **P0-7** `llm/parsing.py` — tolerant JSON salvage (fences, prose wrappers, trailing commas, string-aware brace matching)
- [x] **P0-8** `llm/base.py` — `LLMProvider` ABC the loop depends on
- [x] **P0-9** `llm/openai_compatible.py` — one httpx client for every OpenAI-compatible endpoint (GroqCloud / Ollama / OpenAI / OpenRouter); HTTP status → error taxonomy; config-driven retryable statuses; per-provider capability flags; usage estimation fallback
- [x] **P0-10** `llm/mock.py` — deterministic scripted provider + fault injection + stateful responder
- [x] **P0-11** `llm/factory.py` — build by config key; pre-flight availability check; failover chain reporting
- [x] **P0-12** `config/rubrics/essay_argumentative.yaml` + `config/rubrics/bug_report.yaml` (5 weighted criteria each, level descriptors, improvement hints)
- [x] **P0-13** `samples/` — deliberately weak essay and bug report so the loop has headroom
- [x] **P0-14** `scripts/preflight.py` — validates config, prints the provider chain with reasons, optional live `--ping`
- [x] **P0-15** Test suite for config + LLM layer (52 tests, zero network, zero API key)
- [x] **P0-16** `Plan.md`, `progress.md`, `README.md`
- [x] **P0-17** Initial commit

---

## Milestone 1 — The Core Agentic Loop

*Goal: four genuinely distinct steps, ≥2 tools, and a demo showing 3+ iterations with a rising score.*

### Domain model
- [x] **M1-1** `core/rubric.py` — `Rubric`, `RubricCriterion` dataclasses + YAML loader + weight-sum validation + `ScoreCard.weighted_percent()`
- [x] **M1-2** `core/state.py` — `Observation`, `Decision`, `ActionResult`, `Reflection`, `LoopState`, `RunResult`; `LoopState.advance()` is the feedback edge

### Tools (Tool Design, 10%)
- [x] **M1-3** `tools/` — registry + JSON Schemas + handlers for five tools:
  - `score_against_rubric` (LLM judge, CoT: evidence → justification → score)
  - `revise_text` (LLM, constrained rewrite targeting named criteria; hosts the shallow-ToT branch)
  - `analyze_text` (pure Python: Flesch, sentence-length variance, hedge/filler counts, **plus** the rubric's declared regex probes — `analyze_readability` and `check_structure` merged, since both answer "what is measurably true of this draft")
  - `diff_drafts` (pure Python `difflib`: what the last revision actually changed)
  - `finalize` (control tool; requests termination, which Reflect may decline)
  - Schemas are built **from the rubric**, so criterion arguments carry an `enum` of real ids
  - Registry validates arguments against the schema *before* dispatch; every failure mode returns `ActionResult(ok=False)` rather than raising

### The four steps (Loop Correctness, 25%)
- [x] **M1-4** `core/perceive.py` — **no LLM**. Normalises input, loads rubric, runs deterministic metrics, folds in the previous `Reflection`, recalls memory (stub until M2) → `Observation`
- [x] **M1-5** `core/reason.py` — **one LLM call**, forced tool-use, returns `Decision(thought, action, args)`; ReAct scratchpad of prior Thought/Action/Observation triples
- [x] **M1-6** `core/act.py` — pure dispatcher; no decision-making; captures result, error, timing
- [x] **M1-7** `core/reflect.py` — deterministic checks (score delta, plateau, target met) **plus** an LLM self-critique producing a reusable lesson → `Reflection(done, reason, lesson, next_focus)`
- [x] **M1-8** `core/loop.py` — wires the four; feeds `Reflection` into the next `Perceive`; terminates on `done` or `max_iterations`; tracks best-scoring draft throughout
- [x] **M1-9** `prompts/` — versioned prompt templates, one module per step, with the rubric rendered from YAML rather than hardcoded

### Patterns research (Patterns Understanding, 15%)
- [x] **M1-10** `docs/01_patterns_research.md` — ReAct, Reflexion, CoT, Tree of Thoughts, LATS: mechanism, loop shape, cost profile, failure modes, paper citation
- [x] **M1-11** Same doc: which patterns this loop applies and **why they fit a rubric-scoring task**; explicit, reasoned rejection of LATS; shallow-ToT branch documented as `loop.revise_candidates > 1`

### Entry point + demo
- [x] **M1-12** `cli.py` — `--input --rubric --target --max-iters --provider --config`, dotted `--set key=value` overrides
- [x] **M1-13** End-to-end test against `MockProvider` proving ≥3 iterations, a rising score, and clean termination — no API key needed
- [x] **M1-14** Live demo run on `samples/weak_essay.txt`; capture transcript into `docs/demos/`
- [x] **M1-15** Commit + tag `milestone-1`

---

## Milestone 2 — Memory Integration

*Goal: what the agent stores in iteration N provably changes its behaviour in iteration N+1 and in a later session.*

- [x] **M2-1** `memory/base.py` — `MemoryStore` ABC exposing the three required operations: `save`, `recall(query)`, `clear_session` (+ `list_sessions`, `stats`)
- [x] **M2-2** Record tiers — **episodic** (per-session), **lesson** (cross-session Reflexion output), **profile** (rubric constraints); typed `MemoryRecord` with `session_id`, `iteration`, `rubric_id`, `criterion_id`, `score_delta`, `created_at`, `hits`. *Landed in `memory/base.py` rather than a separate `records.py` — one small module beats two.*
- [x] **M2-3** `memory/sqlite_store.py` — SQLite schema, migrations, FTS5 keyword recall (zero-dependency baseline)
- [x] **M2-4** `memory/embedding.py` + vector half of `sqlite_store.py` — `sqlite-vec` + `fastembed` semantic recall in the *same* database file; automatic degradation to FTS5/BM25 when the embedder is unavailable
- [x] **M2-5** `memory/manager.py` — scope rules (lessons **rubric**-scoped across sessions, episodes session-scoped), hybrid vector+keyword ranking, gate applied to episodic only, dedupe, circuit breaker
- [x] **M2-6** Wire into the loop: **read** at the start of every `perceive`, **write** after every `reflect`; recalled context rendered into the Reason prompt as a labelled block
- [x] **M2-7** `scripts/memory_ab_demo.py` — three arms (cold / warm / no-memory) proving warm memory saves an iteration, with cold≡no-memory as the control
- [x] **M2-8** `docs/02_memory_design.md` — backend choice and why, schema, scope rules, and the concrete A/B transcript with the recalled record quoted
- [x] **M2-9** Tests: round-trip save/recall, session isolation, `clear_session`, embedder-down → BM25 fallback, cross-session lesson recall
- [x] **M2-10** Commit + tag `milestone-2`

---

## Milestone 3 — Harness Engineering

*Goal: safe to run unsupervised. Four areas, each with a named failure mode it defends against.*

### Retry
- [x] **M3-1** `harness/retry.py` — exponential backoff + full/equal jitter, `Retry-After` honoured, attempt cap, retryable-vs-terminal classification, separate policies for LLM calls and tool calls

### Fallbacks (one defined path per failure mode)
- [x] **M3-2** `harness/fallbacks.py`  *(memory circuit breaker already done in `memory/manager.py`)*
  - Unparseable LLM output → forced schema → local salvage → one repair call → safe default action
  - Tool failure → structured `ToolError` fed back as an observation → sanitised retry → mark degraded, route to alternative
  - Iteration cap → return **best draft seen**, `status=max_iterations_reached`
  - Memory read failure → circuit-breaker to no-op store, `degraded_memory=true`, loop continues
  - Token budget exhausted → forced graceful `finalize`
  - Provider unavailable → walk the failover chain (`groq → ollama`), now **inside** the run loop and sticky
  - Also landed: `harness/faults.py` (injection for all seven kinds) and a 404-vs-400 failover rule

### Observability
- [x] **M3-3** `observability/logger.py` — structured JSON logger with secret redaction
- [x] **M3-4** `observability/trace.py` — one JSONL event per step per iteration: `run_id, session_id, iteration, step, tool, duration_ms, tokens, cost_est, error, retry_count`; per-run `runs/<run_id>/trace.jsonl` + `summary.json`
- [x] **M3-5** `observability/render.py` — human-readable console view for the demo video

### Guardrails
- [x] **M3-6** `harness/guardrails.py` — hard iteration cap enforced outside the model, token/cost budget with 80% warning, wall-clock timeout, input truncation
- [x] **M3-7** `harness/loop_detect.py` — repeated `(action, args)` signature, score plateau within `stuck_score_epsilon`, near-identical draft hash → `status=stuck`

### Integration
- [x] **M3-8** `harness/runner.py` — composes retry + fallbacks + guardrails + tracing around the loop; the loop itself stays free of `try/except` sprawl. *Attaches through two new seams on `AgenticLoop` (`controller`, `act_fn`), both defaulting to "no harness" — eleven lines in `core/loop.py`.*
- [x] **M3-9** `--simulate-failure {rate_limit,server_error,bad_json,provider_down,tool_error,memory_down,budget}` for on-camera failure demos. *Seven kinds, not six: `server_error` was already there and earns its place separately from `rate_limit` (no `Retry-After` header, so it exercises computed backoff rather than the honoured hint).*
- [x] **M3-10** `Dockerfile` (python:3.12-slim, non-root, layer-cached deps) + `docker-compose.yml` + volumes for `data/` and `runs/`
- [x] **M3-11** `docs/03_harness_design.md` — each engineering decision paired with the failure mode it defends against
- [x] **M3-12** Tests: backoff timing bounds, jitter spread, terminal-vs-retryable, each fallback path, budget trip, stuck detection
- [x] **M3-13** Commit + tag `milestone-3`

---

## Milestone 4 — Submission

- [x] **M4-1** `README.md` final: LLM choice and rationale, setup, run commands, Docker instructions, architecture diagram
- [x] **M4-2** Coverage pass + `ruff` + `mypy` clean — 80% → **91%**, 303 tests, both linters clean
- [x] **M4-3** `.github/workflows/ci.yml` — lint + tests on push (mock provider, so CI needs no secrets)
- [x] **M4-4** Demo video script: `docs/04_demo_script.md` — shot-by-shot, timed to 4:15 of 5:00. *Script written; **the recording is still to do**.*
- [~] **M4-5** `docs/solution.md` → **Solution PDF** (4–8 pages). *The brief forbids an AI-written solution PDF, so the prose is the author's. Delivered instead: `docs/solution_evidence.md` (every verified number and reference, no drafted prose) and `scripts/make_pdf.py` (renders it print-ready, warns outside 4–8 pages).*
- [~] **M4-6** Push to **private** GitHub repo, add reviewer accounts as collaborators. *Pushed to `Hariish-A/Agentic-Loop` (main + tags `milestone-1/2/3`) after the owner confirmed the repository is private. **Reviewer accounts still need adding as collaborators.***
- [x] **M4-7** Final check against the submission checklist

---

## Standing constraints

1. **No agent frameworks** in `src/agentic_rubric/core/`. LangChain/LlamaIndex/CrewAI/AutoGen are reference-only.
2. **Nothing in `core/` reads `os.environ` or YAML.** It receives a frozen `AppConfig`.
3. **Every LLM-touching path must be testable with `MockProvider`.** The suite runs with no key and no network.
4. **Repository stays private** until results are announced.
5. **`progress.md` is updated in the same commit as the work it describes** — that is what makes a token-exhaustion restart cheap.
