# Progress Log

Companion to [Plan.md](Plan.md). Newest entry first.
If a session ends abruptly, read **▶ Resume here** at the top, diff it against `Plan.md`, and continue.

---

## ▶ Resume here

| | |
|---|---|
| **Last completed** | Phase 0 — Foundation (P0-1 … P0-17) |
| **Next task** | **M1-1** — `core/rubric.py`: `Rubric`/`RubricCriterion` dataclasses, YAML loader, weight validation, `ScoreCard.weighted_percent()` |
| **Then** | M1-2 (state objects) → M1-3 (tools) → M1-4…M1-8 (the four steps + loop) |
| **Blocked on** | Nothing. Optional: add a `GEMINI_API_KEY` to `.env` for live runs; the mock path needs no key. |

**Verify the checkout is healthy before continuing:**

```bash
.venv/Scripts/python.exe -m pytest -q          # expect: all pass
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe scripts/preflight.py  # config + provider chain
```

---

## 2026-08-27 — Phase 0: Foundation ✅

**Status:** complete · **Tasks:** P0-1 … P0-17 · **Tests:** 52 passing · **Lint:** clean

### What was built

| Area | Files | Notes |
|---|---|---|
| Scaffold | `src/agentic_rubric/{core,llm,tools,memory,harness,observability,prompts}/` | src layout, git repo on `main` |
| Packaging | `pyproject.toml`, `requirements*.txt` | ruff + mypy + pytest configured; console script `agentic-rubric` reserved |
| Hygiene | `.gitignore`, `.dockerignore`, `.env.example` | `.env`, `data/*.db`, `runs/*` never committed |
| Config | `config/config.yaml`, `src/agentic_rubric/config.py` | frozen dataclasses; CLI > env > YAML > defaults; unknown keys rejected |
| LLM layer | `llm/{types,parsing,base,openai_compatible,mock,factory}.py` | one httpx client for all OpenAI-compatible providers |
| Rubrics | `config/rubrics/{essay_argumentative,bug_report}.yaml` | 5 weighted criteria each, weights verified to sum to 1.00 |
| Samples | `samples/weak_essay.txt`, `samples/weak_bug_report.txt` | deliberately weak, so iterations have headroom |
| Preflight | `scripts/preflight.py` | config validation + provider availability + optional live ping |
| Tests | `tests/test_config.py`, `tests/test_llm_layer.py` | 52 tests, no network, no API key |
| Docs | `Plan.md`, `progress.md`, `README.md` | |

### Decisions made (and why)

- **Raw `httpx`, not a vendor SDK.** The job of this layer is mapping provider HTTP outcomes onto one error taxonomy. Going through four SDKs would mean re-deriving that mapping from four exception trees.
- **Error taxonomy split by caller action**, not by status code: `RetryableLLMError` / `TerminalLLMError` / `LLMParseError` / `ProviderUnavailableError`. This is what keeps the M3 retry decorator to a dozen lines, and it makes "fail over vs back off" a type-level decision.
- **`requires_key` declared explicitly per provider** rather than inferred from a localhost URL. "Why was this provider skipped?" is answerable from `config.yaml` alone.
- **Which HTTP statuses are retryable comes from config**, not hardcoding — proven by a test that flips 418 from terminal to retryable via config alone.
- **`Usage.estimate()` fallback** at ~4 chars/token for providers that omit usage (Ollama). Crude on purpose: it exists so the M3 token-budget guardrail still has a signal, not to be an accurate biller.
- **`MockProvider` accepts a stateful `responder` callable**, not just a fixed tape — so the offline end-to-end demo can actually react to conversation state instead of replaying a script that ignores its input.
- **`ToolCall.signature()` is order-independent** (`json.dumps(..., sort_keys=True)`) because the M3 stuck-loop detector will hash it.
- **All config knobs for M2 and M3 were declared in `config.yaml` now**, before the code that reads them exists. Cheaper than retrofitting, and it forces the "configurable without touching core loop code" requirement to be designed in rather than bolted on.

### Verification evidence

```
$ .venv/Scripts/python.exe -m pytest -q
....................................................            [100%]      52 passed

$ .venv/Scripts/python.exe -m ruff check src tests scripts
All checks passed!

$ .venv/Scripts/python.exe scripts/preflight.py
[ ok ] config parsed
  max_iterations : 6      target_score : 85.0     token_budget : 200,000
  retry attempts : 4 (full jitter)
  memory backend : sqlite_vec -> ...\data\memory.db
--- provider failover chain ---
  1. [fail] gemini   gemini-2.5-flash      environment variable GEMINI_API_KEY is empty
  2. [fail] grok     grok-4.20-fast        environment variable XAI_API_KEY is empty
  3. [ ok ] ollama   qwen2.5:7b-instruct   ok
```

### Open items carried forward

- `sqlite-vec` and `fastembed` are listed in `requirements.txt` but **not yet installed** — deferred to M2 to keep the Phase 0 environment light. Install with `pip install -r requirements.txt` when starting M2.
- Model IDs in `config.yaml` (`gemini-2.5-flash`, `grok-4.20-fast`, `qwen2.5:7b-instruct`) are placeholders until a live `--ping` confirms them against the account in use.
- `config/config.yaml` intentionally contains no secrets. Keys come from `.env` via `api_key_env`.

---

## Template for future entries

```
## YYYY-MM-DD — <Milestone / Phase>: <title>  ✅ | 🚧 | ⛔
**Status:** … · **Tasks:** … · **Tests:** … · **Lint:** …
### What was built
### Decisions made (and why)
### Verification evidence
### Open items carried forward
```
