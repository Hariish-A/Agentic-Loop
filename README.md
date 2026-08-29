# Rubric Forge

**An AI-powered system that scores text against a configurable rubric and iteratively forges it into a stronger draft.**

[![CI](https://github.com/Hariish-A/Agentic-Loop/actions/workflows/ci.yml/badge.svg)](https://github.com/Hariish-A/Agentic-Loop/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Evaluating and improving written content is usually a fragmented process. A user receives a score,
then has to interpret the feedback, decide what to fix, rewrite the draft, and request another
evaluation. One-shot AI editors reduce some of this effort, but they often provide subjective
feedback, revise without a measurable goal, or declare completion without proving that the new draft
is better.

The challenge is to build a system that can:

- evaluate text against explicit, weighted criteria;
- identify the improvement with the greatest effect on the total score;
- revise the text without losing the original intent;
- learn useful strategies across sessions; and
- stop safely when it reaches the target, stops improving, or exhausts its limits.

## Proposed Solution

Rubric Forge turns text improvement into a measurable agentic loop. It loads a rubric from YAML,
scores the submitted draft, selects the highest-value action, applies a focused revision, evaluates
the result, and repeats until a deterministic completion rule is met.

The system follows a **Perceive → Reason → Act → Reflect** workflow:

1. **Perceive** gathers the draft, rubric, text metrics, failed probes, prior feedback, and relevant
   memories.
2. **Reason** uses an LLM to select exactly one rubric-aware tool and provide validated arguments.
3. **Act** dispatches the selected tool, scores or revises the draft, and records the observation.
4. **Reflect** measures the score change, detects plateaus, extracts a reusable lesson, and decides
   whether another iteration is worthwhile.

Rubric Forge always returns the best-scoring draft observed during the run—not simply the last
revision produced. The rubric remains data rather than application code, so the same engine can move
from argumentative essays to engineering bug reports without a code change.

## Implementation

### Architecture

```text
                         Reliability Harness
               retries · failover · budgets · tracing
                                  │
Input ──► Perceive ──► Reason ──► Act ──► Reflect ──► Complete?
             ▲                                      │       │
             └──────────── next iteration ──────────┘       ▼
             │                                          Best draft
       Persistent Memory                              + run trace
    SQLite · vectors · FTS5
```

The core loop is implemented without an agent framework. Perceive and Act are deterministic Python
stages, while Reason and Reflect contain the model-assisted decisions. This keeps scoring,
validation, stopping conditions, and tool execution under application control.

### Core capabilities

- **Configurable rubrics:** weighted criteria, scoring levels, improvement hints, and deterministic
  probes are defined in YAML.
- **Rubric-aware tools:** tool schemas are generated from the active rubric and validated before
  execution. The agent can score, revise, analyze, compare, or finalize a draft.
- **Persistent memory:** SQLite stores session episodes, cross-session lessons, and rubric-specific
  profiles. `sqlite-vec` and FastEmbed provide semantic recall, with FTS5/BM25 as a fallback.
- **Provider abstraction:** one `httpx` client supports OpenAI-compatible endpoints. Gemini is the
  default provider, followed by Groq and local Ollama; a deterministic mock enables offline tests.
- **Reliability harness:** retry and backoff policies, sticky provider failover, parse repair, token
  and time budgets, plateau detection, loop detection, and graceful degradation protect each run.
- **Observability:** every run produces a JSONL event trace and summary containing iterations,
  scores, model usage, memory activity, recoveries, and guardrail outcomes.
- **Interactive application:** the browser dashboard exposes the complete loop, text revisions,
  score changes, memory activity, harness events, and raw traces.

### Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -r requirements-dev.txt

# macOS/Linux
# .venv/bin/python -m pip install -r requirements-dev.txt
```

For a live provider, copy `.env.example` to `.env`, add a Gemini or Groq API key, and verify the
configuration:

```bash
python scripts/preflight.py
python scripts/preflight.py --ping
```

Run the complete loop offline with the deterministic provider:

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock
```

Launch the interactive application with a configured live provider:

```bash
python demo.py
```

The dashboard opens at `http://127.0.0.1:8000`.

Run the automated checks:

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src
```

Run with Docker:

```bash
docker compose run --rm agent --input samples/weak_essay.txt --provider mock
```

## Tech Stack

| Area | Technology |
|---|---|
| Core application | Python 3.10+, typed dataclasses |
| Agent design | Perceive–Reason–Act–Reflect, ReAct, Reflexion, optional shallow Tree of Thoughts |
| LLM integration | Gemini, Groq, Ollama, OpenAI-compatible APIs, deterministic mock provider |
| HTTP layer | `httpx` |
| Memory and retrieval | SQLite, `sqlite-vec`, FTS5/BM25, FastEmbed |
| Configuration | YAML, environment variables, `python-dotenv`, CLI overrides |
| Web application | Python `http.server`, HTML, CSS, JavaScript |
| Observability | Structured JSON logging, JSONL traces, run summaries |
| Testing and quality | Pytest, Ruff, mypy, coverage, GitHub Actions |
| Deployment | Docker, Docker Compose |

## Use Case

Rubric Forge is useful whenever text quality can be expressed as explicit criteria and improved over
multiple iterations. The repository includes two complete examples:

- **Argumentative essays:** strengthen the thesis, evidence, reasoning, structure, and clarity while
  preserving the writer's central position.
- **Engineering bug reports:** improve reproduction steps, expected-versus-actual behavior,
  environment details, severity, and blameless technical language.

The same workflow can be extended with new YAML rubrics for reports, proposals, documentation,
applications, support responses, or other structured writing tasks. A user supplies the text and
rubric; Rubric Forge produces the best-scoring revision together with the score trajectory,
criterion-level evidence, stored lessons, and a complete audit trace.
