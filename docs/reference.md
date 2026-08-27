# Agentic Rubric Loop — Technical Reference

**Use case:** score text against a rubric and improve it.

This document explains the implemented stack and then walks through each milestone: what was built,
how it works, and why it was built that way.

> **Scope note.** This is a technical reference for the implementation — the companion to the code,
> not the submission write-up. The Solution PDF (`docs/solution.md`) is a separate document and is
> the author's own work.

---

# Part 0 — The stack

## 0.1 What runs

| Layer | Choice | Why this and not the obvious alternative |
|---|---|---|
| Language | Python 3.10+ (CI: 3.10 and 3.12) | Required by the brief |
| Agent framework | **none** | Required by the brief. The loop is ~1,900 lines of explicit functions in `core/` |
| LLM transport | `httpx` against any OpenAI-compatible `/chat/completions` | One client covers Gemini, Groq, Ollama, OpenAI, OpenRouter. Vendor SDKs would mean re-deriving one error taxonomy from several different exception trees |
| Primary model | **Gemini `gemini-3.5-flash-lite`** (free tier) | Chosen by measurement, not documentation: `gemini-3.6-flash` reports `limit: 5` requests/min and 429s after one iteration, while this model absorbed 14 rapid calls without one |
| Fallback chain | `gemini → groq → ollama` | Groq is fast but capped at 8,000 TPM and 200,000 TPD for `gpt-oss-120b` — roughly six full runs a day. Ollama is unlimited but local and slow. Three vendors, so one outage does not end a run |
| Memory store | SQLite + `sqlite-vec` + FTS5 | Records, keyword index and vectors in **one file**. Two stores can disagree about what the agent remembers, and that failure is silent |
| Embeddings | `fastembed` / `BAAI/bge-small-en-v1.5` (384-d, ONNX) | No PyTorch: ~400 MB image instead of ~2.5 GB |
| Config | YAML + env + CLI, typed frozen dataclasses | Every runtime knob swappable without touching loop code |
| Observability | stdlib `logging` + JSONL trace files | No agent-tracing SaaS; the trace is a file you can `grep` |
| UI | stdlib `http.server` + one HTML page | A demo whose first step is "install a web framework" fails on the reviewer's machine |
| Tests | `pytest`, 345 tests, ~10 s, no key, no network | Every LLM path is substitutable with a scripted provider |

**Total running cost: $0.** Every dependency is open source and every provider has a free path.

## 0.2 Repository shape

```
src/agentic_rubric/
  config.py         typed, layered configuration (CLI > env > YAML > defaults)
  core/             the loop — no retry, no backoff, no provider logic lives here
    admission.py    the pre-loop input gate
    perceive.py     step 1
    reason.py       step 2
    act.py          step 3
    reflect.py      step 4
    loop.py         wires the four; owns the iteration cap
    rubric.py       Rubric, Probe, ScoreCard, the headroom maths
    state.py        the typed hand-offs between steps
  llm/              provider ABC, one httpx client, error taxonomy, scripted provider
  tools/            registry, rubric-derived schemas, five handlers
  memory/           three tiers, two recall channels, policy layer, circuit breaker
  harness/          retry, fallbacks, guardrails, stuck detection, Runner
  observability/    JSON logger, JSONL tracer, console renderer
  web/              stdlib demo server + single-page UI
config/
  config.yaml       every runtime parameter
  rubrics/*.yaml    the domain, as data
```

## 0.3 The one architectural rule

**`core/` contains no infrastructure.** No retry, no backoff, no jitter, no failover, no provider
chain, no token budget, no wall-clock timeout. All of that is in `harness/`, and it attaches to the
loop through exactly **two seams**:

```python
AgenticLoop(
    ...,
    controller=...,   # may stop the run at an iteration boundary; may do nothing else
    act_fn=...,       # may retry and sanitise a tool call; may not choose one
)
```

Both default to "no harness", so `core/` runs standalone and reads as the four steps it is meant to
demonstrate. The `LoopController` protocol is deliberately narrow — it can stop and annotate, and it
cannot choose an action, edit the draft or change a score. No amount of harness code can quietly
become a fifth cognitive step.

---

# Part 1 — Milestone 1: The Core Agentic Loop

## 1.1 Patterns research

### ReAct — Reasoning + Acting

*Yao et al., ICLR 2023 ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629))*

ReAct interleaves two things earlier work kept apart: free-text **reasoning traces** and
**environment actions**. Each turn the model emits a Thought, then an Action; the environment
returns an Observation; the triple is appended to context and the cycle repeats.

```
Thought:     I do not know how good this draft is.
Action:      score_against_rubric()
Observation: 28.7%. Weakest: thesis 2/5, evidence 2/5.
Thought:     Thesis and evidence carry the most weighted headroom.
Action:      revise_text(focus=[thesis, evidence], ...)
Observation: 202 -> 243 words, similarity 0.95.
```

The Thought is not decoration: it gives the model somewhere to plan, and it makes the trajectory
auditable — you can read *why* it did a thing, not only what it did.

**Why it matters.** Pure chain-of-thought reasons in a closed world and drifts; it can produce a
beautifully argued conclusion resting on a fact it invented three steps earlier. Pure
action-selection acts without planning. ReAct's contribution is that the Observation is *external* —
it contradicts the model when the model is wrong.

**Cost and failure modes.** One call per decision, plus tool cost. Cheap and linear. It fails when
the environment gives no useful signal, and it is **greedy — there is no backtracking.** A bad
action at step 3 is a permanent part of the trajectory.

### Reflexion — verbal reinforcement learning

*Shinn et al., NeurIPS 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366))*

Reflexion wraps an acting loop in three components:

| Component | Job |
|---|---|
| **Actor** | produces the trajectory (typically a ReAct agent) |
| **Evaluator** | scores it — a test suite, a heuristic, an LLM judge |
| **Self-reflection** | turns that score into **natural-language** feedback about *why* it was low |

The reflection is written to an episodic buffer and pasted into context on the next attempt. Weights
never change; the *prompt* changes. Hence "verbal" reinforcement learning — the policy improves
through language, not gradients.

**Cost and failure modes.** One extra call per episode. Very cheap relative to the gain. But it is
**only as good as its evaluator**: with a noisy or sycophantic one, Reflexion faithfully learns the
wrong lesson — and because lessons persist, a bad one contaminates every future run. It also
degrades when reflections are too specific to generalise, or too vague to act on.

### Chain-of-Thought

*Wei et al., NeurIPS 2022 ([arXiv:2201.11903](https://arxiv.org/abs/2201.11903)); Kojima et al.
([arXiv:2205.11916](https://arxiv.org/abs/2205.11916)); self-consistency, Wang et al.
([arXiv:2203.11171](https://arxiv.org/abs/2203.11171))*

Prompt the model to emit intermediate reasoning before its answer. Mechanically this gives a
fixed-depth-per-token architecture more forward passes to spend, and conditions the answer on
explicit intermediate results rather than a single leap.

CoT is **not an agent architecture** — no actions, no environment, no memory. It is a prompting
technique, and it composes with everything else.

**Cost and failure modes.** Extra output tokens, one call: essentially free. Its central weakness is
**post-hoc rationalisation** — a model that has already committed to an answer will generate a chain
supporting it. That is exactly the failure mode a rubric judge falls into, and it is why the field
ordering in this project's judge schema is load-bearing (§1.2).

### Tree of Thoughts

*Yao et al., NeurIPS 2023 ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601))*

ToT generalises CoT from a chain to a **tree** and adds explicit search: thought decomposition, a
thought generator (propose *k* next steps), a state evaluator (the model scores each), and a search
algorithm (BFS/DFS with pruning and backtracking).

```
                 root
        ┌─────────┼─────────┐
    branch A   branch B   branch C
     (0.3)      (0.8)      (0.2)      <- evaluator scores
                  │
        ┌─────────┴─────────┐
      B1 (0.9)          B2 (0.4)      <- expand the promising branch only
```

**Cost and failure modes.** *branching × depth* generations plus an evaluation per node — an order
of magnitude above CoT. It pays off only when the task genuinely has a **search structure**: several
plausible next moves, most of them dead ends, and a cheap way to tell them apart. It also depends
entirely on the state evaluator; a bad one prunes the correct branch.

### LATS — Language Agent Tree Search

*Zhou et al., ICML 2024 ([arXiv:2310.04406](https://arxiv.org/abs/2310.04406))*

The synthesis: **Monte Carlo Tree Search over ReAct trajectories, with Reflexion as the
backpropagation signal.**

| MCTS phase | LATS realisation |
|---|---|
| Selection | UCT over children, balancing value against visit count |
| Expansion | sample *k* candidate actions from the LM |
| Simulation | roll forward, executing **real** actions in the environment |
| Evaluation | LM value function plus any real environment reward |
| Backpropagation | update ancestor values; on failure generate a *verbal* reflection and store it for future expansions of that subtree |

It acts in a real environment during search, so node values are grounded rather than imagined, and
it **backtracks** — the capability ReAct lacks.

**Cost and failure modes.** By far the most expensive: tens to hundreds of calls per problem. It
also assumes a **resettable environment**, because MCTS must explore a branch, discard it, and
return to the parent state.

## 1.2 What this project applies, and why

| Pattern | Used | Where |
|---|---|---|
| **ReAct** | ✅ backbone | `core/loop.py`, `core/reason.py` |
| **Reflexion** | ✅ learning mechanism | `core/reflect.py` + `memory/` |
| **Chain-of-Thought** | ✅ inside two prompts | `prompts/score.py`, `prompts/reason.py` |
| **Tree of Thoughts** | ⚠️ one level, opt-in, off by default | `tools/handlers/revision.py` |
| **LATS** | ❌ deliberately rejected | — |

### The decision that drives all of it

The use case supplies something most agentic tasks lack: **a cheap, repeatable, numeric evaluation
of the current state.** Re-scoring the draft costs one call and returns a weighted percentage.

That fact sorts the patterns. Some *need* an external signal (ReAct, Reflexion, LATS); others
*substitute reasoning for one* (CoT, ToT). When a real signal is available and cheap, using a
pattern that simulates one means paying for sophistication you do not need — and, worse, trusting
the model's opinion of its own work when a measurement was available.

### ReAct is the backbone

The four steps *are* ReAct with the phases named and separated:

| ReAct | This loop |
|---|---|
| — | **Perceive** (structure the state; no model) |
| Thought | **Reason** (one call, forced tool use) |
| Action | **Act** (dispatch only) |
| Observation | the tool result |
| — | **Reflect** (evaluate; decide whether to continue) |

The rubric score is a genuine external observation: when the judge returns 28.7% with `evidence` at
2/5, that is a measurement the agent must accept, not its own opinion. Prior Thought/Action/
Observation triples are replayed as a scratchpad (`state.ReactStep`), so a strategy already tried is
visibly a repeat.

**One implementation detail worth defending.** `thought` is a **required property on every tool
schema**, not scraped from the assistant's free text. Several providers return empty `content`
alongside a tool call, which would silently reduce ReAct to plain action-selection — the Thought
would exist only when the provider happened to cooperate. Putting it in the schema makes it
structural.

### Reflexion is the learning mechanism

The mapping is one-to-one:

| Reflexion | This project |
|---|---|
| Actor | Perceive → Reason → Act |
| Evaluator | `score_against_rubric` — weighted percentage, decomposed per criterion |
| Self-reflection | the LLM half of `reflect()` |
| Episodic memory | `memory/` — `lesson` records, recalled across sessions |

**The lesson is the payload.** The reflection schema demands a *transferable* lesson and says so:

- ❌ `"the second paragraph needed a source"` — worthless next session
- ✅ `"unattributed statistics score no higher than no statistics on evidence"` — worth carrying

An **empty** lesson is accepted as an honest answer, because a generic one ("be more specific")
pollutes memory for every future run.

**Where this deviates from the paper, on purpose.** In Reflexion the actor's own judgement can end
an episode. Here it cannot: `task_complete` is computed from the numbers, and the model's opinion is
recorded as `model_votes_done` and counts only as one input to the plateau rule. A `finalize` call
made below target while the score is still climbing is **declined**, the flag cleared, and the agent
told why.

### Chain-of-Thought, used structurally

**In the judge.** The output schema orders each criterion's fields
`criterion_id → evidence → justification → score`. Models fill JSON objects in schema order, so the
quote and the argument are generated **before** the number. Reverse those two fields and the model
picks a score and rationalises it — the post-hoc rationalisation problem, which for a grader is
fatal, because the loop steers on the *difference* between two scores. A judge that drifts is
indistinguishable from progress. The judge also runs at `temperature=0.0` for the same reason.

**In the reasoner.** The `thought` field forces the agent to state *why this action, now, given
these scores* before naming the action.

### Tree of Thoughts — one level deep, off by default

Setting `loop.revise_candidates: N` makes `revise_text` generate N revisions at spread temperatures
(`CANDIDATE_TEMPERATURES = (0.4, 0.7, 0.9, 1.0)`), score each with the judge, and keep the winner.

```
        current draft
        ┌─────┼─────┐
       c1    c2    c3          <- generate at 0.4 / 0.7 / 0.9
       │     │     │
      62%   71%   58%          <- the judge is the state evaluator
             ▲
          keep this one
```

That is a **one-level search tree with value-based pruning** — real ToT machinery (generator +
evaluator + selection), labelled as one level rather than presented as the full method.

**Why only one level.** Text revision has no natural intermediate state to branch on. In Game of 24
a partial expression is a meaningful node; a half-revised paragraph is not — the evaluator cannot
score it and the agent cannot continue from it. A deeper tree would branch on *whole drafts*.

**Why off by default.** N revise calls plus N judge calls per iteration, roughly 3× the tokens at
N=3, for a modest gain. Exposed as configuration so the trade-off is the operator's.

### LATS — rejected, with reasons

1. **The environment does not reset.** MCTS requires exploring a branch, discarding it, and
   returning to the parent state. The state here is a document being progressively rewritten; there
   is no cheap `reset()`. Simulating one means keeping every intermediate draft and re-scoring on
   every backtrack — most of the cost of the search before any search has happened.
2. **The branching factor does not justify it.** Five tools, of which one or two are sensible at any
   point: unscored → score it; scored → revise the highest-headroom criterion. The interesting
   variation is not *which action* but *what instruction to give the reviser* — which the shallow
   ToT branch already explores at one level.
3. **The economics are wrong.** Tens to hundreds of calls per problem against a free tier. A pattern
   you cannot afford to run is not a pattern you have implemented.

**When it would be right:** if the rubric were adversarial — if improving `evidence` reliably damaged
`style_clarity`, so the optimal path required accepting a local loss — the greedy hill-climb here
would get stuck and lookahead would earn its cost. The shipped rubrics are close to separable.

## 1.3 The approach in Milestone 1

### The four steps

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

**Perceive** (`core/perceive.py`) — **no LLM.** Loads the rubric, normalises the draft, computes
deterministic metrics (Flesch, sentence-length variance, hedge/filler counts) and runs the rubric's
regex probes, recalls memory, folds in the previous Reflection, and returns an `Observation`.

Three reasons it is model-free: it keeps the four steps genuinely distinct (a prompting Perceive
would be Reason with a different template); it costs nothing and cannot hallucinate; and it makes
Reason **reproducible** — Reason sees an `Observation` and nothing else, so the same Observation
always builds the same prompt.

**Reason** (`core/reason.py`) — one LLM call with `tool_choice="required"`, returning a `Decision`.
Tool use is forced, so the model cannot answer with prose. Combined with rubric-derived enums in the
schemas, most malformed-output failure modes are *designed out* rather than caught. If a provider
ignores `required` anyway, `fallback_decision()` picks a safe **read-only** action and marks the
Decision `degraded=True` — a decision we do not fully understand must never be one that rewrites the
user's text.

**Act** (`core/act.py`) — a dispatcher. It validates arguments against the schema, runs the handler,
and attributes the tool's token cost to the iteration. It makes **no decisions**. Keeping it
incapable of choosing is what makes the Reason/Act separation real rather than stylistic.

**Reflect** (`core/reflect.py`) — two halves that must not be confused. The **deterministic** half
(`assess`) computes score-before, score-after, delta and plateau, and applies the termination rules.
The **verbal** half is one LLM call producing the critique, `next_focus`, and the *lesson*.

### The feedback edge

`LoopState.advance()` stores the Reflection; the next `perceive()` reads it. That single line is
what makes this a loop rather than four functions called in sequence, and there is a test asserting
each iteration's observation holds the previous iteration's reflection object.

### Termination

Three sources, in priority order:

1. **Reflect** decides the work is done — target met, credible `finalize`, or plateau.
2. **The loop** enforces `max_iterations`, in Python, never delegated to a prompt.
3. A **controller** (the harness) may stop at either iteration boundary.

Whatever the reason, the run returns the **best-scoring draft ever seen**, not the last one. A
revision can make things worse, and an improvement agent that hands back text worse than its input
has failed at its one job.

### The tools

Schemas are generated **from the rubric**, so every argument naming a criterion carries an `enum` of
that rubric's real ids. Asking to improve a criterion that does not exist is structurally impossible
rather than validated after the fact — and the same code produces a correct, different tool set for
the bug-report rubric.

| Tool | LLM? | Purpose |
|---|---|---|
| `score_against_rubric` | yes | measure — the loop's only signal |
| `revise_text` | yes | the only tool that edits; hosts the shallow-ToT branch |
| `analyze_text` | **no** | readability metrics + the rubric's declared regex probes |
| `diff_drafts` | **no** | what the last revision actually changed |
| `finalize` | no | request termination (which Reflect may decline) |

**Two of the five never call a model.** An agent whose every tool is another prompt has no way to
check itself; `analyze_text` and `diff_drafts` are the only things in the loop that can contradict
its self-report.

Tools take a **reference** to the working draft, never the text as an argument. Passing the document
through the model would cost thousands of tokens per call and give it a standing opportunity to
corrupt the text in transit.

Every failure — unknown tool, bad arguments, handler exception — becomes an
`ActionResult(ok=False)`. A broken tool call is data the agent reacts to, not an exception that ends
the run.

### Verified live

A full run against `gemini-3.5-flash-lite` on `samples/weak_essay.txt`:

| Iteration | Action | Result |
|---|---|---|
| 1 | `score_against_rubric` | **7.5%** — thesis 1/5, evidence 1/5 |
| 2 | `revise_text` | candidate rejected by the retention guard; `ToolRecovery` retried |
| 3 | `revise_text` | 202 → 204 words, similarity 0.23 |
| 4 | `score_against_rubric` | **50.0%** |
| 5 | `revise_text` | 204 → 106 words, similarity 0.33 |
| 6 | `score_against_rubric` | **100.0%** |

Trajectory **7.5% → 50.0% → 100.0%** across three measurements, with a guard firing and being
recovered from in between.

### The scoring maths

Scores are normalised to a percentage, so `target_score` means the same thing for a 1–5 rubric and a
0–10 one. **"Weakest" means most weighted headroom** — `weight × remaining points` — not lowest raw
score. A 25%-weighted criterion at 2/5 is worth more than a 15%-weighted one at 1/5; ranking by raw
score sends the agent after cheap points.

### The admission gate

Added after a live failure: given `hi, good morning`, the agent scored a greeting 0.0%, expanded
three words into 170 with 0.00 similarity, scored its own invention at 85.0%, and reported
`target_reached` — ~17,000 tokens spent fabricating a document the user never wrote.

`core/admission.py` now runs **once before the loop, with no model call**, against rubric-declared
word and sentence floors. A refusal is a verdict (`RunStatus.INPUT_REJECTED`), not an exception: it
carries a run id, a reason written for a person, and the user's text handed back unchanged. The
reviser's guards were also made **symmetric** — they caught shrinkage and sameness, but nothing
bounded expansion.

---

# Part 2 — Milestone 2: Memory Integration

## 2.1 Why memory exists here

This is a Reflexion agent. Reflect produces a verbal lesson from a numeric score; memory is what
makes that lesson available to a later decision. Without persistence, Reflexion collapses into
ordinary self-critique — the agent notices something useful and immediately forgets it.

So the design question is not "how do I store and retrieve text". It is **which findings deserve to
outlive the run that produced them, and how do I get them in front of the next decision without
burying it in noise.**

## 2.2 The structure — three tiers, three lifetimes, one file

| Tier | Scope | Lifetime | Example |
|---|---|---|---|
| `episodic` | this session | one run | `iteration 2: revise_text -> 202 -> 243 words, similarity 0.95` |
| `lesson` | this **rubric**, any session | across sessions — the Reflexion payload | `Unattributed figures score no better than no figures.` |
| `profile` | this rubric | standing constraints | target score, tone, banned edits |

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    uid TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,              -- episodic | lesson | profile
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,      -- normalised sha256 prefix, for dedupe
    session_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    rubric_id TEXT NOT NULL,         -- the scope key that stops cross-domain leakage
    criterion_id TEXT NOT NULL,
    score_delta REAL,                -- what this action was worth, in points
    metadata TEXT NOT NULL,          -- JSON: action, ok, thought
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 1  -- times this lesson has been relearned
);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', ...);
CREATE VIRTUAL TABLE memories_vec USING vec0(id INTEGER PRIMARY KEY,
                                             embedding float[384] distance_metric=cosine);
```

Triggers keep the FTS index in sync with inserts, updates and deletes, so the two indexes cannot
drift from the table.

## 2.3 How it works in the loop

**Read at the start of every Perceive. Written after every Reflect.**

```
perceive()  ──▶  memory.recall(query, session_id, rubric_id)
                   │
                   └──▶ RECALLED FROM MEMORY block in the Reason prompt
                        + apply_lessons argument passed to revise_text

reflect()   ──▶  memory.save(episodic record)          # always
                 memory.save(lesson record)            # only when Reflect found one
```

Reading in Perceive rather than inside Reason means recall happens exactly once per iteration, is
visible in the `Observation`, and can fail without the reasoning step needing to know.

### The three required operations

| Operation | Signature | Reachable from |
|---|---|---|
| **Save** | `save(record) -> uid` | automatic, after every Reflect |
| **Recall** | `recall(query, *, session_id, rubric_id, kinds, limit, min_score)` | automatic, in every Perceive |
| **Clear session** | `clear_session(session_id) -> int` | `--clear-session <id>`, UI button |

Plus `list_sessions()` and `stats()` for operability (`--memory-stats`).

## 2.4 Four policy decisions

**The recall query describes the *problem*, not the *text*.** It is composed from the rubric name,
the criterion Reflect nominated, and the two criteria with the most headroom — never from the draft.
Querying with the essay retrieves memories about *similar essays*; querying with the problem
retrieves memories about *how to solve this problem*, which is what the next decision needs. A test
asserts the draft's own words never reach the query.

**Lessons are rubric-scoped, not global.** A finding about attributing statistics in essays is not
evidence about bug reports, and recalling it there is worse than recalling nothing.

**Lessons are ranked but *not* relevance-gated; episodic recall is.** This is the most consequential
call in the module. Episodic records are voluminous and noisy, so they must clear
`recall_min_score`. Lessons are the opposite — a handful per rubric, each already filtered by Reflect
judging it worth keeping. Gating them on cosine similarity discards the point of Reflexion the first
time a query is phrased differently. They are ranked, capped at three, and always offered. The risk
accepted: an irrelevant lesson can reach the prompt.

**The keyword channel is ordinal, not calibrated — and says so.** FTS5 BM25 magnitudes depend on
corpus size (~1e-6 on a ten-row table). Squashing that into a pseudo-probability produces a number
that *looks* absolute and is not, so `recall_min_score` would silently reject everything. Keyword
hits therefore score `0.6 / (1 + rank)`, and in keyword-only mode the gate is **skipped** rather
than applied to a fake score. When both channels hit: `0.7 × vector + 0.3 × keyword`.

A relearned lesson increments `hits` instead of duplicating, with a small ranking boost — a finding
independently rediscovered is better evidence than a one-off. Episodic records never deduplicate.

## 2.5 Degradation

| Failure | Response |
|---|---|
| `fastembed` missing / model download blocked | `NullEmbedder`, keyword-only recall, note in the run |
| `sqlite-vec` will not load | keyword-only, same file, same rows |
| Embedding model changed under an existing DB | dimension mismatch caught; note says to delete the DB |
| Neither channel matches | fall back to **recent** records for the scope |
| Database unopenable | `NullMemory`; the run proceeds without memory |
| Store raises repeatedly | **circuit breaker**: after 3 consecutive failures *per operation*, stop calling it |

The breaker counts failures **per operation**, not in one shared total. A shared counter looks
simpler and is wrong: the realistic outage is a store whose reads fail while its writes still work,
and every successful write would reset the failing read's streak, so the breaker would never open.

## 2.6 The concrete example

`python scripts/memory_ab_demo.py` — three arms, identical input, identical target:

| Run | Iterations | Action sequence |
|---|---|---|
| A · memory on, **cold** store | 6 | score → **analyze_text** → revise → score → revise → score |
| B · memory on, **warm** store, *different session* | **5** | score → **revise** → score → revise → score |
| C · memory **off** (control) | 6 | score → **analyze_text** → revise → score → revise → score |

**A ≡ C is the control that makes B mean anything.** A cold store carries the same information as no
store, so they must behave identically — and they do, action for action. Without it, "memory helped"
could just mean "the second run of anything is faster".

Iteration 2, verbatim:

- **A (cold):** `analyze_text` — *"No prior experience with this rubric was recalled. Measure the
  draft directly before spending a revision on a guess."*
- **B (warm):** `revise_text` — *"Memory says: On this rubric, targeting the two highest-weighted
  criteria first moves the total faster than fixing the lowest raw score… Applying that directly
  rather than rediscovering it."*

The effect reaches the run by two paths: the `RECALLED FROM MEMORY` block in the Reason prompt, and
the `apply_lessons` argument passed through to the reviser so it reaches the rewrite itself.

---

# Part 3 — Milestone 3: Harness Engineering

## 3.1 How it attaches

`harness/runner.py` composes retry, fallbacks, guardrails and tracing around the loop and hands back
a `RunnerReport`. It reaches into the loop through the two seams from §0.3 and nothing else:

```
Runner
 ├── ResilientProvider   wraps the LLM: retry, repair, failover      → act_fn / provider
 ├── ToolRecovery        retries and sanitises failed tool calls     → act_fn
 ├── Guardrails          iteration cap, token budget, wall clock     → controller
 ├── StuckDetector       repeated action, draft cycle, frozen score  → controller
 └── Tracer              one JSONL envelope per event                → on_event
```

## 3.2 Retry

`harness/retry.py`. Exponential backoff with **full jitter** by default (`equal` and `none` also
available), `Retry-After` preferred over the computed delay but capped at 60 s, and an attempt cap.

**Separate policies** for transport (4 attempts, 1.0 s base) and tools (2 attempts, 0.25 s base) —
an LLM rate limit and a bad tool argument deserve different patience.

What decides retryability is the **error taxonomy**, split by *what the caller should do* rather than
by status code:

| Exception | Meaning | Harness response |
|---|---|---|
| `RetryableLLMError` | 429, 5xx, timeout | back off and retry |
| `TerminalLLMError` | 401, 400 | stop — retrying cannot help |
| `LLMParseError` | body unusable | salvage, then repair |
| `ProviderUnavailableError` | nothing listening | fail over |

That split is why the retry decorator stays small and why "back off" and "fail over" are different
code paths rather than one function with a flag.

**Live evidence, twice over.** One five-iteration run on Groq's free tier absorbed **HTTP 429
fifteen times**, honouring every `Retry-After` — 139 s of backoff inside 170 s of wall clock — and
still completed at `target_reached`, 15.0% → 87.5%.

A later run exhausted *both* free tiers in sequence and the ladder behaved exactly as designed:
Gemini returned `429 ... limit: 5` (per-minute request quota), the retry policy backed off three
times, the chain **failed over to Groq**, Groq returned `429 ... tokens per day (TPD): Limit 200000,
Used 200000`, and the chain failed over again. No exception reached the caller at any point.

## 3.3 Fallbacks

`harness/fallbacks.py`. One defined path per failure mode:

| Failure | Ladder |
|---|---|
| **Unparseable LLM output** | forced tool schema → local JSON salvage → one repair call → safe read-only default action, marked `degraded` |
| **Failed tool call** | `ActionResult(ok=False)` fed back as an observation → sanitised retry (drop the arguments the schema rejected) → route to an alternative tool |
| **Iteration cap** | return the **best draft seen**, `status=max_iterations_reached` |
| **Memory read failure** | circuit breaker → `NullMemory`, `degraded_memory=true`, run continues |
| **Token budget** | 80% warning, then a forced graceful finalize |
| **Provider unavailable** | walk the chain `gemini → groq → ollama`; failover is **sticky** |

Two distinctions worth noting: **a 400 is raised but a 404 fails over** — a malformed request will be
malformed everywhere, but a missing model is missing *on this provider*. And auth rejection fails
over rather than aborting, because the key is wrong for that provider, not for all of them.

## 3.4 Observability

`observability/`. Every step of every iteration emits a structured event:

- `runs/<run_id>/trace.jsonl` — one JSON envelope per event, **flushed per line**, so a killed run
  still has everything up to the moment it died
- `runs/<run_id>/summary.json` — the run result
- the same event stream drives the console renderer and the browser UI, so what you watch and what
  is recorded cannot drift apart

Each envelope carries `run_id`, `session_id`, `iteration`, `step`, timing, token counts, a cost
estimate and any error. **Redaction happens in the formatter**, matching both key names and value
patterns, so a secret cannot leak by being logged under an unexpected key.

## 3.5 Guardrails

`harness/guardrails.py` and `harness/loop_detect.py`. All enforced in Python, never in a prompt.

| Guardrail | Behaviour |
|---|---|
| Hard iteration cap | the loop owns it; a model cannot argue with it |
| Token budget | warn at 80%, stop at 100% with a graceful finalize |
| Wall-clock deadline | checked at iteration boundaries |
| Ingestion cap | oversized input truncated for the prompt; metrics still cover the whole text |
| **Stuck detection** | three signals: the same `(action, args)` consecutively; a draft A→B→A cycle (fingerprint); a frozen score across a window |

**Every stop is graceful** — status, reason, best draft, full trace. Never an exception.

## 3.6 Configuration

Everything is in `config/config.yaml`, with precedence:

```
CLI flags  >  environment variables  >  config.yaml  >  dataclass defaults
```

```bash
AGENTIC_LLM__PRIMARY=groq
AGENTIC_LLM__PROVIDERS__GEMINI__MODEL=gemini-3.6-flash
AGENTIC_LOOP__MAX_ITERATIONS=8
AGENTIC_MEMORY__BACKEND=sqlite_fts
python -m agentic_rubric.cli --set loop.revise_candidates=3 ...
```

Config is loaded into **frozen, typed dataclasses** with unknown-key validation — a `max_iteration`
typo raises rather than being silently ignored. `core/` imports neither `os` nor YAML; it receives a
finished `AppConfig`. That is what makes "configurable without touching core loop code" structural
rather than aspirational.

## 3.7 Fault injection

Seven kinds, each injected **at the layer where the real thing occurs** — not by mocking the
harness, which would test the test:

```bash
--simulate-failure {rate_limit,server_error,bad_json,provider_down,tool_error,memory_down,budget}
```

Three real bugs were found this way, all of which shared a shape: **the error containment that kept
the run alive is exactly what hid the bug**, and all three were caught by tests asserting behaviour
*changed*, not by anything crashing.

---

# Part 4 — One iteration, end to end

```
  ┌─ Runner ─────────────────────────────────────────────────────────────┐
  │  Guardrails.before_iteration()  → cap? budget? deadline? stuck?      │
  │                                                                      │
  │   PERCEIVE   metrics + probes (no LLM)                               │
  │              memory.recall(problem-shaped query)  ──▶ Observation    │
  │                                                                      │
  │   REASON     1 call, tool_choice=required                            │
  │              ResilientProvider: retry → salvage → repair → failover  │
  │                                                    ──▶ Decision      │
  │                                                                      │
  │   ACT        validate args against the rubric-derived schema         │
  │              ToolRecovery: retry, sanitise, alternative              │
  │                                                    ──▶ ActionResult  │
  │                                                                      │
  │   REFLECT    rules: delta, plateau, target, finalize-credibility     │
  │              1 call: critique + lesson + next_focus  ──▶ Reflection   │
  │                                                                      │
  │   state.advance(record)   ← the feedback edge                        │
  │   memory.save(episodic [+ lesson])                                   │
  │   Guardrails.after_iteration()                                       │
  │   Tracer writes 6+ JSONL envelopes                                   │
  └──────────────────────────────────────────────────────────────────────┘
```

---

# Appendix — commands

```bash
python scripts/preflight.py --ping        # verify config + provider chain live
python demo.py                            # browser UI
python -m pytest -q                       # 345 tests, no key, no network
python scripts/memory_ab_demo.py          # the memory A/B
python -m agentic_rubric.cli --input samples/weak_essay.txt
python -m agentic_rubric.cli --memory-stats
python -m agentic_rubric.cli --clear-session <id>
python -m agentic_rubric.cli --simulate-failure rate_limit --fail-step judge
```

**Design documents:** `docs/01_patterns_research.md`, `02_memory_design.md`,
`03_harness_design.md`. **Transcripts:** `docs/demos/`.
