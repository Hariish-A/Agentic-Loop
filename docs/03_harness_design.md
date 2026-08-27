# Harness Design — every decision, and the failure mode it defends against

> Milestone 3. Companion to [01_patterns_research.md](01_patterns_research.md) (why the loop is
> shaped this way) and [02_memory_design.md](02_memory_design.md) (what it remembers).

A loop that only works on the happy path is not a deployable agent. This document pairs each piece
of scaffolding with the specific way an unsupervised run goes wrong without it.

The organising principle is stated once, up front, because everything else follows from it:

> **`core/` contains no `try`/`except` at all.** Perceive, Reason, Act, Reflect and the loop that
> wires them read as the four steps they are meant to demonstrate. Every failure is handled by an
> object that was *substituted for one of the loop's collaborators* before it started.

---

## 1. The seam: how the harness attaches without touching the loop

`AgenticLoop` takes four collaborators. The harness replaces all four with decorated versions that
satisfy exactly the same interfaces:

| Collaborator | Default (Milestones 1–2) | What the runner passes | File |
|---|---|---|---|
| `provider` | an `LLMProvider` | `ResilientProvider` — retry, salvage, repair, sticky failover | [`harness/fallbacks.py`](../src/agentic_rubric/harness/fallbacks.py) |
| `act_fn` | `core.act.act` | `ToolRecovery` — same signature, with a recovery ladder | `harness/fallbacks.py` |
| `controller` | `None` | `Guardrails` — budget, clock, cap, stuck detection | [`harness/guardrails.py`](../src/agentic_rubric/harness/guardrails.py) |
| `on_event` | a console renderer, or nothing | tracer + renderer + logger, fanned out | [`observability/trace.py`](../src/agentic_rubric/observability/trace.py) |

Two of these were already there for other reasons (`provider`, `on_event`). The two added in
Milestone 3 total **eleven lines** in `core/loop.py`.

The `LoopController` protocol is deliberately unable to do anything except stop the run:

```python
class LoopController(t.Protocol):
    def before_iteration(self, state) -> StopSignal | None: ...
    def after_iteration(self, state, record) -> StopSignal | None: ...
```

It cannot choose an action, edit the draft, or change a score. **A guardrail that could redirect the
agent would be a fifth cognitive step hiding in the harness**, and the four-step separation the
challenge asks for would quietly stop being true.

`test_the_loop_itself_still_runs_without_a_harness` and
`test_a_controller_can_only_stop_a_run_never_steer_it` pin both halves of that claim.

---

## 2. Retry — [`harness/retry.py`](../src/agentic_rubric/harness/retry.py)

### The decision that made this module short

Back in Phase 0 the LLM error taxonomy was split by **what the caller should do**, not by status
code:

```
RetryableLLMError    -> back off and try again
TerminalLLMError     -> stop; retrying cannot help
ProviderUnavailable  -> try a different provider
LLMParseError        -> the call worked; the payload did not
```

So "should I retry this?" is an `isinstance` check, not a table of HTTP codes maintained in two
places. That is why the retry module is ~120 lines of logic and can be read in one sitting.

### Failure modes and answers

| Failure | Defence | Why that one |
|---|---|---|
| **429 rate limit** — the likeliest free-tier failure | exponential backoff, `Retry-After` honoured in preference to our computed delay | The provider knows when its window resets; we are guessing |
| **5xx / timeouts** | the same backoff | Transient by definition |
| **Synchronised retry storms** | jitter (`full` by default) | Without it, every client that hit the same limit retries at the same instant and rebuilds the spike. `full` = sleep `U(0, d)`; `equal` = `d/2 + U(0, d/2)` trades spread for a guaranteed minimum wait. Which is right depends on how many clients share the quota — a deployment fact, so it is a config knob, not a constant |
| **A provider asking for a 20-minute wait** | `Retry-After` capped at `retry.max_retry_after_s` (60s) | A run that appears to have hung is worse than one that failed over |
| **A bad API key** | *not* retried — raised immediately | Four attempts turn one clear error into a 30-second wait for the same clear error |

### Two policies, not one

```yaml
retry:
  max_attempts: 4          # transport
  base_delay_s: 1.0
  tool_max_attempts: 2     # tools
  tool_base_delay_s: 0.25
```

A failing tool is usually failing *deterministically* — bad arguments, a missing criterion, an empty
diff — and two of the five tools spend tokens on every attempt. Retrying a tool four times is mostly
a way to spend four times the tokens on the same error. The one extra attempt exists for the case
that genuinely is transient: the reviser's own model call getting rate-limited.

### Tested, not asserted

Sleep and randomness are injected, so the tests assert real bounds in milliseconds rather than
waiting through thirty seconds of exponential delay:

```python
def test_jitter_actually_spreads_the_herd() -> None:
    delays = {round(policy.delay_for(2)[0], 4) for _ in range(200)}
    assert len(delays) > 150      # synchronised clients would collide here
```

---

## 3. Fallbacks — [`harness/fallbacks.py`](../src/agentic_rubric/harness/fallbacks.py)

Each entry is a **ladder**, not a single answer: the cheap local recovery is tried before the
expensive one, and every ladder ends in a floor that always succeeds. *A harness whose last rung can
itself fail has not defined a path — it has moved the crash.*

| Failure mode | Rung 1 | Rung 2 | Rung 3 | Floor |
|---|---|---|---|---|
| **Unparseable LLM output** | forced tool schema (the malformed shape is designed out) | local JSON salvage — fences, prose wrappers, trailing commas | **one** repair call stating exactly what was wrong | `fallback_decision()` picks a safe read-only action |
| **Model answers with prose** | same schema | lift the tool call out of the prose | same repair call | same floor |
| **Tool call failed** | typed `ErrorKind` decides | sanitise arguments / back off / route to an alternative | — | the error goes back as an **observation** the agent reacts to |
| **Provider rate-limited** | retry with backoff | fail over to the next provider | — | `ProviderUnavailableError` naming every reason |
| **Provider unreachable / bad key** | fail over immediately | — | — | as above |
| **Memory read failure** | circuit breaker opens after 3 | store behaves as `NullMemory` | — | run continues, `degraded_memory=true` |
| **Iteration cap** | — | — | — | return the **best draft seen**, `max_iterations_reached` |
| **Token budget** | 80% warning | graceful stop | — | best draft, `budget_exhausted` |

### Why only *one* repair call

A model that cannot produce valid JSON twice will not produce it on the third try either. The second
repair round trip costs tokens and latency to re-learn what the first one established. It is a
config knob (`retry.repair_attempts`) because that judgement could reasonably differ per model.

### Why failover is sticky

Once the chain moves on, later calls start from the *new* provider rather than re-testing the dead
one. An agent loop makes ~3 calls per iteration; re-probing a backend that just exhausted its retry
budget would pay the full backoff on every one of them.

### Why a 400 is raised but a 404 fails over

A 404 means "this model id does not exist here" — a deprecated model is exactly what the backup
provider is for. Any other terminal error means **our request is wrong**, and failing over would
burn the whole chain to reproduce our own bug, replacing a precise error message with
"every provider failed".

### Tool recovery is typed, not string-matched

The registry contains every exception to keep the loop alive — which destroys the exception type. So
it classifies at the point of containment:

```python
class ErrorKind(str, Enum):
    VALIDATION    # schema rejected the arguments  -> sanitise and retry
    UNKNOWN_TOOL  # hallucinated name              -> route to the nearest real tool
    TRANSIENT     # rate limit inside a handler    -> back off, retry once
    RECOVERABLE   # the handler declined, and said why -> hand back to the agent
    TERMINAL      # a bug                          -> hand back to the agent
```

Two constraints on recovery:

- **Sanitising only ever *removes* arguments.** Guessing a replacement value would put words in the
  agent's mouth and hide the mistake from the trace. Dropping an argument lets the handler's own
  documented default apply.
- **A substituted tool is always read-only** (`analyze_text`, `score_against_rubric`). A degraded
  decision is one we do not fully understand, and the wrong response to not understanding the
  situation is to start rewriting the user's text.

The `RECOVERABLE`/`TERMINAL` branch does nothing — but it *emits an event saying so*, so that
"the harness did nothing" and "the harness chose to do nothing" are distinguishable afterwards.

---

## 4. Observability — [`logger.py`](../src/agentic_rubric/observability/logger.py), [`trace.py`](../src/agentic_rubric/observability/trace.py), [`render.py`](../src/agentic_rubric/observability/render.py)

### One event source, three subscribers

The console renderer, the JSONL tracer and the structured logger all consume the same `on_event`
stream. **What a reviewer watches on screen and what the trace file records cannot drift apart**,
because they are fed from the same call.

`fanout()` wraps each subscriber in a try/except: observability is the one component that absolutely
must not be able to kill the thing it observes.

### Why JSONL and not one JSON document

A trace is append-only and is most useful when the run *did not* finish. A file that only becomes
valid on its final closing brace is useless in exactly the situation observability exists for. Every
line stands alone, the handle is flushed after every write, and `jq` and `grep` both work without a
parser.

```
runs/<run_id>/trace.jsonl     every event, in order
runs/<run_id>/summary.json    the RunResult plus harness telemetry
```

Every event carries the same envelope — `run_id`, `session_id`, `iteration`, `step`, `tool`,
`duration_ms`, `tokens`, `cost_est`, `error`, `retry_count` — so a column means the same thing on
every row. "Where did the time go" and "which iteration cost the most" are one `jq` away rather than
a parsing exercise. Step-specific detail lands under `detail`.

### Redaction happens in the formatter, not at the call sites

A rule that depends on every caller remembering it is not a rule. Everything reaching a log record
or a trace line passes through `redact()` first, so the way to leak a key is to bypass logging
entirely rather than to forget a keyword. Both key names (`logging.redact_keys`) **and value
patterns** (`Bearer …`, `sk-…`, `gsk_…`) are matched — a key pasted into an error message does not
arrive under a helpfully-named field.

Long strings are truncated *visibly* (`... [+4900 chars]`): a reader can tell "short" from
"shortened", and a trace nobody can open is not observability.

### Cost estimation is honestly zero

`cost_est` comes from `cost_per_1k_input` / `cost_per_1k_output` on the provider's config entry,
which default to `0.0`. Every provider in the shipped chain has a free path, and **inventing a price
would be worse than an honest zero**. Set them to your plan's rates and the estimate becomes real
without a code change.

### The run_summary event

The loop emits `run_end` *without* the harness block. It never learns that a retry, a repair or a
failover happened — by design — so reporting zeroes for them would be a confident lie. The runner
emits `run_summary` afterwards with the real numbers.

---

## 5. Guardrails — [`harness/guardrails.py`](../src/agentic_rubric/harness/guardrails.py)

| Guardrail | Failure it defends against | Where enforced |
|---|---|---|
| `max_iterations` | An agent that can decide it is finished can also decide it is not | The `while` condition in `core/loop.py`, in Python. **Never expressed as a request in a prompt** |
| `token_budget` | The archetypal runaway cost: a revision loop re-reads and rewrites a long document every iteration | Checked *before and after* each iteration — one iteration can spend several thousand tokens across Reason, judge, reviser and Reflect |
| `token_warn_ratio` | Discovering the budget only when it stops you | A note and an event at 80%, once |
| `wall_clock_timeout_s` | Tokens do not measure a hung socket, a 90-second provider, or an embedder that decided to download a model | Both boundaries |
| `max_input_chars` | An unbounded prompt | Perceive, where the prompt view is built — measurements still cover the whole document |
| `max_document_chars` | A pasted 40 MB file making metrics, diffing and hashing crawl | Once, at ingestion, before the run starts |

**Every stop is graceful.** The run ends with a status, a reason, the best draft seen and a complete
trace — never a raised exception. An agent that crashes on its budget has thrown away the work it
already paid for.

### A bug this found in the token accounting

Wiring the budget guardrail required a running token total, which exposed that `Reflect`'s usage had
never been added to the run total — only Reason's and the tools'. The budget would have been
enforced against a number roughly a third too low. `test_token_accounting_includes_every_call`
counts against the provider's own call log so the assertion cannot drift with the loop's shape.

---

## 6. Stuck detection — [`harness/loop_detect.py`](../src/agentic_rubric/harness/loop_detect.py)

The failure mode this defends against is the expensive one, **because it does not look like a
failure**. Milestone 1 produced a live example: a registry bug made every tool call fail, and the
loop ran all six iterations, failed identically each time, and returned cleanly. Nothing raised. The
error containment that kept the run alive is exactly what hid the problem.

Three independent signals, because the same pathology surfaces in three different places:

**`repeated_action`** — the same `(action, arguments)` signature N times **consecutively**.
Consecutive matters: a healthy agent alternating `score → revise → score → revise` calls
`score_against_rubric()` with identical (empty) arguments many times, and a *total* count would flag
it as stuck. `test_alternating_actions_are_not_stuck` pins that.

**`draft_cycle`** — the draft returns to a state it already held. Not "the draft did not change" —
that is normal on every scoring turn — but A → B → A, meaning two revisions are undoing each other.
Compared on a whitespace- and case-normalised hash so cosmetic churn cannot disguise it.

**`score_plateau`** — the last `stuck_score_window` scorecards span less than `stuck_score_epsilon`.
Normally pre-empted by Reflect's own plateau rule; this is the backstop for when `min_improvement` is
configured looser than epsilon.

Being stuck produces `status=stuck` and the best draft. It is not an error: the agent did real work,
and the harness is declining to pay for more of it.

---

## 7. Everything is configurable without touching loop code

The challenge asks that model, iteration limits, token budget, retry settings and memory backend all
be configurable outside the core loop. All of them live in
[`config/config.yaml`](../config/config.yaml), reachable four ways:

```bash
# 1. the file                    2. environment
AGENTIC_GUARDRAILS__TOKEN_BUDGET=50000 AGENTIC_RETRY__JITTER=equal python -m agentic_rubric.cli ...

# 3. a CLI flag                  4. --set, for any key at all
python -m agentic_rubric.cli --max-iters 3 --set retry.tool_max_attempts=1
```

Precedence: **CLI > env > YAML > dataclass defaults**. Nothing under `core/` imports `os` or reads
YAML; the loop is handed a frozen `AppConfig`.

Provider quirks are **capability flags on the config record**, never branches on a provider name:
`supports_message_name: false` for Groq, `requires_key: false` for Ollama, `retry_on_status` for the
whole taxonomy. A test flips 418 from terminal to retryable through config alone.

---

## 8. Failure injection — [`harness/faults.py`](../src/agentic_rubric/harness/faults.py)

Every ladder above is reachable from the command line. The alternative is asserting that the harness
*would* recover if a provider ever rate-limited us, which is a claim, not evidence.

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock \
    --simulate-failure {rate_limit|server_error|bad_json|provider_down|tool_error|memory_down|budget}
```

Each is injected **at the layer where the real thing would occur** — a rate limit inside the
provider, a fault inside a real tool handler, an outage inside the memory store — rather than by
short-circuiting the harness into pretending.

| Flag | Observed result | Transcript |
|---|---|---|
| `rate_limit` | `retry 1 on mock after 0.05s (Retry-After)`, run completes | [m3_failure_rate_limit.txt](demos/m3_failure_rate_limit.txt) |
| `bad_json` | `unusable reply from mock: sent one repair prompt`, run completes | [m3_failure_bad_json.txt](demos/m3_failure_bad_json.txt) |
| `provider_down` | `failover mock-primary -> mock`, run completes | [m3_failure_provider_down.txt](demos/m3_failure_provider_down.txt) |
| `tool_error` | `revise_text failed; backoff_retry -> revise_text`, run completes | [m3_failure_tool_error.txt](demos/m3_failure_tool_error.txt) |
| `memory_down` | breaker opens after 3, `running without memory: …` on every later iteration, run completes | [m3_failure_memory_down.txt](demos/m3_failure_memory_down.txt) |
| `budget` | `[token_budget] 1,080 of 900 tokens spent` → `STOP (budget_exhausted)`, best draft returned | [m3_failure_budget.txt](demos/m3_failure_budget.txt) |

### A bug `--simulate-failure memory_down` found immediately

`memory_down` fails **reads only** — a corrupt index or a locked reader, which is the realistic
half-outage and is strictly harder on the harness than a store that is uniformly down. The first run
of it showed the circuit breaker never opening.

The Milestone 2 `MemoryManager` counted "consecutive failures" in **one shared integer across every
operation**. Since the loop writes after every Reflect, each successful write reset the failing
read's streak. The breaker could never open, and the run paid for a failing read on every single
iteration, forever, while reporting itself healthy.

Fixed by counting per operation. This is the second time in this project that an error-containment
mechanism has hidden a bug from itself — the first was in Milestone 2 — and both were caught by a
test asserting that behaviour **changed**, not by anything crashing.

---

## 9. Evidence

### A live run, no simulation

`groq / openai/gpt-oss-120b`, free tier, `samples/weak_essay.txt`
([full transcript](demos/m3_live_groq_run.txt)):

```
status      : target_reached
iterations  : 5
trajectory  : 15.0% -> 32.5% -> 87.5%
score       : 15.0% -> 87.5%  (+72.5 points)
tokens      : 21,552 in / 8,540 out
elapsed     : 169.67s
harness     : provider=groq:openai/gpt-oss-120b (fallbacks: ollama)
              retries=15 repairs=0 failovers=0 tool_recoveries=0
budget      : 30,092 / 200,000 tokens (15%)
```

**Fifteen real rate limits, absorbed.** Not injected — Groq's free tier genuinely returned HTTP 429
fifteen times across this five-iteration run, each carrying its own `Retry-After` (1s to 24s, 139s
of waiting in total), **all fifteen honoured** in preference to our computed backoff. That is why a
170-second run spent only ~30 seconds doing work. Without this module the run dies on the second
call.

```bash
$ jq -r 'select(.event=="retry")
         | "\(.detail.delay_s)s \(.detail.error_type) honoured=\(.detail.honoured_retry_after)"' \
      docs/demos/m3_live_groq_trace.jsonl | head -4
12.0s RateLimitError honoured=true
4.0s  RateLimitError honoured=true
5.0s  RateLimitError honoured=true
24.0s RateLimitError honoured=true
```

Note what the harness did *not* have to do: `repairs=0`, `failovers=0`. All fifteen failures were
answered at the cheapest available rung of the ladder.

An earlier live run of the same command reached `target_reached` at **100.0%** in 5 iterations with
**16** absorbed rate limits (211s of honoured backoff, 246s wall clock). Two runs, sixteen and
fifteen genuine 429s, both completed — the recovery is reproducible rather than a lucky sample. The
final scores differ (87.5% vs 100.0%) because the judge is a real LLM and the reviser above it runs
at `llm.temperature`: the judge asks for `temperature=0`, which Groq silently converts to `1e-8`, so
it is *near*-deterministic rather than deterministic. The *shape* — climb, re-score, terminate on the
rule — is the same both times, and both runs cleared the 85% target.

Artifacts: [m3_live_groq_run.txt](demos/m3_live_groq_run.txt),
[m3_live_groq_trace.jsonl](demos/m3_live_groq_trace.jsonl),
[m3_live_groq_summary.json](demos/m3_live_groq_summary.json).

### A second thing the live run found

`preflight --ping` reported that Groq replied `''` — an empty string, HTTP 200. `openai/gpt-oss-120b`
is a **reasoning model**: it spends output tokens on an internal `reasoning` field *before* emitting
any content, so a `max_tokens` that looks generous returns a successful response with nothing in it.

Two changes, both earned rather than anticipated:

1. The client now raises `LLMParseError` when a completion is empty **and** `finish_reason=length`,
   with a message naming `llm.max_tokens` as the fix. Returning it as valid would have let
   `revise_text` replace the user's draft with an empty string.
2. `llm.max_tokens` raised from 2048 to 4096, with the reason written next to it in the config.

### Test coverage

```
$ python -m pytest -q
242 passed
    test_config 12   test_harness 58   test_llm_layer 42   test_loop 30
    test_memory 31   test_rubric 19    test_tools 34       test_web 16
```

Every harness test runs offline against `MockProvider`, with every sleep injected. The suite needs
no API key, no network, and takes under five seconds.

---

## 10. Container — [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml)

| Decision | Failure it defends against |
|---|---|
| `python:3.12-slim`, not alpine | `onnxruntime` (via fastembed) ships no musl wheel; on alpine pip compiles from source, turning a 30-second build into twenty minutes for a *larger* image |
| Embedding model baked in at build time (`scripts/warm_models.py`) | Otherwise the first `recall` downloads ~90 MB — a demo that looks broken, an air-gapped deployment that is impossible, and a download inside the wall-clock budget |
| Dependencies installed before source is copied | Editing a Python file rebuilds one small layer instead of re-downloading onnxruntime |
| Non-root `uid 10001`, `data/` and `runs/` pre-created and chowned | A host bind mount landing root-owned |
| `.env` excluded from the image; keys arrive via `env_file` at run time | A credential baked into a layer that could be pushed |
| `restart: "no"` | A restarting batch job silently re-spends the token budget |
| `HEALTHCHECK` runs `preflight.py` | "Will this even work here?" answered without spending a token |
| Two named volumes, not one | The memory database and the traces have different lifetimes and different reasons to be kept |

---

## 11. What I would do differently

**The stuck detector's `score_plateau` signal barely earns its place.** Reflect's own
`min_improvement` rule fires first in every realistic configuration, so the guardrail version is
dead code most of the time. It is kept because the two rules answer to different config knobs and
a user who loosens one should not silently lose the other — but if I were cutting scope, this goes
first.

**Repair could be smarter and is not.** The repair prompt states the error and re-asks. It does not
narrow the tool set to the one that failed, lower the temperature, or reduce `max_tokens` to make
truncation less likely. All three are cheap and would probably raise the repair success rate; I ran
out of evidence to choose between them, and guessing would have produced three knobs nobody had
measured.

**The token budget is enforced between iterations, not within one.** An iteration that decides to
revise a 20,000-character draft can overshoot the budget by a few thousand tokens before anyone
checks. Enforcing mid-iteration means threading the guardrail into the tool context, which would
have put budget logic inside the tool handlers — a worse trade than a bounded overshoot.

**`cost_est` is per-event and priced at the *active* provider**, so a run that fails over mid-way
prices its early events at the new provider's rate. Correct pricing means carrying the provider name
on every event, which the envelope does not currently do. `summary.json`'s total is computed once at
the end and is right; the per-event column is indicative.

**Wall-clock is checked at iteration boundaries only.** A single hung call can exceed the limit by
the provider timeout (90 seconds on Groq) before anyone notices. A real fix is a deadline threaded
into the HTTP client, which `httpx` supports and I did not wire, because it means the timeout has
two owners.
