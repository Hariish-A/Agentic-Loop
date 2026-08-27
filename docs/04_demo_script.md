# Demo video script — 3 to 5 minutes

> **What the brief requires on camera:** the loop running end to end with **at least 3 full
> iterations**, **memory recall visibly influencing the output**, and **at least one failure
> scenario handled gracefully**. All three are below, plus the live-provider proof.

---

## Before you record

```bash
# 1. Clean slate, so the cold run is genuinely cold.
rm -f data/memory.db
rm -rf runs/*

# 2. Confirm the live key works (you will show this in Shot 5).
python scripts/preflight.py --ping

# 3. Terminal at ~100 columns, large font. The transcript is 88 columns wide;
#    anything narrower wraps and the score bars stop lining up.
export PYTHONPATH=src        # Windows: $env:PYTHONPATH="src"
```

**Do not run the loop again between the reset and recording.** Shot 2 depends on `data/memory.db`
containing exactly what Shot 1 wrote.

**A note on why most of this is `--provider mock`.** The offline provider is deterministic, so the
run on camera is the run you rehearsed — no rate limit, no re-take, no waiting. The *live* provider
gets its own shot at the end, where the point is precisely that it is unpredictable. Say this out
loud on camera; it reads as rigour, not as hiding.

---

## Shot 1 — the loop, cold · ~60s

**Command**

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --session demo-a
```

**What appears** (6 iterations, ~3 seconds — pause and scroll):

```
==== ITERATION 1 ====
  PERCEIVE  [??????????????????????????????????]  unscored
  REASON    -> score_against_rubric
            "The draft is unscored, so I cannot tell where the points are. Measure first."
  ACT       score_against_rubric [ok]
            Scored 28.7%. Biggest opportunities: thesis 2/5 (18.8pts available), ...
  REFLECT   complete=False
            next focus: thesis

==== ITERATION 2 ====
  REASON    -> analyze_text
            "No prior experience with this rubric was recalled. Measure the draft
             directly before spending a revision on a guess."
...
  status      : target_reached
  iterations  : 6
  trajectory  : 28.7% -> 66.2% -> 96.2%
```

**Say, over the scroll:**

> Four steps, and they are genuinely different things. **Perceive** uses no LLM at all — it computes
> word counts, a Flesch score and regex probes, so the agent always has a source of truth it cannot
> talk itself out of. **Reason** is the only step that decides, and it is one call with forced tool
> use. **Act** just dispatches — it cannot choose. **Reflect** measures the score delta in Python
> and asks the model for a critique and a lesson.
>
> Three scorings: 28.7 to 66.2 to 96.2 per cent. Note that completion is decided by a **rule**, not
> by the model. If the agent calls `finalize` while it is still below target and still climbing, the
> call is declined and it is told why — because an agent that can declare itself finished will.

**Point at iteration 2 specifically.** It spends a whole turn on `analyze_text`, and the thought
says why: *nothing was recalled*. That is the line the next shot pays off.

---

## Shot 2 — memory changing the behaviour · ~60s

**Command** — same text, same rubric, same target. Only the session id differs:

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --session demo-b
```

**What appears** (5 iterations):

```
==== ITERATION 2 ====
  REASON    -> revise_text
            "Memory says: On this rubric, targeting the two highest-weighted criteria
             first moves the total faster than fixing the lowest raw score...
             Applying that directly to Thesis and Position rather than rediscovering it."
```

**Say:**

> Different session, same everything else. Iteration 2 no longer explores — it recalls a lesson that
> the *previous* session wrote after its Reflect step, and goes straight to revising. Six iterations
> becomes five.
>
> And the lesson does not just sit in the prompt. It is passed through to the reviser as an
> `apply_lessons` argument, so it reaches the rewrite itself.

**Then the control** — this is the part that makes the claim honest:

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock --no-memory
```

> Back to six iterations. That is the control. Without it, "memory helped" could just mean "the
> second run of anything is faster". A cold store and no store behave **identically**, action for
> action, which is what makes the warm run meaningful.

If time is tight, replace both memory shots with the single scripted A/B:

```bash
python scripts/memory_ab_demo.py          # runs all three arms and prints the comparison
```

---

## Shot 3 — a failure, handled · ~50s

Run two. Both take about three seconds.

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock \
    --simulate-failure rate_limit
```

```
  HARNESS   ! retry 1 on mock after 0.05s (Retry-After): RateLimitError
  ACT       score_against_rubric [ok]
  ...
  status      : target_reached
  harness     : retries=1 repairs=0 failovers=0 tool_recoveries=0
```

```bash
python -m agentic_rubric.cli --input samples/weak_essay.txt --provider mock \
    --simulate-failure budget
```

```
  HARNESS   ! [token_budget] 1,080 of 900 tokens spent
  HARNESS   STOP (budget_exhausted)
  status      : budget_exhausted
  best        : [##########-------------------|----]
```

**Say:**

> The rate limit is injected **inside the provider**, where a real one occurs — not by
> short-circuiting the harness into pretending. It backs off, honours the `Retry-After` header, and
> the run completes. The summary line reports what recovery actually cost.
>
> The budget case is the more interesting one. The guardrail stops the run — and hands back the
> **best draft it ever saw**, not the last one. An agent that crashes on its budget has thrown away
> the work it already paid for.

There are seven of these — `rate_limit`, `server_error`, `bad_json`, `provider_down`, `tool_error`,
`memory_down`, `budget`. Mention that; show two.

---

## Shot 4 — what got recorded · ~30s

```bash
ls runs/
cat runs/<run_id>/summary.json | head -30
jq -r 'select(.event=="reason") | "\(.iteration)  \(.detail.action)  \(.tokens) tok"' \
    runs/<run_id>/trace.jsonl
```

**Say:**

> Every step of every iteration, one JSON object per line. JSONL rather than one document on
> purpose — a trace matters most when the run *did not* finish, and a file that only becomes valid
> on its closing brace is useless in exactly that case. Secrets are redacted in the formatter, so
> the way to leak a key is to bypass logging entirely rather than to forget a keyword.

---

## Shot 5 — the live provider · ~40s

Do **not** run a full live loop on camera — Groq's free tier will rate-limit and you will be
watching backoff for three minutes. Show the ping, then the archived transcript.

```bash
python scripts/preflight.py --ping
```

```
  1. [ ok ] groq     openai/gpt-oss-120b      ok
  2. [ ok ] ollama   qwen2.5:7b-instruct      ok
[ ok ] groq:openai/gpt-oss-120b replied 'ready'
```

Then open [`docs/demos/m3_live_groq_run.txt`](demos/m3_live_groq_run.txt) and scroll to the end:

```
status      : target_reached
trajectory  : 15.0% -> 32.5% -> 87.5%
harness     : provider=groq:openai/gpt-oss-120b (fallbacks: ollama)
              retries=15 repairs=0 failovers=0 tool_recoveries=0
elapsed     : 169.67s
```

**Say:**

> This is a real run against Groq. Fifteen genuine 429s — not injected — every `Retry-After`
> honoured, 139 of those 170 seconds were backoff. The run finished anyway. Without the retry module
> it dies on the second call.

---

## Optional bonus — the browser demo · ~20s

Only if you are comfortably under time.

```bash
python demo.py       # http://127.0.0.1:8000
```

Run once with session `demo-a`, then again with `demo-b`, and point at the recalled lesson
highlighted in the Perceive panel. Note on camera that this view drives the loop **directly**, so it
shows no harness events.

---

## Closing line · ~15s

> No agent framework anywhere in the core path. Four steps, five tools, three memory tiers, and a
> harness that has absorbed fifteen real rate limits without the loop containing a single
> `try`/`except`. Three hundred and three tests, none of which need an API key.

---

## Timing

| Shot | Target | Cumulative |
|---|---|---|
| 1 · the loop, cold | 60s | 1:00 |
| 2 · memory + control | 60s | 2:00 |
| 3 · two failures | 50s | 2:50 |
| 4 · the trace | 30s | 3:20 |
| 5 · live provider | 40s | 4:00 |
| closing | 15s | 4:15 |

Leaves ~45 seconds of slack inside the 5-minute limit. If you overrun, **cut Shot 4** — the trace is
easy to verify from the repository and is the least visual of the five. Never cut Shot 2; memory
recall changing behaviour is 20% of the marks.

---

## Recording checklist

- [ ] `rm -f data/memory.db` immediately before Shot 1, and not again after
- [ ] `.env` is **not** on screen at any point, and neither is `docker compose config` output
- [ ] Terminal ≥ 100 columns
- [ ] `preflight --ping` succeeds before you start recording
- [ ] Iteration counts land at 6 / 5 / 6 — if not, the store was not reset
- [ ] Upload unlisted; put the link in the submission form
