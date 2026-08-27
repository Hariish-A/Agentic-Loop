# Agentic Reasoning Patterns — Research and Applied Choices

> **Deliverable:** Milestone 1, patterns research.
> **Scope:** five patterns explained, then the ones this project applies, with the reasoning.

---

## 0. Why the choice of pattern is not cosmetic here

The use case is *score text against a rubric and improve it*. That produces one property most
agentic tasks lack: **a cheap, repeatable, numeric evaluation of the current state**. Re-scoring the
draft costs one model call and returns a weighted percentage.

That single fact decides most of what follows. Patterns divide roughly into those that *need* an
external signal (ReAct, Reflexion, LATS) and those that *substitute reasoning for one* (CoT, ToT).
When a real signal is available and cheap, using a pattern that simulates one is paying for
sophistication you do not need — and, worse, trusting a model's opinion of its own work when a
measurement was sitting right there.

---

## 1. ReAct — Reasoning + Acting

**Paper:** Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023
([arXiv:2210.03629](https://arxiv.org/abs/2210.03629))

### How it works

ReAct interleaves two things earlier work kept separate: free-text **reasoning traces** and
**environment actions**. Each turn the model emits a Thought, then an Action, and the environment
returns an Observation. The triple is appended to the context and the cycle repeats.

```
Thought:     I do not know how good this draft is.
Action:      score_against_rubric()
Observation: 28.7%. Weakest: thesis 2/5, evidence 2/5.
Thought:     Thesis and evidence carry the most weighted headroom.
Action:      revise_text(focus=[thesis, evidence], ...)
Observation: 202 -> 243 words, similarity 0.95.
```

The Thought is not decoration. It gives the model a place to do the planning that a bare
action-selection head has nowhere to put, and it makes the trajectory auditable — you can read why
the agent did a thing, not merely what it did.

### Why it matters

Pure chain-of-thought reasons in a closed world and drifts: it can produce a beautifully argued
conclusion built on a fact it invented three steps earlier. Pure action-selection acts without
planning. ReAct's contribution is that the Observation is *external* — it contradicts the model when
the model is wrong.

### Cost and failure modes

One model call per decision, plus whatever the tools cost. Cheap and linear.

It fails when the environment gives no useful signal (nothing to correct the reasoning), and it is
greedy: ReAct has **no backtracking**. A bad action at step 3 is a permanent part of the trajectory
and the agent can only reason forward from the mess it made.

---

## 2. Reflexion — verbal reinforcement learning

**Paper:** Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*,
NeurIPS 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366))

### How it works

Reflexion sits *around* an acting loop and adds three components:

| Component | Job |
|---|---|
| **Actor** | Produces the trajectory (typically a ReAct agent) |
| **Evaluator** | Scores the trajectory — a test suite, a heuristic, an LLM judge |
| **Self-reflection** | Turns that score into **natural-language** feedback about *why* it was low |

The reflection is written to an episodic memory buffer and pasted into the context on the next
attempt. The model's weights never change; the *prompt* changes. Hence "verbal" reinforcement
learning — the policy improves through language, not gradients.

```
attempt 1 ── evaluator: 28.7% ── reflect: "unattributed figures score no
                                           better than no figures"
                                              │
                                              ▼  (written to memory)
attempt 2 ── context now includes that lesson ── evaluator: 66.2% ...
```

The reported gains are large where the evaluator is trustworthy — the paper reports 91% pass@1 on
HumanEval, where unit tests provide an unambiguous signal.

### Cost and failure modes

One extra model call per episode, plus memory storage. Very cheap relative to the improvement.

It is **only as good as its evaluator**. With a noisy or sycophantic evaluator, Reflexion faithfully
learns and reinforces the wrong lesson — and because lessons persist, a bad one contaminates every
future run. It also degrades when reflections are too specific to generalise ("fix paragraph two") or
too vague to act on ("be more specific").

---

## 3. Chain-of-Thought — reasoning as generated tokens

**Papers:** Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*,
NeurIPS 2022 ([arXiv:2201.11903](https://arxiv.org/abs/2201.11903));
Kojima et al., *Large Language Models are Zero-Shot Reasoners*
([arXiv:2205.11916](https://arxiv.org/abs/2205.11916)) — the "let's think step by step" variant;
Wang et al., *Self-Consistency* ([arXiv:2203.11171](https://arxiv.org/abs/2203.11171)) — sample many
chains, take the majority answer.

### How it works

Prompt the model to emit intermediate reasoning steps before its answer, rather than answering
immediately. Mechanically, this gives a fixed-depth-per-token architecture more forward passes to
spend on a problem, and it conditions the final answer on explicit intermediate results instead of
on a single leap.

CoT is not an agent architecture. It has no actions, no environment and no memory — it is a
*prompting technique*, and it composes with everything below.

### Cost and failure modes

Extra output tokens, one call. Essentially free.

Its central weakness is **post-hoc rationalisation**: a model that has already committed to an
answer will generate a chain that supports it. This is exactly the failure mode a rubric judge falls
into — pick "4", then write a justification for 4 — and it is why the ordering of fields in this
project's judge schema is load-bearing (see §6.3).

---

## 4. Tree of Thoughts — deliberate search over reasoning states

**Paper:** Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*,
NeurIPS 2023 ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601))

### How it works

ToT generalises CoT from a chain to a **tree**, and adds explicit search. Four ingredients:

1. **Thought decomposition** — what constitutes one step
2. **Thought generator** — propose *k* candidate next steps from a state
3. **State evaluator** — the model scores or votes on how promising each state is
4. **Search algorithm** — BFS or DFS over the tree, with pruning and backtracking

```
                 root
        ┌─────────┼─────────┐
    branch A   branch B   branch C
     (0.3)      (0.8)      (0.2)      <- evaluator scores
                  │
        ┌─────────┴─────────┐
      B1 (0.9)          B2 (0.4)      <- expand the promising branch only
```

The headline result is Game of 24, where CoT solves roughly 4% of instances and ToT roughly 74% —
because the task requires exploring and abandoning arithmetic paths, which a chain structurally
cannot do.

### Cost and failure modes

Expensive: *branching × depth* generations plus an evaluation call per node. An order of magnitude
above CoT, sometimes more.

It only pays off when the task genuinely has a **search structure** — several plausible next moves,
most of them dead ends, and a cheap way to tell them apart. On tasks with one obvious next step, ToT
spends 10× the tokens to rediscover it. It also depends entirely on the state evaluator; a bad
evaluator prunes the correct branch.

---

## 5. LATS — Language Agent Tree Search

**Paper:** Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting and Planning in
Language Models*, ICML 2024 ([arXiv:2310.04406](https://arxiv.org/abs/2310.04406))

### How it works

LATS is the synthesis: **Monte Carlo Tree Search over ReAct trajectories, with Reflexion as the
backpropagation signal.** Each node is a state reached by a sequence of thought/action pairs, and
the standard MCTS cycle applies:

| MCTS phase | LATS realisation |
|---|---|
| **Selection** | UCT over child nodes, balancing value against visit count |
| **Expansion** | Sample *k* candidate actions from the LM |
| **Simulation** | Roll the trajectory forward, executing real actions in the environment |
| **Evaluation** | LM value function + any real environment reward |
| **Backpropagation** | Update ancestor values; on failure, generate a *verbal* reflection and store it for future expansions from that subtree |

Crucially LATS acts in a real environment during search, so its node values are grounded in actual
observations rather than imagined ones. It also **backtracks** — the capability ReAct lacks. Reported
results include 92.7% pass@1 on HumanEval with GPT-4.

### Cost and failure modes

By far the most expensive: tens to hundreds of model calls per problem, multiplied by rollout depth.

It also assumes something this use case does not provide: a **resettable environment**. MCTS needs
to explore a branch, discard it, and return to the parent state. That is trivial for code (re-run
the tests) and impossible for anything with irreversible side effects. And when the action space is
narrow, the search finds the same answer a single ReAct pass would have, at 50× the cost.

---

## 6. What this project applies, and why

### Summary

| Pattern | Used? | Where |
|---|---|---|
| **ReAct** | ✅ Core architecture | [`core/loop.py`](../src/agentic_rubric/core/loop.py), [`core/reason.py`](../src/agentic_rubric/core/reason.py) |
| **Reflexion** | ✅ The learning mechanism | [`core/reflect.py`](../src/agentic_rubric/core/reflect.py), [`memory/`](../src/agentic_rubric/memory/) |
| **Chain-of-Thought** | ✅ Inside two prompts | [`prompts/score.py`](../src/agentic_rubric/prompts/score.py), [`prompts/reason.py`](../src/agentic_rubric/prompts/reason.py) |
| **Tree of Thoughts** | ⚠️ One level, opt-in | [`tools/handlers/revision.py`](../src/agentic_rubric/tools/handlers/revision.py) |
| **LATS** | ❌ Deliberately rejected | §6.5 |

---

### 6.1 ReAct is the backbone

The four-step loop *is* ReAct with the phases named and separated:

| ReAct | This loop |
|---|---|
| — | **Perceive** (structure the state; no model) |
| Thought | **Reason** (one call, forced tool use) |
| Action | **Act** (dispatch only) |
| Observation | tool result |
| — | **Reflect** (evaluate; decide whether to continue) |

**Why it fits.** The rubric score is a genuine external observation. When the judge returns 28.7%
with `evidence` at 2/5, that is not the agent's opinion of its work — it is a measurement the agent
must accept. This is precisely the condition under which ReAct beats pure CoT, and precisely what an
imagined-observation pattern would throw away.

The prior Thought/Action/Observation triples are replayed as a scratchpad
([`state.ReactStep`](../src/agentic_rubric/core/state.py)), so a strategy the agent already tried is
visibly a repeat rather than a fresh idea.

**One implementation detail worth defending.** `thought` is a **required property on every tool
schema**, not scraped from the assistant's free text. Several providers return empty content
alongside a tool call, which would silently reduce ReAct to plain action-selection — the Thought
would exist only when the provider happened to cooperate. Putting it in the schema makes it
structurally guaranteed. See
[`tools/registry.build_spec`](../src/agentic_rubric/tools/registry.py).

---

### 6.2 Reflexion is the learning mechanism

This is the strongest fit of the five, because Reflexion assumes exactly what this use case supplies.

Reflexion needs an **Evaluator** producing a scalar signal. A weighted rubric percentage is that
signal — better-behaved than most, since it is decomposed per criterion, so the reflection can be
attributed to a dimension rather than to the draft as a whole.

The implementation maps one-to-one:

| Reflexion | This project |
|---|---|
| Actor | Perceive → Reason → Act |
| Evaluator | `score_against_rubric` (weighted percentage per criterion) |
| Self-reflection | The LLM half of `reflect()` |
| Episodic memory | `memory/` — `lesson` records, recalled across sessions |

**The lesson is the payload.** The reflection schema demands a *transferable* lesson and says so
explicitly ([`prompts/reflect.py`](../src/agentic_rubric/prompts/reflect.py)):

- ❌ `"the second paragraph needed a source"` — worthless next session
- ✅ `"unattributed statistics score no higher than no statistics on evidence"` — worth carrying

The prompt also accepts an **empty** lesson as an honest answer, because a generic lesson
("be more specific") pollutes memory for every future run — the Reflexion failure mode from §2.

**Where this project deviates from the paper, on purpose.** In Reflexion the actor's own judgement
can end an episode. Here it cannot: `task_complete` is computed from the numbers, and the model's
opinion is recorded as `model_votes_done` and counts only as one input to the plateau rule. A
`finalize` call made below target while the score is still climbing is **declined**, the flag is
cleared, and the agent is told why. An agent permitted to declare its own success will do so as soon
as the remaining points get hard.

---

### 6.3 Chain-of-Thought, used narrowly and structurally

CoT appears in two places, both to fight the same failure mode.

**In the judge.** The output schema orders each criterion's fields
`criterion_id → evidence → justification → score`. Models fill JSON objects in schema order, so the
quote and the argument are generated **before** the number. Reverse those two fields and the model
picks a score first and rationalises it afterwards — the post-hoc rationalisation problem from §3,
which for a grader is fatal, because the loop steers on the *difference* between two scores. A judge
that drifts is indistinguishable from progress.

The judge also runs at `temperature=0.0` for the same reason: sampling noise in a measuring
instrument reads as improvement.

**In the reasoner.** The `thought` field forces the agent to state *why this action, now, given
these scores* before naming the action.

---

### 6.4 Tree of Thoughts — one level deep, off by default

Setting `loop.revise_candidates: N` (N > 1) makes `revise_text` generate N independent revisions at
spread temperatures, score each with the judge, and keep the winner.

```
        current draft
        ┌─────┼─────┐
       c1    c2    c3          <- generate at temperature 0.4 / 0.7 / 0.9
       │     │     │
      62%   71%   58%          <- the judge is the state evaluator
             ▲
          keep this one, discard the rest
```

That is a **one-level search tree with value-based pruning** — real ToT machinery (generator +
evaluator + selection), honestly labelled as one level rather than presented as the full method.
There is no backtracking, no multi-step lookahead, and no re-expansion of discarded branches.

**Why only one level.** Text revision has no natural intermediate state to branch on. In Game of 24
a partial arithmetic expression is a meaningful node; a half-revised paragraph is not — the
evaluator cannot score it and the agent cannot continue from it. Any deeper tree would branch on
*whole drafts*, which multiplies cost by branching factor per level for a task where the second-best
branch is usually 3–4 points behind the best.

**Why it is off by default.** It costs N revise calls plus N judge calls per iteration — roughly
3× the tokens at N=3 — for a modest gain. It is exposed as configuration so the trade-off is the
operator's, not baked in.

---

### 6.5 LATS — rejected, and why

LATS is the natural "go further" answer, and it is the wrong one here. Three concrete reasons:

**1. The environment does not reset.** MCTS requires exploring a branch, discarding it, and
returning to the parent state. The state here is a document being progressively rewritten. Each
revision is a mutation of the text; there is no cheap `reset()`. Simulating one means keeping every
intermediate draft and re-scoring on every backtrack — which is most of the cost of the search
before any search has happened.

**2. The branching factor does not justify the machinery.** The action space is five tools, and at
any point in a real run one or two are sensible: if the draft is unscored, score it; if it is
scored, revise the highest-headroom criterion. The interesting variation is not *which action* but
*what instruction to give the reviser* — which the shallow ToT branch in §6.4 already explores, at
one level, for a fraction of the cost.

**3. The economics are wrong for the deliverable.** LATS runs tens to hundreds of model calls per
problem. This project targets free-tier quotas (see the [README](../README.md)); a single LATS run
would exhaust a day's Gemini allowance. A pattern you cannot afford to run is not a pattern you have
implemented.

**When it would be right.** If the rubric were adversarial — where improving `evidence` reliably
damaged `style_clarity`, so the optimal path required accepting a local loss for a later gain — the
greedy hill-climb here would get stuck and a lookahead search would earn its cost. The rubrics
shipped with this project are close to separable, so it does not.

---

## 7. Comparison

| | ReAct | Reflexion | CoT | ToT | LATS |
|---|---|---|---|---|---|
| Structure | chain | chain + memory | chain | tree | tree + MCTS |
| External environment | ✅ | ✅ | ❌ | ❌ | ✅ |
| Backtracking | ❌ | ❌ | ❌ | ✅ | ✅ |
| Learns across attempts | ❌ | ✅ | ❌ | ❌ | ✅ |
| Model calls / decision | 1 | 1 + 1 | 1 | k × depth | 10–100+ |
| Needs a resettable env | no | no | no | no | **yes** |
| Used here | ✅ core | ✅ core | ✅ prompts | ⚠️ 1 level | ❌ |

---

## 8. Honest limitations

- **The judge is the weak link.** Every pattern used here inherits the evaluator's quality, and the
  evaluator is an LLM scoring prose. Temperature 0, level descriptors, forced evidence quotes and
  deterministic probes all narrow the variance, but they do not eliminate it. Milestone 3's plateau
  detection partly exists because a 1-point "improvement" may well be judge noise. A stronger design
  would score each draft *n* times and take the median — a direct application of self-consistency
  (§3) — which was left out on cost grounds and is the first thing I would add with more budget.
- **The scratchpad is truncated** to the last four steps to bound prompt growth. On a long run the
  agent can forget an early failed strategy. Memory partly compensates, but only for findings that
  got promoted to a lesson.
- **The shallow-ToT branch is unverified at scale.** It is tested for correct selection behaviour,
  not for whether it produces materially better text than N=1 on real drafts. Claiming a quality
  gain would need an evaluation I have not run.
- **Reflexion lessons are never retired.** A lesson learned against one rubric persists. If it turns
  out to be wrong, nothing currently removes it — there is a `clear_session` operation but no
  mechanism for down-weighting a lesson that stops paying off.

---

## References

1. Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
2. Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
3. Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
4. Kojima et al. (2022). *Large Language Models are Zero-Shot Reasoners.* NeurIPS. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
5. Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
6. Yao et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
7. Zhou et al. (2024). *Language Agent Tree Search Unifies Reasoning, Acting and Planning in Language Models.* ICML. [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)
