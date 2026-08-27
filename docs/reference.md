# Agentic Rubric Loop — Milestone Notes

Use case: score text against a rubric and improve it.

---

## Milestone 1 — Agentic Reasoning Patterns

### ReAct

- Thinking and doing are interleaved. The model writes a short thought, picks an action, and gets
  an observation back from the environment before it thinks again.
- That observation comes from outside the model. It can therefore contradict whatever the model had
  talked itself into, which is the entire reason the pattern works.
- A trace looks like this: *Thought* — I do not know how good this draft is. *Action* —
  score it. *Observation* — 28.7%, weakest on thesis and evidence. *Thought* — those two carry the
  most weight, so edit there.
- Cost is low. One model call per decision, plus whatever the tool itself costs.
- The weakness is that ReAct never backtracks. A bad action at step three stays in the trajectory
  forever, and the agent can only reason forward from the mess it made.

### Reflexion

- Three parts sit around an acting loop: an actor that produces the trajectory, an evaluator that
  scores it, and a self-reflection step that turns that score into plain-language feedback about
  what went wrong.
- The reflection gets written to memory. On the next attempt it is pasted back into the prompt.
- No weights change. Only the prompt changes, which is why the paper calls it verbal reinforcement
  learning rather than fine-tuning.
- It is cheap. One extra call per episode buys a genuine improvement across attempts.
- Everything depends on the evaluator. Feed Reflexion a sycophantic judge and it will faithfully
  learn the wrong lesson, then carry that lesson into every future run.

### Chain-of-Thought

- Ask the model to write out intermediate steps before committing to an answer.
- Mechanically it buys more forward passes on a hard problem, and it makes the final answer depend
  on explicit intermediate results instead of one leap.
- This is a prompting technique, not an agent architecture. There are no actions, no environment
  and no memory in it. It layers underneath everything else.
- The failure mode is post-hoc rationalisation. Once a model has settled on an answer it will
  happily generate a chain that justifies it, which matters enormously if the thing you are asking
  it to produce is a score.

### Tree of Thoughts

- Instead of one chain, build a tree. Generate several candidate next steps, have the model score
  each one, expand the promising branches and prune the rest.
- Backtracking is possible here, which is the capability a chain structurally cannot have.
- On tasks like Game of 24 the difference is dramatic, because solving those problems genuinely
  requires trying a path and abandoning it.
- It is expensive. Branching multiplied by depth, plus an evaluation call at every node.
- Two conditions have to hold before it pays off: several plausible moves must exist at each step,
  and there has to be a cheap way of telling them apart. Without both, you spend ten times the
  tokens rediscovering the obvious answer.

### LATS

- Monte Carlo Tree Search laid over ReAct trajectories, with Reflexion supplying the signal that
  gets propagated back up the tree.
- Each node is a state reached by some sequence of thoughts and actions. Selection uses UCT,
  expansion samples candidate actions from the model, simulation runs those actions for real, and
  failures produce a written reflection stored for later expansions of that subtree.
- Because the actions are executed against a real environment, node values are grounded rather than
  imagined.
- It is the most capable of the five and by far the most expensive — tens to hundreds of calls for
  a single problem.
- One assumption is easy to miss: MCTS needs an environment it can reset. Explore a branch, throw
  it away, return to the parent state.

---

## Milestone 1 — Patterns Applied, and Why

### What the use case gives us

- Scoring a draft costs one model call and returns a number. That single fact decided most of the
  design.
- Patterns split into two groups. Some need a real external signal — ReAct, Reflexion, LATS.
  Others substitute reasoning where no signal exists, which is what CoT and Tree of Thoughts are
  for.
- We had a real signal. Paying for a pattern that simulates one would mean trusting the model's
  opinion of its own work when a measurement was sitting right there.

### ReAct is the backbone

- The four steps are ReAct with the phases named and pulled apart. Perceive structures the world,
  Reason produces the Thought and the Action, Act executes it, and the tool result is the
  Observation.
- The rubric score is a genuine external observation. When the judge comes back with 28.7% and
  evidence at 2 out of 5, that is a measurement the agent has to accept.
- Previous thought/action/observation triples are replayed into the prompt as a scratchpad, so a
  strategy the agent already tried shows up as a repeat rather than a fresh idea.
- One implementation detail is worth defending. The `thought` field is a required property on every
  tool schema instead of being scraped out of the assistant's free text. Several providers return
  empty content alongside a tool call, and scraping would have quietly reduced ReAct to plain
  action-selection whenever the provider felt like it.

### Reflexion is how the agent learns

- The mapping is one to one. Our actor is Perceive → Reason → Act. Our evaluator is the rubric
  judge. Our self-reflection is the LLM half of the Reflect step, and our episodic buffer is the
  memory module.
- Reflect is asked for a *transferable* lesson. "The second paragraph needed a source" is useless
  next session. "Unattributed statistics score no better than no statistics" is worth keeping
  forever.
- An empty lesson is accepted as an honest answer. A vague one like "be more specific" would
  pollute every future run, so we would rather store nothing.
- We deviate from the paper deliberately in one place. In Reflexion the actor's own judgement can
  end an episode. Here it cannot. Completion is computed from the numbers, and the model's opinion
  is recorded but only ever counts as one input to the plateau rule. An agent allowed to declare
  its own success will do exactly that, right at the point where the remaining work gets hard.

### Chain-of-Thought, used narrowly

- The judge's output schema orders each criterion as `evidence`, then `justification`, then
  `score`. Models fill JSON in schema order, so the quote and the argument get generated before the
  number.
- Swap those two fields and you get the post-hoc rationalisation problem. For a grader that is
  fatal. The loop steers on the difference between two scores, and a judge that drifts is
  indistinguishable from real progress.
- The judge also runs at temperature 0 for the same reason. Sampling noise in a measuring
  instrument reads as improvement.
- In the reasoning step, the `thought` field forces the agent to say why this action and why now
  before it names the action.

### Tree of Thoughts, one level deep and off by default

- Setting `revise_candidates` above 1 makes the reviser generate several drafts at different
  temperatures, score each with the judge, and keep the winner.
- That is a genuine one-level search tree with value-based pruning. Generator, evaluator, selection.
  No backtracking and no lookahead, and the documentation says so rather than dressing it up.
- Going deeper does not work here. Text revision has no meaningful intermediate state to branch on.
  A partial arithmetic expression is a node you can score; half a rewritten paragraph is not.
- It defaults to off because it roughly triples the token cost of an iteration for a modest gain.
  Exposing it as configuration puts that trade-off where it belongs, with whoever is running it.

### LATS, rejected

- The environment does not reset. Our state is a document being progressively rewritten, and there
  is no cheap way to undo that. Simulating a reset means keeping every intermediate draft and
  re-scoring on each backtrack, which is most of the cost of the search before any searching has
  happened.
- The branching factor is too small to justify the machinery. There are five tools, and at any
  given moment one or two make sense. If the draft is unscored, score it. If it is scored, revise
  the criterion with the most headroom.
- The economics are wrong. Hundreds of calls per problem against a free tier is not a pattern you
  have implemented, it is a pattern you have described.
- It would be the right choice if the rubric were adversarial — if improving evidence reliably
  damaged clarity, so the best path required accepting a loss first. Ours are close to separable.

---

## Milestone 2 — Memory

### Why the backend is SQLite

- Everything lives in one file. Records, the keyword index and the vectors all sit in
  `data/memory.db`, kept in sync by triggers.
- That was the deciding factor over Chroma, Qdrant or LanceDB. Two separate stores can disagree
  about what the agent remembers, and when they do the failure is silent — retrieval returns a
  vector id whose record was deleted, and nothing complains.
- One file also means one Docker volume, one backup, and one thing to delete when a demo needs a
  clean slate.
- It works offline. No service call at recall time, so a throttled free tier cannot break a
  demonstration.
- `sqlite3` is in the standard library. The vector half is `sqlite-vec`, a small loadable
  extension, and when it will not load the whole thing degrades to FTS5 keyword search over the
  same rows in the same file.

### Why fastembed, not sentence-transformers

- Quality is comparable at this scale. We are embedding a few hundred short strings, not a corpus.
- `sentence-transformers` drags in PyTorch. That takes the container image from roughly 400 MB to
  roughly 2.5 GB.
- `fastembed` runs ONNX on CPU. The model is `BAAI/bge-small-en-v1.5` at 384 dimensions, about
  130 MB on disk.

### How memory is structured

Three tiers with three different lifetimes:

- **Episodic** — what happened in one iteration of one session. Scoped to that session and useless
  outside it. Example: *iteration 2: revise_text, 202 to 243 words, similarity 0.95*.
- **Lesson** — a transferable finding produced by Reflect. Scoped to the rubric, but recalled
  across every session. This is the Reflexion payload and the reason memory exists at all.
- **Profile** — standing constraints for a rubric, such as tone or a target score.

Every row carries `kind`, `content`, a normalised content hash, `session_id`, `iteration`,
`rubric_id`, `criterion_id`, a `score_delta`, timestamps, and a `hits` counter.

Alongside the table sit two indexes. FTS5 handles keyword search; a `vec0` virtual table holds the
384-dimension embeddings with cosine distance. Triggers keep both in step with inserts, updates and
deletes, so the indexes cannot drift away from the rows they describe.

### The policy decisions that actually matter

- **The recall query describes the problem, not the text.** It is built from the rubric name and
  the criteria with the most headroom, never from the draft. Query with the essay and you retrieve
  memories about similar essays. Query with the problem and you retrieve memories about how to
  solve it, which is what the next decision needs.
- **Lessons are scoped to their rubric, not global.** Something learned about attributing
  statistics in essays is not evidence about bug reports. Recalling it there is worse than
  recalling nothing, because it spends prompt budget arguing for an irrelevant edit.
- **Lessons are ranked but not relevance-gated. Episodic records are.** This is the most
  consequential call in the module. Episodic memory is voluminous and noisy, so it has to clear a
  similarity threshold. Lessons are the opposite — a handful per rubric, each already filtered by
  Reflect deciding it was worth keeping. Gate them on cosine similarity and you throw away the
  point of Reflexion the first time somebody phrases a query differently. They are capped at three
  and always offered.
- **The keyword channel is ordinal, not calibrated, and says so.** BM25 magnitudes depend on corpus
  size. On a ten-row table they come back around 1e-6. Squashing that into something that looks
  like a probability would produce a number that appears absolute and is not, and the relevance
  gate would then silently reject everything. Keyword hits therefore score by rank, and in
  keyword-only mode the gate is skipped rather than applied to a fake number.
- **A relearned lesson increments a counter rather than duplicating.** A finding rediscovered
  independently in a later session is better evidence than a one-off, so `hits` feeds a small
  ranking boost. Episodic records never deduplicate, because two identical events in two sessions
  are two facts.

### Reading and writing

- Memory is read once at the start of every Perceive and written after every Reflect.
- Reading in Perceive rather than inside Reason means recall happens exactly once per iteration,
  shows up in the observation, and can fail without the reasoning step ever knowing.
- Two records get written per iteration: an episodic one always, and a lesson only when Reflect
  produced something worth keeping.

### A concrete example of memory changing the output

Same input, same target, same rubric. Three runs:

- **Run A** — memory on, cold store. Six iterations. Sequence: score, analyze_text, revise, score,
  revise, score.
- **Run B** — memory on, warm store, **a different session id**. Five iterations. Sequence: score,
  revise, score, revise, score.
- **Run C** — memory off, the control. Six iterations. Identical to A, action for action.

Run A wrote two lessons before it finished. One of them:

> On this rubric, targeting the two highest-weighted criteria first moves the total faster than
> fixing the lowest raw score.

Look at iteration 2 in each run.

- **A had nothing recalled.** Its thought: *"No prior experience with this rubric was recalled.
  Measure the draft directly before spending a revision on a guess."* It spent the turn on
  `analyze_text`.
- **B recalled the lesson above.** Its thought: *"Memory says: On this rubric, targeting the two
  highest-weighted criteria first moves the total faster than fixing the lowest raw score. Applying
  that directly rather than rediscovering it."* It went straight to `revise_text`.

One whole iteration saved, and the lesson reached the run by two separate paths — it appeared in
the Reason prompt under a `RECALLED FROM MEMORY` heading, and it was passed through to the reviser
as an `apply_lessons` argument so it influenced the rewrite itself.

The reason run C matters is that A and C behaving identically is the control. A cold store carries
exactly as much information as no store, so they had better produce the same actions. Without that
check, "memory helped" could just as easily mean "the second run of anything is faster".

---

## Milestone 3 — Engineering Decisions and the Failures They Defend Against

### Where the harness attaches

- **Decision:** the harness reaches the loop through exactly two seams — a controller that may stop
  a run at an iteration boundary, and a substitutable Act function.
- **Defends against:** infrastructure quietly becoming a fifth cognitive step. The controller
  protocol is deliberately narrow. It can stop the run and annotate it, and it cannot choose an
  action, edit the draft or change a score.
- There is no retry, backoff, jitter, failover, budget or timeout logic anywhere in `core/`. Both
  seams default to "no harness", so the loop still runs standalone and still reads as four steps.

### Retry

- **Decision:** exponential backoff with full jitter, an attempt cap, and `Retry-After` preferred
  over our computed delay but capped at 60 seconds.
- **Defends against:** rate limits, timeouts and transient 5xx responses. Also against
  synchronised retry storms, which is what the jitter is for.
- **Decision:** separate policies for transport and for tools. Four attempts at a 1-second base for
  the LLM; two attempts at 0.25 seconds for tool calls.
- **Defends against:** treating an API rate limit and a malformed tool argument as the same kind of
  problem. They deserve very different amounts of patience.
- **Decision:** the error taxonomy is split by what the caller should do, not by status code.
  Retryable, terminal, unparseable, unavailable.
- **Defends against:** a retry loop that hammers a provider over a bad API key. It also keeps the
  retry decorator small, because "back off" and "fail over" become genuinely different code paths
  instead of one function with a flag.

Live evidence: a single five-iteration run on Groq's free tier absorbed fifteen HTTP 429s, honoured
every `Retry-After`, and still finished at target. Backoff accounted for 139 seconds of a
170-second run.

### Fallbacks

- **Unparseable model output.** Forced tool schemas first, then local JSON salvage, then one repair
  call, then a safe read-only default action flagged as degraded. *Defends against:* a run dying
  because a provider returned prose where a tool call was required. The fallback deliberately never
  picks an action that rewrites the user's text — a decision we do not fully understand is not one
  that should be allowed to edit anything.
- **Failed tool call.** The failure comes back as a normal result object with `ok=False` and goes
  into the next observation. Then a sanitised retry that drops the arguments the schema rejected,
  then an alternative tool. *Defends against:* one bad argument ending a run. A broken tool call is
  data the agent can react to, not an exception.
- **Hitting the iteration cap.** Return the best-scoring draft ever seen, with a status that says
  what happened. *Defends against:* handing back text that is worse than the input. Revisions can
  make things worse, and an improvement agent that returns something worse than what it was given
  has failed at its only job.
- **Memory read failure.** A circuit breaker counts consecutive failures per operation and swaps in
  a no-op store once it trips. The run continues and reports itself as degraded. *Defends against:*
  paying the latency of a failing read on every single iteration, forever.
- **Token budget exhausted.** Warn at 80%, then force a graceful finalize. *Defends against:*
  burning a quota to no purpose and returning nothing for it.
- **Provider unavailable.** Walk the chain — Gemini, then Groq, then a local Ollama — and stick
  with whichever link answers. *Defends against:* one vendor's outage taking the whole run down.

Two distinctions in that ladder are worth pointing out. A 400 is raised, but a 404 fails over: a
malformed request will be malformed everywhere, whereas a missing model is only missing on that one
provider. Auth rejection also fails over rather than aborting, for the same reason.

### Observability

- **Decision:** every step of every iteration emits a structured event, written to
  `runs/<run_id>/trace.jsonl` as one JSON object per line and flushed per line.
- **Defends against:** losing the evidence when a run is killed. Flushing per line means a
  terminated process still leaves everything up to the moment it died.
- **Decision:** the same event stream drives the console renderer and the browser UI.
- **Defends against:** what you watch on screen drifting away from what the trace recorded. There
  is one source, so they cannot disagree.
- **Decision:** redaction happens in the log formatter, matching key names *and* value patterns.
- **Defends against:** a secret leaking because somebody logged it under a key nobody thought to
  add to the blocklist.
- Each event carries run id, session id, iteration, step, timing, token counts, a cost estimate and
  any error.

### Guardrails

- **Hard iteration cap.** Enforced in Python by the loop, never delegated to a prompt. *Defends
  against:* an agent that talks itself into one more turn, indefinitely.
- **Token budget.** Tracked across every call in a run, with a warning threshold before the stop.
  *Defends against:* an unattended run quietly consuming a day's quota.
- **Wall-clock deadline.** Checked at iteration boundaries. *Defends against:* a run that is not
  looping but is simply never going to finish.
- **Stuck detection, three signals.** The same action with the same arguments twice in a row, a
  draft that cycles A to B and back to A, or a score frozen across a window. *Defends against:* the
  agent burning its whole budget repeating one thing. This is not hypothetical — an early bug made
  every tool call fail identically, and the loop ran all six iterations and returned cleanly
  without noticing.
- Every stop is graceful. Status, reason, best draft, full trace. Never an exception.

### The admission gate

- **Decision:** check the submission against rubric-declared word and sentence floors before the
  loop starts, with no model call at all.
- **Defends against:** the worst failure this system had. Given the input "hi, good morning", the
  agent scored a three-word greeting as an essay, expanded it into 170 words with zero similarity
  to the input, scored its own invention at 85%, and reported success. Roughly 17,000 tokens went
  into fabricating a document the user never wrote.
- A refusal is a verdict rather than an error. It carries a run id, a reason written for a person,
  and the original text handed back untouched.
- The gate is deterministic on purpose. Asking a model whether something is gradeable costs a call,
  can be wrong, and can be argued out of its answer by the next prompt. Word counts cannot.

### The reviser guards

- **Decision:** four checks on every candidate revision — too short, too long, too little of the
  original vocabulary surviving, and no change at all.
- **Defends against:** the reviser replacing the draft instead of improving it. The guards used to
  be one-sided, catching only shrinkage and sameness, which is precisely how a 3-word input became
  a 170-word fabrication without anything objecting.
- The vocabulary check uses set overlap rather than `difflib`. Line-based similarity cannot tell a
  legitimate paragraph rewrite from an invention, because both score near zero. Vocabulary survival
  can, since a real revision keeps the subject matter even when every sentence changes.
- The threshold was set from live data, not guessed. Genuine deep rewrites of hedge-heavy prose
  measured 0.18 to 0.23. Wholesale replacement with unrelated content measured below 0.05. The
  threshold sits at 0.15.

### Configuration

- **Decision:** every runtime parameter lives in `config/config.yaml`, layered so that CLI flags
  beat environment variables, which beat the file, which beats the dataclass defaults.
- **Defends against:** the thing the brief explicitly asks about — having to edit loop code to
  change a model, an iteration cap, a token budget or a memory backend.
- Config is loaded into frozen, typed dataclasses with unknown-key validation, so a typo like
  `max_iteration` raises instead of being silently ignored.
- Nothing in `core/` imports `os` or reads YAML. It receives a finished config object, which is
  what makes the separation structural rather than aspirational.

### Fault injection

- **Decision:** seven injectable failure kinds, each injected at the layer where the real thing
  actually happens.
- **Defends against:** testing the test. Mocking the harness itself would prove nothing about
  whether the harness works.
- Three real bugs were found this way, and they shared a shape worth remembering. In each case the
  error containment that kept the run alive is exactly what hid the bug. All three were caught by
  tests asserting that behaviour *changed*, not by anything crashing.
