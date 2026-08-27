/* ---------------------------------------------------------------------------
   Agentic Rubric Loop - application

   No framework and no build step, for the same reason the server is stdlib
   http.server: the whole thing has to run from a fresh checkout with one
   command. State lives in one object, rendering is one function per tab, and
   the run is an NDJSON stream read incrementally so the four steps appear as
   they happen rather than two minutes later in a lump.
--------------------------------------------------------------------------- */
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html != null) node.innerHTML = html;
  return node;
};
const esc = (s) =>
  String(s == null ? "" : s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
const pct = (v) => (typeof v === "number" ? v.toFixed(1) + "%" : "—");
const num = (v) => (typeof v === "number" ? v.toLocaleString() : "—");

/* --- state --------------------------------------------------------------- */

const S = {
  boot: null,
  rubric: null,
  running: false,
  target: 85,
  iterations: [],       // [{n, steps:[...]}]
  notes: [],
  harnessEvents: [],
  traceRows: [],
  result: null,
  stages: [],
  stageIndex: 0,
  memory: null,
};

/* --- boot ---------------------------------------------------------------- */

async function boot() {
  let data;
  try {
    data = await (await fetch("/api/bootstrap")).json();
  } catch (err) {
    document.querySelector(".main").prepend(
      el("div", "banner", `<b>Cannot reach the server.</b> ${esc(err)}`)
    );
    return;
  }
  S.boot = data;
  S.memory = data.memory;

  fill($("rubric"), data.rubrics.map((r) => ({ v: r.id, t: r.name })));
  fill($("provider"), data.providers.map((p) => ({
    v: p.name,
    t: `${p.name} — ${p.model}${p.available ? "" : "  (unavailable)"}`,
    disabled: !p.available,
  })));
  fill($("failure"), [{ v: "", t: "none" }].concat(
    data.failures.map((f) => ({ v: f, t: f.replaceAll("_", " ") }))
  ));
  fill($("failStep"), data.failure_steps.map((s) => ({ v: s, t: s })));

  const d = data.defaults;
  $("target").value = d.target_score;
  $("iters").value = d.max_iterations;
  $("temperature").value = d.temperature;
  $("budget").value = d.token_budget;
  $("candidates").value = String(d.revise_candidates);
  $("memoryOn").checked = d.memory_enabled;
  if (d.provider) $("provider").value = d.provider;
  $("failStep").value = "judge";
  S.target = d.target_score;

  renderProviderPill();
  renderMemoryPill();
  onRubricChange();
  renderMemoryTab();
  renderSetupTab();
  wire();

  if (!data.ready) {
    $("run").disabled = true;
    $("runHint").textContent = "No live provider is usable — see the Setup tab.";
    $("pane-run").prepend(
      el("div", "banner",
        "<b>No live provider is configured.</b> This application only runs against a real " +
        "model — there is no simulated mode. Put <code>GROQ_API_KEY</code> in " +
        "<code>.env</code> (or start <code>ollama serve</code>), then reload. " +
        "The <b>Setup</b> tab shows exactly which link in the chain is missing.")
    );
  }
}

function fill(select, options) {
  select.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o.v;
    opt.textContent = o.t;
    if (o.disabled) opt.disabled = true;
    select.appendChild(opt);
  }
}

/* --- wiring -------------------------------------------------------------- */

function wire() {
  $("rubric").addEventListener("change", onRubricChange);
  $("sample").addEventListener("change", onSampleChange);
  $("text").addEventListener("input", countWords);
  $("provider").addEventListener("change", renderProviderPill);
  $("failure").addEventListener("change", () => {
    const kind = $("failure").value;
    const perStep = ["rate_limit", "server_error", "bad_json", "provider_down"].includes(kind);
    $("failStepField").style.display = perStep ? "" : "none";
  });
  $("failure").dispatchEvent(new Event("change"));

  $("run").addEventListener("click", startRun);
  $("clearSession").addEventListener("click", () => clearMemory($("session").value.trim()));
  $("clearAll").addEventListener("click", () => {
    if (confirm("Delete every memory record, in every session?")) clearMemory("");
  });

  for (const button of $("tabs").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      for (const b of $("tabs").querySelectorAll("button")) b.classList.remove("on");
      for (const p of document.querySelectorAll(".pane")) p.classList.remove("on");
      button.classList.add("on");
      $("pane-" + button.dataset.tab).classList.add("on");
    });
  }
}

function onRubricChange() {
  const id = $("rubric").value;
  S.rubric = S.boot.rubrics.find((r) => r.id === id) || null;
  const samples = (S.boot.samples || {})[id] || [];
  fill($("sample"), [{ v: "", t: samples.length ? "— choose a preset —" : "none" }]
    .concat(samples.map((s) => ({ v: s.id, t: s.label }))));
  if (samples.length) {
    $("sample").value = samples[0].id;
    $("text").value = samples[0].text;
  } else {
    $("text").value = "";
  }
  if (S.rubric && S.rubric.target_score) {
    $("target").value = S.rubric.target_score;
    S.target = S.rubric.target_score;
  }
  countWords();
  renderSetupTab();
}

function onSampleChange() {
  const samples = (S.boot.samples || {})[$("rubric").value] || [];
  const found = samples.find((s) => s.id === $("sample").value);
  if (found) $("text").value = found.text;
  countWords();
}

function countWords() {
  const words = $("text").value.trim().split(/\s+/).filter(Boolean).length;
  $("wordcount").textContent = `${words} word${words === 1 ? "" : "s"}`;
}

/* --- header pills -------------------------------------------------------- */

function renderProviderPill() {
  const row = (S.boot.providers || []).find((p) => p.name === $("provider").value);
  const pill = $("providerPill");
  if (!row) {
    setPill(pill, "bad", "no provider");
    return;
  }
  setPill(pill, row.available ? "ok" : "bad", `${row.name}:${row.model}`);
  $("providerHint").textContent = row.available
    ? `Live. Failover chain position ${row.position}.`
    : `Unavailable — ${row.reason}`;
}

function renderMemoryPill() {
  const stats = (S.memory && S.memory.stats) || {};
  const total = stats.total || 0;
  setPill($("memoryPill"), total ? "mem" : "", `memory ${total}`);
}

function setPill(pill, cls, text) {
  pill.className = "pill " + (cls || "");
  pill.innerHTML = `<i class="dot"></i><span>${esc(text)}</span>`;
}

/* --- the run ------------------------------------------------------------- */

async function startRun() {
  if (S.running) return;
  const text = $("text").value.trim();
  if (!text) { alert("There is no text to work on."); return; }

  S.running = true;
  S.iterations = [];
  S.notes = [];
  S.harnessEvents = [];
  S.traceRows = [];
  S.result = null;
  S.stages = [];
  S.stageIndex = 0;
  S.target = Number($("target").value);

  $("run").disabled = true;
  $("run").textContent = "Running…";
  $("runEmpty").style.display = "none";
  $("timeline").innerHTML = "";
  $("notes").innerHTML = "";
  $("summaryTop").innerHTML = "";
  $("runHint").textContent = "A live run makes ~3 model calls per iteration.";
  setPill($("statusPill"), "busy", "running");
  counts();

  const body = {
    rubric_id: $("rubric").value,
    text,
    provider: $("provider").value,
    target_score: Number($("target").value),
    max_iterations: Number($("iters").value),
    revise_candidates: Number($("candidates").value),
    temperature: Number($("temperature").value),
    token_budget: Number($("budget").value),
    memory_enabled: $("memoryOn").checked,
    session_id: $("session").value.trim() || "app",
    simulate_failure: $("failure").value,
    fail_step: $("failStep").value,
  };

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await readStream(response.body.getReader());
  } catch (err) {
    onEvent("error", { message: String(err) });
  } finally {
    S.running = false;
    $("run").disabled = false;
    $("run").textContent = "Run the loop";
    $("runHint").textContent = "";
  }
}

async function readStream(reader) {
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, cut).trim();
      buffer = buffer.slice(cut + 1);
      if (!line) continue;
      try {
        const message = JSON.parse(line);
        onEvent(message.event, message.data || {});
      } catch { /* a partial line; the next chunk completes it */ }
    }
  }
}

function onEvent(name, data) {
  S.traceRows.push({ event: name, ...data });

  switch (name) {
    case "note":
      S.notes.push({ text: data.message, bad: false });
      renderNotes();
      break;

    case "error":
      setPill($("statusPill"), "bad", "failed");
      S.notes.push({ text: data.message + (data.hint ? " — " + data.hint : ""), bad: true });
      renderNotes();
      break;

    case "run_start":
      S.target = data.target_score || S.target;
      $("summaryTop").innerHTML = runHeader(data);
      break;

    case "iteration_start":
      S.iterations.push({ n: data.iteration, steps: [] });
      renderTimeline();
      break;

    case "perceive": case "reason": case "act": case "reflect":
      pushStep(name, data);
      break;

    case "retry": case "repair": case "failover":
    case "tool_recovery": case "guardrail_trip": case "budget_warning": case "guardrail":
      S.harnessEvents.push({ name, ...data });
      pushStep("harness", { kind: name, ...data });
      renderHarnessTab();
      break;

    case "complete":
      S.result = data;
      S.stages = data.stages || [];
      S.stageIndex = Math.max(0, S.stages.length - 1);
      S.memory = data.memory || S.memory;
      setPill($("statusPill"), data.status === "target_reached" ? "ok" : "warn",
              String(data.status || "done").replaceAll("_", " "));
      renderMemoryPill();
      $("summaryTop").innerHTML = summaryCard(data);
      renderTextTab();
      renderScoresTab();
      renderMemoryTab();
      renderHarnessTab();
      renderTraceTab();
      break;
  }
  counts();
}

function pushStep(kind, data) {
  if (!S.iterations.length) S.iterations.push({ n: data.iteration || 1, steps: [] });
  S.iterations[S.iterations.length - 1].steps.push({ kind, data });
  renderTimeline();
}

function counts() {
  $("cStages").textContent = S.stages.length;
  $("cScores").textContent = S.result ? (S.result.scorecards || []).length : 0;
  $("cMemory").textContent = (S.memory && S.memory.stats && S.memory.stats.total) || 0;
  $("cHarness").textContent = S.harnessEvents.length;
  $("cTrace").textContent = S.traceRows.length;
}

/* --- run tab ------------------------------------------------------------- */

function runHeader(d) {
  return `<div class="card"><div class="body stats">
    ${stat("", esc(d.run_id || "").slice(0, 16), "run id")}
    ${stat("", esc(d.session_id || ""), "session")}
    ${stat("accent", esc(d.provider || ""), "provider")}
    ${stat("", pct(d.target_score), "target")}
    ${stat("", d.max_iterations, "max iterations")}
  </div></div>`;
}

function stat(cls, value, label) {
  return `<div class="stat ${cls}"><b class="wrap">${value}</b><span>${esc(label)}</span></div>`;
}

function renderNotes() {
  $("notes").innerHTML = S.notes
    .map((n) => `<div class="note ${n.bad ? "bad" : ""}">${esc(n.text)}</div>`)
    .join("");
}

function renderTimeline() {
  $("timeline").innerHTML = S.iterations.map(iterationCard).join("");
}

function iterationCard(iter) {
  return `<div class="iter">
    <header><b>Iteration ${iter.n}</b></header>
    <div class="steps">${iter.steps.map((s) => stepRow(s.kind, s.data)).join("")}</div>
  </div>`;
}

function stepRow(kind, d) {
  const body = {
    perceive: perceiveStep, reason: reasonStep, act: actStep,
    reflect: reflectStep, harness: harnessStep,
  }[kind](d);
  return `<div class="step ${kind}"><div class="tag">${kind.toUpperCase()}</div>
    <div class="content">${body}</div></div>`;
}

function perceiveStep(d) {
  let html = `<div class="headline">${scoreLine(d.score)}</div>
    <div class="kv">words ${num(d.words)} &middot; flesch ${d.flesch ?? "—"}
      &middot; failing probes ${(d.failing_probes || []).length}
      &middot; recalled ${d.recalled ?? 0}</div>`;
  const items = d.recalled_items || [];
  if (items.length) {
    html += `<div class="recall"><div class="head">RECALLED FROM MEMORY</div><ul>` +
      items.map((h) =>
        `<li><span class="chip mem">${esc(h.kind)}</span>
         ${esc(h.content)}
         <span class="dim mono">(${esc((h.session_id || "").slice(0, 10))}, iter ${h.iteration},
         rel ${Number(h.score || 0).toFixed(2)})</span></li>`).join("") +
      `</ul></div>`;
  }
  for (const note of d.notes || []) html += `<div class="detail dim">! ${esc(note)}</div>`;
  return html;
}

function reasonStep(d) {
  const flag = d.degraded ? ` <span class="chip bad">degraded fallback</span>` : "";
  const args = d.arguments && Object.keys(d.arguments).length
    ? `<div class="kv">${esc(JSON.stringify(d.arguments))}</div>` : "";
  return `<div class="headline"><span class="chip accent">${esc(d.action)}</span>${flag}
      <span class="dim mono">${num(d.tokens)} tok</span></div>
    ${d.thought ? `<div class="quote">${esc(d.thought)}</div>` : ""}${args}`;
}

function actStep(d) {
  const ok = d.ok;
  const badge = ok ? `<span class="chip ok">ok</span>` : `<span class="chip bad">failed</span>`;
  const recovered = d.recovered ? ` <span class="chip ok">recovered</span>` : "";
  const stage = d.draft_stage != null
    ? ` <span class="chip mem">draft stage ${d.draft_stage} — ${num(d.draft_words)} words</span>`
    : "";
  return `<div class="headline"><span class="chip">${esc(d.action)}</span>${badge}${recovered}${stage}
      <span class="dim mono">${Math.round(d.duration_ms || 0)} ms &middot; ${num(d.tokens)} tok</span></div>
    <div class="detail">${esc(ok ? d.summary : d.error)}</div>`;
}

function reflectStep(d) {
  const delta = typeof d.score_delta === "number"
    ? `<span class="chip ${d.score_delta > 0 ? "ok" : "bad"}">${d.score_delta > 0 ? "+" : ""}${d.score_delta.toFixed(1)} pts</span>` : "";
  const plateau = d.plateau ? `<span class="chip bad">plateau</span>` : "";
  const ruled = d.degraded ? `<span class="chip">rule-based</span>` : "";
  let html = `<div class="headline">
      <span class="chip ${d.task_complete ? "ok" : ""}">complete=${d.task_complete}</span>
      ${delta}${plateau}${ruled}</div>
    <div class="detail">${esc(d.reason || "")}</div>`;
  if (d.critique) html += `<div class="quote">${esc(d.critique)}</div>`;
  if (d.lesson) {
    html += `<div class="recall"><div class="head">LESSON STORED</div>
      <div style="font-size:12px;color:#d6c9f5">${esc(d.lesson)}</div></div>`;
  }
  if (d.next_focus) html += `<div class="kv">next focus: ${esc(d.next_focus)}</div>`;
  return html;
}

function harnessStep(d) {
  const text = {
    retry: () => `retry ${d.attempt} on ${d.provider} after ${d.delay_s}s ` +
      `(${d.honoured_retry_after ? "Retry-After honoured" : "computed backoff"}): ${d.error_type}`,
    repair: () => d.method === "local_salvage"
      ? `unusable reply from ${d.provider}: salvaged the tool call locally, no extra call`
      : `unusable reply from ${d.provider}: sent one repair prompt`,
    failover: () => `failover ${d["from"]} → ${d.to || "nothing left in the chain"} — ${d.reason || ""}`,
    tool_recovery: () => `${d.action} failed — ${d.method}` +
      (d.retry_as ? ` → ${d.retry_as}` : " (handed back to the agent as an observation)"),
    guardrail_trip: () => `[${d.guardrail}] ${d.detail}`,
    budget_warning: () => d.reason,
    guardrail: () => `STOP (${d.status}) — ${d.reason}`,
  }[d.kind];
  return `<div class="headline"><span class="chip">${esc(d.kind)}</span></div>
    <div class="detail">${esc(text ? text() : JSON.stringify(d))}</div>`;
}

function scoreLine(score) {
  const value = typeof score === "number" ? score : null;
  const hit = value != null && value >= S.target;
  return `<div class="scoreline" style="flex:1">
    <div class="bar">
      <i class="${hit ? "hit" : ""}" style="width:${value == null ? 0 : Math.min(100, value)}%"></i>
      <u style="left:${S.target}%"></u>
    </div>
    <span class="num">${value == null ? "unscored" : pct(value)}</span></div>`;
}

function summaryCard(d) {
  const h = d.harness || {};
  const improved = typeof d.improvement === "number" ? d.improvement : null;
  return `<div class="card"><header>Run complete</header><div class="body">
    <div class="stats">
      ${stat(d.status === "target_reached" ? "ok" : "warn",
             esc(String(d.status).replaceAll("_", " ")), "status")}
      ${stat("", d.iterations, "iterations")}
      ${stat("accent", pct(d.best_score), "best score")}
      ${stat(improved > 0 ? "ok" : "", improved == null ? "—" :
             (improved > 0 ? "+" : "") + improved.toFixed(1), "points gained")}
      ${stat("", num((d.tokens || {}).total), "tokens")}
      ${stat("", (d.elapsed_s || 0).toFixed(1) + "s", "elapsed")}
    </div>
    <div style="margin-top:11px" class="scoreline">
      <span class="dim mono">${pct(d.initial_score)}</span>
      ${scoreLine(d.best_score)}
    </div>
    <div class="kv" style="margin-top:9px">trajectory:
      ${(d.score_trajectory || []).map((v) => v.toFixed(1) + "%").join(" → ") || "never scored"}</div>
    <div class="kv">harness: retries ${h.retries || 0} &middot; repairs ${h.repairs || 0}
      &middot; failovers ${h.failovers || 0} &middot; tool recoveries ${h.tool_recoveries || 0}
      ${h.degraded_memory ? '&middot; <span class="chip bad">memory degraded</span>' : ""}</div>
  </div></div>`;
}

/* --- text tab: the transformation ---------------------------------------- */

function renderTextTab() {
  const pane = $("pane-text");
  if (!S.stages.length) return;

  const bar = S.stages.map((s, i) =>
    `<button class="stagebtn ${i === S.stageIndex ? "on" : ""}" data-stage="${i}">
       <b>${esc(s.label)}</b>
       <span>${num(s.words)} words${s.iteration ? " · iter " + s.iteration : ""}</span>
     </button>`).join("");

  const stage = S.stages[S.stageIndex];
  const previous = S.stageIndex > 0 ? S.stages[S.stageIndex - 1] : null;

  let compare = "";
  if (previous) {
    compare = `<div class="card"><header>What changed at this stage
        <span class="chip">${esc(previous.label)} → ${esc(stage.label)}</span></header>
      <div class="body">
        <div class="diffkey"><span><em class="add">green</em> added</span>
          <span><em class="rem">red</em> removed</span>
          <span class="dim">word-level</span></div>
        <div class="doc">${diffHtml(previous.text, stage.text)}</div>
      </div></div>`;
  }

  const focus = (stage.focus || []).length
    ? `<span class="chip accent">targeting ${stage.focus.map(esc).join(", ")}</span>` : "";

  pane.innerHTML = `
    <div class="stagebar">${bar}</div>
    <div class="card"><header>${esc(stage.label)} ${focus}
        <span class="spacer"></span>
        <span class="chip">${num(stage.words)} words</span></header>
      <div class="body">
        ${stage.summary ? `<div class="note">${esc(stage.summary)}</div>` : ""}
        <div class="doc">${esc(stage.text)}</div>
      </div></div>
    ${compare}
    ${beforeAfter()}`;

  for (const button of pane.querySelectorAll(".stagebtn")) {
    button.addEventListener("click", () => {
      S.stageIndex = Number(button.dataset.stage);
      renderTextTab();
    });
  }
}

function beforeAfter() {
  if (!S.result) return "";
  return `<div class="card"><header>Original vs the draft the run returned</header>
    <div class="body"><div class="textgrid">
      <div><div class="kv" style="margin-bottom:6px">INPUT
        &middot; ${pct(S.result.initial_score)}
        &middot; ${num((S.result.initial_draft || "").split(/\s+/).filter(Boolean).length)} words</div>
        <div class="doc">${esc(S.result.initial_draft)}</div></div>
      <div><div class="kv" style="margin-bottom:6px">BEST DRAFT
        &middot; ${pct(S.result.best_score)}
        &middot; ${num((S.result.best_draft || "").split(/\s+/).filter(Boolean).length)} words</div>
        <div class="doc">${esc(S.result.best_draft)}</div></div>
    </div></div></div>`;
}

/* A word-level diff, longest-common-subsequence over tokens.
   Line diffs are useless here: a revision rewrites sentences in place, so a
   line-level view marks every paragraph as wholly changed and shows nothing. */
function diffHtml(before, after) {
  const a = tokenize(before), b = tokenize(after);
  const n = a.length, m = b.length;
  // Guard the O(n*m) table. Whole essays are a few hundred tokens; a pasted
  // book is not, and a hung tab is worse than a coarse diff.
  if (n * m > 4_000_000) {
    return `<del>${esc(before)}</del>\n\n<ins>${esc(after)}</ins>`;
  }
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1
                                : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  let out = "", i = 0, j = 0;
  const flush = (tag, text) => { if (text) out += `<${tag}>${esc(text)}</${tag}>`; };
  let pendingDel = "", pendingIns = "";
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      flush("del", pendingDel); flush("ins", pendingIns);
      pendingDel = pendingIns = "";
      out += esc(a[i]); i++; j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      pendingDel += a[i++];
    } else {
      pendingIns += b[j++];
    }
  }
  while (i < n) pendingDel += a[i++];
  while (j < m) pendingIns += b[j++];
  flush("del", pendingDel); flush("ins", pendingIns);
  return out;
}

/* Keep whitespace attached to its word so the reassembled text still reads. */
function tokenize(text) {
  return String(text || "").match(/\s*\S+|\s+/g) || [];
}

/* --- scores tab ---------------------------------------------------------- */

function renderScoresTab() {
  const pane = $("pane-scores");
  const cards = (S.result && S.result.scorecards) || [];
  if (!cards.length) {
    pane.innerHTML = `<div class="empty">This run recorded no scorecard.</div>`;
    return;
  }
  const byId = Object.fromEntries((S.rubric ? S.rubric.criteria : []).map((c) => [c.id, c]));
  const last = cards[cards.length - 1];

  const trend = `<div class="card"><header>Every scoring, in order</header><div class="body">
    <table><thead><tr><th>#</th><th>Weighted</th>
      ${(last.scores || []).map((s) => `<th class="right">${esc((byId[s.criterion_id] || {}).name || s.criterion_id)}</th>`).join("")}
    </tr></thead><tbody>
      ${cards.map((card, index) => `<tr>
        <td class="mono">${index + 1}</td>
        <td class="num"><b>${pct(card.weighted_percent)}</b></td>
        ${(card.scores || []).map((s) => `<td class="num">${s.score}</td>`).join("")}
      </tr>`).join("")}
    </tbody></table></div></div>`;

  const detail = `<div class="card"><header>Final scorecard, with the judge's evidence</header>
    <div class="body">${(last.scores || []).map((s) => {
      const c = byId[s.criterion_id] || {};
      const max = (S.rubric && S.rubric.scale.max) || 5;
      return `<div style="margin-bottom:13px">
        <div class="headline" style="display:flex;align-items:center;gap:9px">
          <b>${esc(c.name || s.criterion_id)}</b>
          <span class="chip">weight ${((c.weight || 0) * 100).toFixed(0)}%</span>
          <span class="spacer" style="flex:1"></span>
          <span class="mono">${s.score}/${max}</span>
        </div>
        <div class="bar" style="margin:6px 0"><i style="width:${(s.score / max) * 100}%"></i></div>
        ${s.evidence ? `<div class="quote">&ldquo;${esc(s.evidence)}&rdquo;</div>` : ""}
        ${s.justification ? `<div class="detail dim">${esc(s.justification)}</div>` : ""}
      </div>`;
    }).join("")}</div></div>`;

  pane.innerHTML = trend + detail;
}

/* --- memory tab ---------------------------------------------------------- */

function renderMemoryTab() {
  const pane = $("pane-memory");
  const memory = S.memory || {};
  const stats = memory.stats || {};
  const writes = (S.result && S.result.memory_writes) || [];

  pane.innerHTML = `
    <div class="card"><header>Store</header><div class="body">
      <div class="stats">
        ${stat("", num(stats.total || 0), "records")}
        ${stat("", num((stats.by_kind || {}).lesson || 0), "lessons")}
        ${stat("", num((stats.by_kind || {}).episodic || 0), "episodes")}
        ${stat("", num(stats.sessions || (memory.sessions || []).length), "sessions")}
        ${stat(stats.vector_enabled ? "ok" : "warn",
               stats.vector_enabled ? "vector" : "keyword", "recall mode")}
        ${stat(stats.degraded ? "bad" : "ok", stats.degraded ? "degraded" : "healthy", "health")}
      </div>
      <div class="kv" style="margin-top:9px">${esc(stats.embedder || "")}
        ${stats.vector_dimension ? "&middot; " + stats.vector_dimension + "d" : ""}</div>
      ${(memory.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("")}
    </div></div>

    ${writes.length ? `<div class="card"><header>Written by this run
        <span class="chip">${writes.length}</span></header><div class="body">
      <table><thead><tr><th>Iter</th><th>Kind</th><th>Content</th><th class="right">&Delta;</th></tr></thead>
      <tbody>${writes.map((w) => `<tr>
        <td class="mono">${w.iteration}</td>
        <td><span class="chip ${w.kind === "lesson" ? "mem" : ""}">${esc(w.kind)}</span></td>
        <td>${esc(w.content)}</td>
        <td class="num">${typeof w.score_delta === "number" ? w.score_delta.toFixed(1) : "—"}</td>
      </tr>`).join("")}</tbody></table></div></div>` : ""}

    <div class="card"><header>Lessons in the store
      <span class="chip">${(memory.lessons || []).length}</span></header><div class="body">
      ${(memory.lessons || []).length ? `<table>
        <thead><tr><th>Lesson</th><th>Rubric</th><th>Session</th><th class="right">Hits</th></tr></thead>
        <tbody>${memory.lessons.map((l) => `<tr>
          <td>${esc(l.content)}</td>
          <td class="mono">${esc(l.rubric_id)}</td>
          <td class="mono">${esc((l.session_id || "").slice(0, 14))}</td>
          <td class="num">${l.hits}</td></tr>`).join("")}</tbody></table>`
        : `<div class="dim">Nothing stored yet. Run once, then change the session id and run
             again — the second run recalls what the first learned.</div>`}
    </div></div>

    <div class="card"><header>Sessions</header><div class="body">
      ${(memory.sessions || []).length
        ? memory.sessions.map((s) => `<span class="chip" style="margin:2px">${esc(s)}</span>`).join("")
        : `<span class="dim">none</span>`}
    </div></div>`;
  counts();
}

async function clearMemory(sessionId) {
  const response = await fetch("/api/memory/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  const data = await response.json();
  S.memory = await (await fetch("/api/memory")).json();
  renderMemoryTab();
  renderMemoryPill();
  alert(`Removed ${data.removed} record(s) from ${data.session_id === "*" ? "every session" : data.session_id}.`);
}

/* --- harness tab --------------------------------------------------------- */

function renderHarnessTab() {
  const pane = $("pane-harness");
  const result = S.result;
  const h = (result && result.harness) || {};
  const g = (result && result.guardrails) || {};
  const tokens = g.tokens || {};
  const used = tokens.used || 0, budget = tokens.budget || 0;
  const ratio = budget ? Math.min(100, (used / budget) * 100) : 0;

  const gauge = budget ? `<div class="card"><header>Token budget</header><div class="body">
      <div class="scoreline"><span class="mono">${num(used)} / ${num(budget)}</span>
        <span class="spacer" style="flex:1"></span>
        <span class="mono">${ratio.toFixed(0)}%</span></div>
      <div class="gauge"><i class="${ratio > 95 ? "bad" : ratio > 80 ? "warn" : ""}"
        style="width:${ratio}%"></i></div>
      <div class="kv">wall clock ${(g.wall_clock_s || {}).used ?? "—"}s of
        ${(g.wall_clock_s || {}).limit ?? "—"}s &middot;
        iterations ${(g.iterations || {}).used ?? "—"} of ${(g.iterations || {}).cap ?? "—"}</div>
      ${(g.triggered || []).length
        ? `<div class="note harness">tripped: ${g.triggered.map(esc).join(", ")}</div>` : ""}
    </div></div>` : "";

  const events = S.harnessEvents.length ? `<div class="card">
      <header>What the harness had to do <span class="chip">${S.harnessEvents.length}</span></header>
      <div class="body">${S.harnessEvents.map((e) =>
        `<div class="note harness"><b>${esc(e.name)}</b> — ${esc(summariseHarness(e))}</div>`
      ).join("")}</div></div>`
    : `<div class="card"><header>What the harness had to do</header><div class="body">
        <span class="dim">Nothing. Every call succeeded first time.</span></div></div>`;

  pane.innerHTML = `
    <div class="card"><header>Recovery</header><div class="body"><div class="stats">
      ${stat(h.retries ? "warn" : "ok", h.retries || 0, "retries")}
      ${stat(h.repairs ? "warn" : "ok", h.repairs || 0, "repairs")}
      ${stat(h.failovers ? "warn" : "ok", h.failovers || 0, "failovers")}
      ${stat(h.tool_recoveries ? "warn" : "ok", h.tool_recoveries || 0, "tool recoveries")}
      ${stat(h.degraded_memory ? "bad" : "ok", h.degraded_memory ? "yes" : "no", "memory degraded")}
    </div>
    <div class="kv" style="margin-top:9px">provider: ${esc(h.provider || "—")}</div>
    ${h.trace ? `<div class="kv wrap">trace: ${esc(h.trace)}</div>` : ""}
    </div></div>
    ${gauge}${events}`;
}

function summariseHarness(e) {
  if (e.name === "retry") {
    return `attempt ${e.attempt} on ${e.provider}, waited ${e.delay_s}s ` +
      `(${e.honoured_retry_after ? "Retry-After" : "computed backoff"}) after ${e.error_type}`;
  }
  if (e.name === "failover") return `${e["from"]} → ${e.to || "chain exhausted"}: ${e.reason || ""}`;
  if (e.name === "repair") return `${e.provider}: ${e.method}`;
  if (e.name === "tool_recovery") return `${e.action}: ${e.method}`;
  if (e.name === "guardrail_trip") return `${e.guardrail}: ${e.detail}`;
  return e.reason || e.detail || JSON.stringify(e);
}

/* --- trace tab ----------------------------------------------------------- */

function renderTraceTab() {
  const rows = S.traceRows;
  $("pane-trace").innerHTML = `<div class="card">
    <header>Event stream <span class="chip">${rows.length} events</span>
      ${S.result && S.result.trace_path
        ? `<span class="spacer" style="flex:1"></span>
           <span class="chip wrap">${esc(S.result.trace_path)}</span>` : ""}</header>
    <div class="body"><table>
      <thead><tr><th>#</th><th>Iter</th><th>Event</th><th>Tool</th>
        <th class="right">ms</th><th class="right">Tokens</th><th>Detail</th></tr></thead>
      <tbody>${rows.map((r, i) => `<tr>
        <td class="mono dim">${i + 1}</td>
        <td class="mono">${r.iteration ?? ""}</td>
        <td><span class="chip ${r.event === "error" ? "bad" : ""}">${esc(r.event)}</span></td>
        <td class="mono">${esc(r.action || r.tool || "")}</td>
        <td class="num">${r.duration_ms != null ? Math.round(r.duration_ms) : ""}</td>
        <td class="num">${r.tokens != null ? num(r.tokens) : ""}</td>
        <td class="dim wrap">${esc(traceDetail(r)).slice(0, 160)}</td>
      </tr>`).join("")}</tbody></table></div></div>`;
}

function traceDetail(r) {
  return r.thought || r.summary || r.error || r.reason || r.message || r.critique || "";
}

/* --- setup tab ----------------------------------------------------------- */

function renderSetupTab() {
  const d = (S.boot && S.boot.defaults) || {};
  const tools = ((S.boot && S.boot.tools) || {})[$("rubric").value] || [];
  const rubric = S.rubric;

  $("pane-setup").innerHTML = `
    <div class="card"><header>Provider chain</header><div class="body">
      <table><thead><tr><th>#</th><th>Provider</th><th>Model</th><th>Key</th><th>Status</th></tr></thead>
      <tbody>${(S.boot.providers || []).map((p) => `<tr>
        <td class="mono">${p.position}</td>
        <td class="mono">${esc(p.name)}</td>
        <td class="mono">${esc(p.model)}</td>
        <td class="mono dim">${esc(p.key_env || "—")}</td>
        <td><span class="chip ${p.available ? "ok" : "bad"}">${esc(p.reason)}</span></td>
      </tr>`).join("")}</tbody></table>
      <div class="hint">The first usable link serves the run; the rest are the failover
        chain, walked inside the run if the primary dies. There is no simulated provider
        here — this application always talks to a real model.</div>
    </div></div>

    ${rubric ? `<div class="card"><header>${esc(rubric.name)}
        <span class="chip">${esc(rubric.id)}</span></header><div class="body">
      <div class="detail dim" style="margin-bottom:9px">${esc(rubric.description)}</div>
      <table><thead><tr><th>Criterion</th><th class="right">Weight</th><th>Probes</th></tr></thead>
      <tbody>${rubric.criteria.map((c) => `<tr>
        <td><b>${esc(c.name)}</b><div class="dim" style="font-size:11.5px">${esc(c.description)}</div></td>
        <td class="num">${(c.weight * 100).toFixed(0)}%</td>
        <td class="dim" style="font-size:11.5px">${c.probes.map(esc).join("<br>") || "—"}</td>
      </tr>`).join("")}</tbody></table></div></div>` : ""}

    <div class="card"><header>Tools, generated from this rubric</header><div class="body">
      <table><thead><tr><th>Tool</th><th>LLM?</th><th>Arguments</th><th>What it does</th></tr></thead>
      <tbody>${tools.map((t) => `<tr>
        <td class="mono">${esc(t.name)}</td>
        <td><span class="chip ${t.uses_llm ? "accent" : "ok"}">${t.uses_llm ? "model" : "pure python"}</span></td>
        <td class="mono dim">${t.arguments.map(esc).join(", ") || "—"}</td>
        <td class="dim">${esc(t.description)}</td>
      </tr>`).join("")}</tbody></table>
      <div class="hint">Criterion arguments carry an <code>enum</code> of this rubric's real
        ids, so the model cannot name a criterion that does not exist.</div>
    </div></div>

    <div class="card"><header>Configuration in force</header><div class="body">
      <div class="stats">
        ${stat("", d.target_score, "target score")}
        ${stat("", d.max_iterations, "max iterations")}
        ${stat("", num(d.token_budget), "token budget")}
        ${stat("", (d.retry || {}).max_attempts, "retry attempts")}
        ${stat("", esc((d.retry || {}).jitter || ""), "jitter")}
        ${stat("", (d.retry || {}).tool_max_attempts, "tool attempts")}
        ${stat("", esc(d.memory_backend || ""), "memory backend")}
        ${stat("", d.max_lessons_per_recall, "lessons/recall")}
      </div>
      <div class="hint">Every one of these comes from <code>config/config.yaml</code> and is
        overridable by environment variable or CLI flag without touching loop code.</div>
    </div></div>`;
}

boot();
