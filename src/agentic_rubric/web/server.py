"""The application server.

Built on ``http.server`` from the standard library, on purpose. An application
whose first step is "pip install a web framework" is one that fails on someone
else's machine, and nothing here needs more than routing and a streaming
response.

**Live providers only.** There is no mock path through this server. The
deterministic :class:`~..llm.mock.MockProvider` still backs the test suite --
that is what keeps 300-odd tests free of an API key -- but a run started from
this application always goes to a real model. A UI that can quietly show you
simulated scores is a UI that will eventually show you simulated scores while
you believe otherwise.

Every run goes through :class:`~..harness.runner.Runner`, not
:class:`~..core.loop.AgenticLoop` directly, so retries, repairs, failovers,
guardrail trips and the JSONL trace are all real and all visible.

Endpoints::

    GET  /api/bootstrap        rubrics, samples, provider chain, defaults, memory
    POST /api/run              one run, streamed as NDJSON, one line per event
    GET  /api/memory           stats, sessions, stored lessons
    POST /api/memory/clear     clear one session, or everything
    GET  /api/runs             recent runs on disk
    GET  /api/trace            one run's trace.jsonl, parsed

The run endpoint streams rather than returning at the end because the point is
watching Perceive -> Reason -> Act -> Reflect happen. A live run takes a couple
of minutes; a spinner for two minutes is not an interface.
"""

from __future__ import annotations

import contextlib
import errno
import json
import mimetypes
import os
import socket
import threading
import typing as t
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import AppConfig, load_config
from ..core.rubric import Rubric
from ..core.state import Decision, MemoryHit, RunResult
from ..envfile import ENV_FILENAME, EnvFileReport, load_env_file
from ..envfile import summary as env_summary
from ..harness.fallbacks import ProviderChain
from ..harness.faults import FAILURE_KINDS, FAULT_STEPS, FaultyMemory, FaultyProvider
from ..harness.runner import Runner
from ..llm.base import LLMProvider
from ..llm.factory import availability, build_provider
from ..llm.types import LLMError, ProviderUnavailableError
from ..memory.base import MemoryRecord, MemoryStore
from ..memory.factory import build_memory
from ..memory.manager import MemoryManager
from ..memory.sqlite_store import SQLiteMemory
from ..observability.trace import read_trace
from ..tools.definitions import build_registry
from ..tools.registry import ToolContext, ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RUBRIC_DIR = PROJECT_ROOT / "config" / "rubrics"
SAMPLE_DIR = PROJECT_ROOT / "samples"

#: Which sample texts belong to which rubric. A manifest rather than a directory
#: scan, so the picker never offers a bug report to the essay rubric -- which
#: would score 1/5 across the board and teach nothing.
SAMPLES: dict[str, list[dict[str, str]]] = {
    "essay_argumentative": [
        {"id": "weak_essay", "label": "Weak - hedged, no evidence", "file": "weak_essay.txt"},
        {
            "id": "mediocre_essay",
            "label": "Mediocre - some structure, vague claims",
            "file": "mediocre_essay.txt",
        },
        {
            "id": "padded_essay",
            "label": "Padded - heavy filler, no position",
            "file": "padded_essay.txt",
        },
    ],
    "bug_report": [
        {
            "id": "weak_bug_report",
            "label": "Bad - accusatory, no repro steps",
            "file": "weak_bug_report.txt",
        },
        {
            "id": "vague_bug_report",
            "label": "Vague - no versions, no numbers",
            "file": "vague_bug_report.txt",
        },
    ],
}

#: One run at a time. Concurrent runs would interleave writes to the same
#: SQLite file and make the memory panel unreadable.
_LOCK = threading.Lock()

#: Loaded once, at import, so the answer is the same however the server was
#: started -- `python demo.py`, `python -m agentic_rubric.web.server`, or a
#: WSGI-ish embedding that never calls our `main()`. The report is kept because
#: "no key" has several quite different causes and the page has to name the
#: right one.
ENV_REPORT: EnvFileReport = load_env_file(PROJECT_ROOT / ENV_FILENAME)


class ServerBindError(RuntimeError):
    """The UI cannot bind because another process already owns the address."""


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that never shares a listening address with a stale process.

    ``ThreadingHTTPServer`` enables ``SO_REUSEADDR``.  On Windows that can let
    two Python processes bind the same host/port, after which the browser may
    reach either process.  ``SO_EXCLUSIVEADDRUSE`` makes ownership unambiguous.
    Disabling reuse on every platform also makes an already-running demo fail
    immediately instead of serving an arbitrary checkout.
    """

    allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


# ---------------------------------------------------------------------------
# Instrumentation: two thin wrappers that keep reporting out of core/
# ---------------------------------------------------------------------------


class RecallSpy(MemoryStore):
    """Records what each recall actually returned, and what was written.

    The loop's ``perceive`` event reports *how many* records were recalled but
    not what they said -- fine for a log line, useless for a panel whose whole
    point is showing memory change a decision. Wrapping the store keeps that
    reporting entirely outside ``core/``.
    """

    def __init__(self, inner: MemoryStore) -> None:
        self.inner = inner
        self.last: list[MemoryHit] = []
        self.writes: list[dict[str, t.Any]] = []

    def save(self, record: MemoryRecord) -> str:
        uid = self.inner.save(record)
        if uid:
            self.writes.append(
                {
                    "kind": record.kind,
                    "content": record.content,
                    "iteration": record.iteration,
                    "criterion_id": record.criterion_id,
                    "score_delta": record.score_delta,
                }
            )
        return uid

    def recall(self, query: str, **kwargs: t.Any) -> list[MemoryHit]:
        hits = self.inner.recall(query, **kwargs)
        self.last = list(hits)
        return hits

    def clear_session(self, session_id: str) -> int:
        return self.inner.clear_session(session_id)

    def list_sessions(self) -> list[str]:
        return self.inner.list_sessions()

    def stats(self) -> dict[str, t.Any]:
        return self.inner.stats()

    def close(self) -> None:
        self.inner.close()

    @property
    def describe(self) -> str:
        return self.inner.describe

    @property
    def degraded(self) -> bool:
        # Read by the runner when it annotates the RunResult. Without the
        # pass-through the harness panel reports a healthy store while the
        # circuit breaker underneath it is open -- which is exactly the silent
        # failure this project has been bitten by twice already.
        return bool(getattr(self.inner, "degraded", False))

    @property
    def degraded_reason(self) -> str:
        return str(getattr(self.inner, "degraded_reason", ""))


class DraftRecorder(ToolRegistry):
    """Snapshots the working draft after every tool call.

    The whole point of this application is showing what happened to the *text*,
    and no event in the loop carries it: ``revise_text`` reports word counts and
    a similarity ratio, not the new draft. So the registry is wrapped, because
    ``dispatch`` is handed the :class:`~..tools.registry.ToolContext` and the
    context holds the workspace.

    Subclasses rather than delegates so it *is* a ToolRegistry everywhere one is
    expected, including inside the harness's tool-recovery ladder.
    """

    def __init__(self, inner: ToolRegistry) -> None:
        super().__init__()
        for spec in inner.specs():
            entry = inner.get(spec.name)
            assert entry is not None
            self.register(entry.spec, entry.handler, terminal=entry.terminal)
        #: One entry per *change* to the draft, in order. The original is
        #: recorded before the run starts.
        self.stages: list[dict[str, t.Any]] = []
        self._current = ""

    def start(self, original: str) -> None:
        self._current = original
        self.stages = [
            {
                "stage": 0,
                "iteration": 0,
                "label": "Original",
                "action": "input",
                "text": original,
                "words": len(original.split()),
                "focus": [],
                "summary": "The text as submitted.",
            }
        ]

    def dispatch(self, decision: Decision, ctx: ToolContext) -> t.Any:
        result = super().dispatch(decision, ctx)
        after = ctx.workspace.draft
        if after != self._current:
            self._current = after
            focus = [str(c) for c in (result.arguments or {}).get("focus_criteria", [])]
            self.stages.append(
                {
                    "stage": len(self.stages),
                    "iteration": ctx.iteration,
                    "label": f"Revision {len(self.stages)}",
                    "action": result.action,
                    "text": after,
                    "words": len(after.split()),
                    "focus": focus,
                    "summary": result.summary,
                }
            )
        return result


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def load_rubrics() -> dict[str, Rubric]:
    """Every rubric on disk, in the order the picker should offer them.

    Ordered by the preset manifest rather than by filename, so the first entry
    is the one with presets and the worked example behind it. Sorting by
    filename put "bug_report" first and made an essay tool open on a bug-report
    rubric, which is a confusing first impression for no reason.
    """
    found = {Rubric.from_yaml(path).id: Rubric.from_yaml(path)
             for path in sorted(RUBRIC_DIR.glob("*.yaml"))}
    ordered = {rid: found[rid] for rid in SAMPLES if rid in found}
    ordered.update({rid: r for rid, r in found.items() if rid not in ordered})
    return ordered


def rubric_payload(rubric: Rubric) -> dict[str, t.Any]:
    return {
        "id": rubric.id,
        "name": rubric.name,
        "domain": rubric.domain,
        "description": rubric.description.strip(),
        "target_score": rubric.target_score,
        "scale": {"min": rubric.scale.min, "max": rubric.scale.max},
        "criteria": [
            {
                "id": c.id,
                "name": c.name,
                "weight": c.weight,
                "description": c.description.strip(),
                "probes": [p.describe for p in c.probes],
            }
            for c in rubric.criteria
        ],
    }


def sample_payload() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for rubric_id, entries in SAMPLES.items():
        items = []
        for entry in entries:
            path = SAMPLE_DIR / entry["file"]
            if path.exists():
                items.append(
                    {
                        "id": entry["id"],
                        "label": entry["label"],
                        "text": path.read_text(encoding="utf-8").strip(),
                    }
                )
        out[rubric_id] = items
    return out


def provider_payload(config: AppConfig) -> list[dict[str, t.Any]]:
    """The configured failover chain and whether each link is usable.

    Reported rather than filtered, so "why can I not run anything?" is
    answerable from the page instead of from a stack trace.
    """
    rows: list[dict[str, t.Any]] = []
    for position, name in enumerate(config.llm.chain, start=1):
        settings = config.llm.providers.get(name)
        if settings is None or name == "mock":
            continue
        ok, reason = availability(settings)
        rows.append(
            {
                "position": position,
                "name": name,
                "model": settings.model,
                "available": ok,
                # "environment variable GROQ_API_KEY is empty" is true and
                # useless. Say which of the several causes actually applies.
                "reason": reason if ok else ENV_REPORT.describes(settings.api_key_env),
                "key_env": settings.api_key_env,
            }
        )
    return rows


def tool_payload(rubric: Rubric) -> list[dict[str, t.Any]]:
    """The tool set as the model actually sees it, generated from the rubric."""
    registry = build_registry(rubric)
    llm_backed = {"score_against_rubric", "revise_text"}
    return [
        {
            "name": spec.name,
            "description": spec.description.strip().splitlines()[0],
            "uses_llm": spec.name in llm_backed,
            "terminal": registry.is_terminal(spec.name),
            "arguments": [
                key
                for key in spec.parameters.get("properties", {})
                if key != "thought"
            ],
        }
        for spec in registry.specs()
    ]


def unwrap_store(store: MemoryStore) -> MemoryStore:
    """Peel the spy and the policy layer off to reach the SQLite store."""
    seen = store
    for _ in range(4):
        inner = getattr(seen, "inner", None) or getattr(seen, "store", None)
        if inner is None:
            break
        seen = inner
    return seen


def stored_lessons(store: MemoryStore, limit: int = 25) -> list[dict[str, t.Any]]:
    inner = unwrap_store(store)
    if not isinstance(inner, SQLiteMemory):
        return []
    try:
        # An empty query has no vector and no keyword terms, so the store falls
        # through to its recency path -- what a "what do you know?" panel wants.
        results = inner.search("", kinds=["lesson"], limit=limit)
    except Exception:  # noqa: BLE001 - a panel must not break the page
        return []
    return [
        {
            "content": item.record.content,
            "rubric_id": item.record.rubric_id,
            "session_id": item.record.session_id,
            "iteration": item.record.iteration,
            "hits": int(item.record.metadata.get("hits", 1)),
            "created_at": item.record.created_at,
        }
        for item in results
    ]


def memory_payload(store: MemoryStore, notes: list[str]) -> dict[str, t.Any]:
    try:
        stats = store.stats()
    except Exception as exc:  # noqa: BLE001
        stats = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        sessions = store.list_sessions()
    except Exception:  # noqa: BLE001
        sessions = []
    return {
        "stats": stats,
        "sessions": sessions,
        "lessons": stored_lessons(store),
        "notes": notes,
    }


def recent_runs(config: AppConfig, limit: int = 20) -> list[dict[str, t.Any]]:
    root = config.path(config.logging.trace_dir)
    if not root.exists():
        return []
    rows: list[dict[str, t.Any]] = []
    for directory in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        summary = directory / "summary.json"
        if not summary.is_file():
            continue
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append(
            {
                "run_id": data.get("run_id", directory.name),
                "session_id": data.get("session_id", ""),
                "status": data.get("status", ""),
                "rubric_id": data.get("rubric_id", ""),
                "iterations": data.get("iterations", 0),
                "initial_score": data.get("initial_score"),
                "best_score": data.get("best_score"),
                "tokens": (data.get("tokens") or {}).get("total", 0),
                "elapsed_s": data.get("elapsed_s", 0),
                "harness": data.get("harness", {}),
            }
        )
        if len(rows) >= limit:
            break
    return rows


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def build_overrides(body: dict[str, t.Any]) -> dict[str, t.Any]:
    """Map the form onto config keys. Nothing here is special-cased in the loop."""
    overrides: dict[str, t.Any] = {}
    numeric = {
        "target_score": ("loop.target_score", float),
        "max_iterations": ("loop.max_iterations", int),
        "revise_candidates": ("loop.revise_candidates", int),
        "token_budget": ("guardrails.token_budget", int),
        "temperature": ("llm.temperature", float),
    }
    for field, (key, caster) in numeric.items():
        if body.get(field) not in (None, ""):
            overrides[key] = caster(body[field])
    if body.get("provider"):
        overrides["llm.primary"] = str(body["provider"])
    if body.get("memory_backend"):
        overrides["memory.backend"] = str(body["memory_backend"])
    overrides["memory.enabled"] = bool(body.get("memory_enabled", True))
    return overrides


def build_chain(config: AppConfig, requested: str) -> tuple[LLMProvider, ProviderChain, list[str]]:
    """The live failover chain, primary first. Raises if nothing is usable."""
    notes: list[str] = []
    ordered = [requested, *[n for n in config.llm.chain if n != requested]] if requested \
        else list(config.llm.chain)
    links: list[tuple[str, t.Callable[[], LLMProvider]]] = []
    primary: LLMProvider | None = None

    for name in ordered:
        if name == "mock":
            continue  # this application does not run against a simulation
        try:
            provider = build_provider(config, name)
        except (ProviderUnavailableError, LLMError, ValueError) as exc:
            notes.append(f"provider {name!r} unavailable: {exc}")
            continue
        if primary is None:
            primary = provider
            links.append((name, lambda p=provider: p))  # type: ignore[misc]
        else:
            provider.close()
            links.append((name, lambda n=name: build_provider(config, n)))  # type: ignore[misc]

    if primary is None:
        detail = "; ".join(
            ENV_REPORT.describes(config.llm.providers[name].api_key_env)
            for name in ordered
            if name in config.llm.providers and name != "mock"
        )
        raise ProviderUnavailableError(
            "No live provider is usable. " + detail + " " + " ".join(notes)
        )
    if len(links) > 1:
        notes.append("failover chain: " + " -> ".join(name for name, _ in links))
    return primary, ProviderChain(links=links), notes


def final_payload(
    result: RunResult,
    *,
    spy: RecallSpy,
    recorder: DraftRecorder,
    guardrails: dict[str, t.Any],
    trace_path: str,
) -> dict[str, t.Any]:
    payload = result.to_dict()
    payload["scorecards"] = [card.to_dict() for card in result.scorecards]
    payload["initial_draft"] = result.initial_draft
    payload["final_draft"] = result.final_draft
    payload["best_draft"] = result.best_draft

    # The transformation timeline, which is the thing this application exists to
    # show. The final entry is the best draft, which is not always the last one
    # produced -- a revision can make things worse.
    stages = list(recorder.stages)
    if stages and stages[-1]["text"] != result.best_draft:
        stages.append(
            {
                "stage": len(stages),
                "iteration": result.iterations,
                "label": "Best draft (returned)",
                "action": "select_best",
                "text": result.best_draft,
                "words": len(result.best_draft.split()),
                "focus": [],
                "summary": "The highest-scoring draft seen, which is what the run returns.",
            }
        )
    elif stages:
        stages[-1] = {**stages[-1], "label": stages[-1]["label"] + " - returned as best"}
    payload["stages"] = stages

    payload["memory_writes"] = spy.writes
    payload["memory"] = memory_payload(spy, [])
    payload["guardrails"] = guardrails
    payload["trace_path"] = trace_path
    return payload


#: How the server obtains its providers. Injected so the test suite can supply a
#: deterministic one and stay free of an API key -- the same substitution every
#: other test in this project makes. It is a *parameter*, not a mode: nothing the
#: browser can send selects a simulated provider, because a UI that can quietly
#: show you invented scores eventually will.
ChainFactory = t.Callable[[AppConfig, str], "tuple[LLMProvider, ProviderChain, list[str]]"]


def run_loop_streaming(
    body: dict[str, t.Any],
    emit: t.Callable[[str, dict], None],
    *,
    chain_factory: ChainFactory | None = None,
) -> None:
    """Execute one supervised run, pushing every event to ``emit`` as it happens."""
    build: ChainFactory = chain_factory if chain_factory is not None else build_chain
    rubrics = load_rubrics()
    rubric_id = str(body.get("rubric_id") or "essay_argumentative")
    rubric = rubrics.get(rubric_id)
    if rubric is None:
        emit("error", {"message": f"unknown rubric {rubric_id!r}"})
        return

    text = str(body.get("text") or "").strip()
    if not text:
        emit("error", {"message": "No text to work on. Pick a sample or paste your own."})
        return

    config = load_config(CONFIG_PATH, overrides=build_overrides(body), project_root=PROJECT_ROOT)
    session_id = str(body.get("session_id") or "app").strip() or "app"
    memory_on = bool(body.get("memory_enabled", True))
    failure = str(body.get("simulate_failure") or "")
    fail_step = str(body.get("fail_step") or "judge")

    try:
        primary, chain, notes = build(config, str(body.get("provider") or ""))
    except ProviderUnavailableError as exc:
        emit("error", {"message": str(exc)})
        return

    if failure in ("rate_limit", "server_error", "bad_json", "provider_down"):
        primary = FaultyProvider(primary, kind=failure, step=fail_step)
        chain = ProviderChain(
            links=[(primary.name, lambda p=primary: p), *chain.links[1:]]  # type: ignore[misc]
        )
        notes.append(f"injected a {failure} failure into the {fail_step} step")

    store, memory_notes = build_memory(config, enabled=memory_on)
    notes.extend(memory_notes)
    if failure == "memory_down" and memory_on:
        store = MemoryManager(FaultyMemory(unwrap_store(store)), config.memory)
        notes.append("memory reads will fail; the circuit breaker should open")

    spy = RecallSpy(store)

    # The recorder wraps the *rigged* registry rather than the other way round.
    # Both classes copy their handlers out of the registry they are given, so
    # whichever one ends up outermost is the one whose `dispatch` actually runs
    # -- and if that is not the recorder, the text timeline comes back empty.
    inner_registry: ToolRegistry = build_registry(rubric)
    if failure == "tool_error":
        from ..harness.faults import FaultyRegistry

        inner_registry = FaultyRegistry(inner_registry)
        notes.append("the reviser's model call will be rate limited once")

    recorder = DraftRecorder(inner_registry)
    recorder.start(text)
    registry: ToolRegistry = recorder

    for note in notes:
        emit("note", {"message": note})

    def on_event(name: str, payload: dict[str, t.Any]) -> None:
        enriched = dict(payload)
        if name == "perceive":
            enriched["recalled_items"] = [
                {
                    "kind": hit.kind,
                    "content": hit.content,
                    "score": hit.score,
                    "session_id": hit.session_id,
                    "iteration": hit.iteration,
                }
                for hit in spy.last
            ]
        if name == "act" and recorder.stages:
            latest = recorder.stages[-1]
            if latest["iteration"] == payload.get("iteration"):
                enriched["draft_stage"] = latest["stage"]
                enriched["draft_words"] = latest["words"]
        emit(name, enriched)

    runner = Runner(
        config=config,
        rubric=rubric,
        provider=primary,
        chain=chain,
        memory=spy,
        registry=registry,
        console=on_event,
    )
    try:
        report = runner.run(text, session_id=session_id)
        emit(
            "complete",
            final_payload(
                report.result,
                spy=spy,
                recorder=recorder,
                guardrails=report.guardrails,
                trace_path=report.trace_path,
            ),
        )
    except LLMError as exc:
        emit(
            "error",
            {
                "message": f"{type(exc).__name__}: {exc}",
                "hint": "Every provider in the chain was exhausted. "
                        "Check the key and the model id.",
            },
        )
    except Exception as exc:  # noqa: BLE001 - report, never 500 the stream
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "AgenticRubricLoop/2.0"

    def log_message(self, fmt: str, *args: t.Any) -> None:
        # The default logger writes a line per asset request and drowns the run's
        # own output, which is what the operator is watching.
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    # -- helpers ------------------------------------------------------------

    def _send_json(self, payload: t.Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, t.Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        if STATIC_DIR.resolve() not in resolved.parents and resolved != STATIC_DIR.resolve():
            self._send_json({"error": "not found"}, status=404)
            return
        if not resolved.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        data = resolved.read_bytes()
        mime, _ = mimetypes.guess_type(resolved.name)
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif route == "/api/bootstrap":
            self._bootstrap()
        elif route == "/api/memory":
            self._memory()
        elif route == "/api/runs":
            self._runs()
        elif route == "/api/trace":
            self._trace()
        elif route.startswith("/static/"):
            self._send_file(STATIC_DIR / route[len("/static/") :])
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/run":
            self._run()
        elif route == "/api/memory/clear":
            self._memory_clear()
        else:
            self._send_json({"error": "not found"}, status=404)

    # -- handlers -----------------------------------------------------------

    def _bootstrap(self) -> None:
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        rubrics = load_rubrics()
        providers = provider_payload(config)
        store, notes = build_memory(config, enabled=True, warm_embedder=False)
        try:
            memory = memory_payload(store, notes)
        finally:
            store.close()

        self._send_json(
            {
                "rubrics": [rubric_payload(r) for r in rubrics.values()],
                "tools": {rid: tool_payload(r) for rid, r in rubrics.items()},
                "samples": sample_payload(),
                "providers": providers,
                "ready": any(row["available"] for row in providers),
                "failures": list(FAILURE_KINDS),
                "failure_steps": list(FAULT_STEPS),
                "memory": memory,
                "env": env_summary(ENV_REPORT),
                "runs": recent_runs(config),
                "defaults": {
                    "provider": next(
                        (row["name"] for row in providers if row["available"]), ""
                    ),
                    "target_score": config.loop.target_score,
                    "max_iterations": config.loop.max_iterations,
                    "revise_candidates": config.loop.revise_candidates,
                    "temperature": config.llm.temperature,
                    "token_budget": config.guardrails.token_budget,
                    "memory_enabled": config.memory.enabled,
                    "memory_backend": config.memory.backend,
                    "embed_model": config.memory.embed_model,
                    "lesson_scope": config.memory.lesson_scope,
                    "max_lessons_per_recall": config.memory.max_lessons_per_recall,
                    "retry": {
                        "max_attempts": config.retry.max_attempts,
                        "jitter": config.retry.jitter,
                        "tool_max_attempts": config.retry.tool_max_attempts,
                    },
                },
            }
        )

    def _memory(self) -> None:
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        store, notes = build_memory(config, enabled=True, warm_embedder=False)
        try:
            self._send_json(memory_payload(store, notes))
        finally:
            store.close()

    def _memory_clear(self) -> None:
        body = self._read_json()
        session_id = str(body.get("session_id") or "").strip()
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        store, _ = build_memory(config, enabled=True, warm_embedder=False)
        try:
            if session_id:
                removed = store.clear_session(session_id)
            else:
                inner = unwrap_store(store)
                removed = inner.clear_all() if isinstance(inner, SQLiteMemory) else 0
            self._send_json({"removed": removed, "session_id": session_id or "*"})
        finally:
            store.close()

    def _runs(self) -> None:
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        self._send_json({"runs": recent_runs(config)})

    def _trace(self) -> None:
        run_id = (parse_qs(urlparse(self.path).query).get("run_id") or [""])[0]
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        path = config.path(config.logging.trace_dir) / run_id / "trace.jsonl"
        # `run_id` arrives from the page; keep it from walking out of runs/.
        root = config.path(config.logging.trace_dir).resolve()
        if not run_id or root not in path.resolve().parents or not path.is_file():
            self._send_json({"error": "no such trace", "run_id": run_id}, status=404)
            return
        try:
            self._send_json({"run_id": run_id, "events": read_trace(path)})
        except (OSError, ValueError) as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _run(self) -> None:
        body = self._read_json()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(name: str, payload: dict[str, t.Any]) -> None:
            line = json.dumps({"event": name, "data": payload}, default=str) + "\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        try:
            with _LOCK:
                run_loop_streaming(body, emit)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-run
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(OSError):
                emit("error", {"message": f"{type(exc).__name__}: {exc}"})


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    on_ready: t.Callable[[str], None] | None = None,
) -> None:
    url = f"http://{host}:{port}"
    try:
        server = ExclusiveThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        address_in_use = exc.errno == errno.EADDRINUSE or getattr(exc, "winerror", None) == 10048
        if address_in_use:
            raise ServerBindError(
                f"cannot start {url}: port {port} is already in use. Stop the older "
                f"demo process, or start this one with --port {port + 1}"
            ) from exc
        raise

    print(f"Rubric Forge -> {url}")
    print("Press Ctrl+C to stop.")
    if on_ready is not None:
        on_ready(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def main(argv: t.Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Rubric Forge application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    # .env was already loaded at import; report anything surprising about it.
    for note in ENV_REPORT.notes:
        print(f"note: {note}")
    try:
        serve(args.host, args.port)
    except ServerBindError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ChainFactory",
    "DraftRecorder",
    "Handler",
    "RecallSpy",
    "ExclusiveThreadingHTTPServer",
    "ServerBindError",
    "build_chain",
    "main",
    "run_loop_streaming",
    "serve",
]
