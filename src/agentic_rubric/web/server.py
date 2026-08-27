"""A small demo server for the Milestone 1 + 2 loop.

Deliberately built on ``http.server`` from the standard library. A demo whose
first step is "pip install a web framework" is a demo that fails on the reviewer's
machine, and nothing here needs more than routing and a streaming response.

Three endpoints do the work:

``GET  /api/bootstrap``   rubrics, sample texts, provider availability, defaults
``POST /api/run``         runs the loop, streaming one NDJSON line per event
``POST /api/memory/*``    stats and clear-session, the operable half of memory

The run endpoint streams rather than returning at the end, because the point of
the demo is watching Perceive -> Reason -> Act -> Reflect happen, not seeing a
final score appear after ten silent seconds.
"""

from __future__ import annotations

import contextlib
import json
import mimetypes
import threading
import typing as t
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..config import AppConfig, load_config
from ..core.loop import AgenticLoop
from ..core.rubric import Rubric
from ..core.state import MemoryHit, RunResult
from ..llm.base import LLMProvider
from ..llm.demo_responder import ScriptedAgentResponder, make_failure
from ..llm.factory import availability, build_provider
from ..llm.mock import MockProvider
from ..llm.types import LLMError
from ..memory.base import MemoryStore
from ..memory.factory import build_memory
from ..memory.sqlite_store import SQLiteMemory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RUBRIC_DIR = PROJECT_ROOT / "config" / "rubrics"
SAMPLE_DIR = PROJECT_ROOT / "samples"

#: Which sample texts belong to which rubric. Keeping this a manifest rather
#: than a directory scan means the picker never offers a bug report to the essay
#: rubric, which would score 1/5 across the board and teach nothing.
SAMPLES: dict[str, list[dict[str, str]]] = {
    "essay_argumentative": [
        {"id": "weak_essay", "label": "Weak essay - hedged, no evidence", "file": "weak_essay.txt"},
        {
            "id": "mediocre_essay",
            "label": "Mediocre essay - some structure, vague claims",
            "file": "mediocre_essay.txt",
        },
        {
            "id": "padded_essay",
            "label": "Padded essay - heavy filler, no position",
            "file": "padded_essay.txt",
        },
    ],
    "bug_report": [
        {
            "id": "weak_bug_report",
            "label": "Bad bug report - accusatory, no repro",
            "file": "weak_bug_report.txt",
        },
        {
            "id": "vague_bug_report",
            "label": "Vague bug report - no versions, no numbers",
            "file": "vague_bug_report.txt",
        },
    ],
}

FAILURE_KINDS = ("rate_limit", "bad_json", "server_error", "provider_down")
FAILURE_STEPS = ("reason", "judge", "revise", "reflect")

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Memory spy
# ---------------------------------------------------------------------------


class RecallSpy(MemoryStore):
    """Wraps a store to record what each recall actually returned.

    The loop's ``perceive`` event reports how many records were recalled but not
    what they said -- fine for a log line, useless for a demo whose whole point
    is showing memory changing a decision. Wrapping the store keeps that
    reporting entirely outside ``core/``.
    """

    def __init__(self, inner: MemoryStore) -> None:
        self.inner = inner
        self.last: list[MemoryHit] = []
        self.writes: list[dict[str, t.Any]] = []

    def save(self, record: t.Any) -> str:
        uid = self.inner.save(record)
        if uid:
            self.writes.append(
                {"kind": record.kind, "content": record.content, "iteration": record.iteration}
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


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def load_rubrics() -> dict[str, Rubric]:
    return {
        path.stem if path.stem in SAMPLES else Rubric.from_yaml(path).id: Rubric.from_yaml(path)
        for path in sorted(RUBRIC_DIR.glob("*.yaml"))
    }


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
    rows = [
        {
            "name": "mock",
            "model": "scripted, offline",
            "available": True,
            "reason": "no API key needed; simulates the agent deterministically",
        }
    ]
    for name in config.llm.chain:
        settings = config.llm.providers.get(name)
        if settings is None or name == "mock":
            continue
        ok, reason = availability(settings)
        rows.append(
            {"name": name, "model": settings.model, "available": ok, "reason": reason}
        )
    return rows


def make_provider(
    name: str, config: AppConfig, rubric: Rubric, target: float, failure: str, step: str
) -> LLMProvider:
    if name == "mock":
        responder = ScriptedAgentResponder(rubric=rubric, target_score=target)
        if failure:
            responder.fail_on[step] = make_failure(failure)
        return MockProvider(responder=responder, name="mock")
    return build_provider(config, name)


def unwrap_store(store: MemoryStore) -> MemoryStore:
    """Peel the spy and the policy layer off to reach the SQLite store."""
    seen = store
    for _ in range(4):
        inner = getattr(seen, "inner", None) or getattr(seen, "store", None)
        if inner is None:
            break
        seen = inner
    return seen


def recent_lessons(store: MemoryStore, limit: int = 12) -> list[dict[str, t.Any]]:
    """Most recent lessons across every session, for the memory panel."""
    inner = unwrap_store(store)
    if not isinstance(inner, SQLiteMemory):
        return []
    try:
        # An empty query has no vector and no keyword terms, so the store falls
        # through to its recency path -- exactly what a "what do you know?"
        # panel wants.
        results = inner.search("", kinds=["lesson"], limit=limit)
    except Exception:  # noqa: BLE001 - a demo panel must not break the page
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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def build_overrides(body: dict[str, t.Any]) -> dict[str, t.Any]:
    overrides: dict[str, t.Any] = {}
    if body.get("target_score") is not None:
        overrides["loop.target_score"] = float(body["target_score"])
    if body.get("max_iterations") is not None:
        overrides["loop.max_iterations"] = int(body["max_iterations"])
    if body.get("revise_candidates") is not None:
        overrides["loop.revise_candidates"] = int(body["revise_candidates"])
    if body.get("memory_backend"):
        overrides["memory.backend"] = str(body["memory_backend"])
    overrides["memory.enabled"] = bool(body.get("memory_enabled", True))
    return overrides


def final_payload(result: RunResult, spy: RecallSpy | None, store: MemoryStore) -> dict[str, t.Any]:
    payload = result.to_dict()
    payload["scorecards"] = [card.to_dict() for card in result.scorecards]
    payload["initial_draft"] = result.initial_draft
    payload["best_draft"] = result.best_draft
    payload["final_draft"] = result.final_draft
    payload["memory_writes"] = spy.writes if spy else []
    try:
        payload["memory_stats"] = store.stats()
    except Exception:  # noqa: BLE001
        payload["memory_stats"] = {}
    payload["lessons"] = recent_lessons(store)
    return payload


def run_loop_streaming(body: dict[str, t.Any], emit: t.Callable[[str, dict], None]) -> None:
    """Execute one run, pushing every loop event to ``emit`` as it happens."""
    rubrics = load_rubrics()
    rubric_id = str(body.get("rubric_id") or "essay_argumentative")
    rubric = rubrics.get(rubric_id)
    if rubric is None:
        emit("error", {"message": f"unknown rubric {rubric_id!r}"})
        return

    text = str(body.get("text") or "").strip()
    if not text:
        emit("error", {"message": "no text supplied"})
        return

    config = load_config(CONFIG_PATH, overrides=build_overrides(body), project_root=PROJECT_ROOT)
    target = float(body.get("target_score") or config.loop.target_score)
    provider_name = str(body.get("provider") or "mock")
    failure = str(body.get("simulate_failure") or "")
    fail_step = str(body.get("fail_step") or "judge")
    session_id = str(body.get("session_id") or "web-demo")

    if failure and provider_name != "mock":
        emit(
            "note",
            {"message": "failure injection is only wired to the mock provider; ignoring it"},
        )
        failure = ""

    try:
        provider = make_provider(provider_name, config, rubric, target, failure, fail_step)
    except (LLMError, ValueError) as exc:
        emit("error", {"message": f"could not build provider {provider_name!r}: {exc}"})
        return

    store, notes = build_memory(config, enabled=bool(body.get("memory_enabled", True)))
    spy = RecallSpy(store)
    for note in notes:
        emit("note", {"message": note})
    if failure:
        emit("note", {"message": f"injected a {failure} failure into the {fail_step} step"})

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
        emit(name, enriched)

    loop = AgenticLoop(
        config=config, provider=provider, rubric=rubric, memory=spy, on_event=on_event
    )
    try:
        result = loop.run(text, session_id=session_id, target_score=target)
        emit("complete", final_payload(result, spy, spy))
    except Exception as exc:  # noqa: BLE001 - report, never 500 the stream
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
    finally:
        provider.close()
        spy.close()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "AgenticRubricDemo/1.0"

    def log_message(self, fmt: str, *args: t.Any) -> None:
        # The default logger writes a line per asset request and drowns the
        # loop's own output, which is what the operator is watching.
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
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(path.name)
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
        elif route == "/api/memory/stats":
            self._memory_stats()
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
        self._send_json(
            {
                "rubrics": [rubric_payload(r) for r in rubrics.values()],
                "samples": sample_payload(),
                "providers": provider_payload(config),
                "failures": list(FAILURE_KINDS),
                "failure_steps": list(FAILURE_STEPS),
                "defaults": {
                    "target_score": config.loop.target_score,
                    "max_iterations": config.loop.max_iterations,
                    "revise_candidates": config.loop.revise_candidates,
                    "memory_enabled": config.memory.enabled,
                    "memory_backend": config.memory.backend,
                    "embed_model": config.memory.embed_model,
                    "lesson_scope": config.memory.lesson_scope,
                    "max_lessons_per_recall": config.memory.max_lessons_per_recall,
                },
            }
        )

    def _memory_stats(self) -> None:
        config = load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
        store, notes = build_memory(config, enabled=True, warm_embedder=False)
        try:
            self._send_json(
                {
                    "stats": store.stats(),
                    "lessons": recent_lessons(store),
                    "sessions": store.list_sessions(),
                    "notes": notes,
                }
            )
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
            # One run at a time: concurrent runs would interleave writes to the
            # same SQLite file and make the memory demo unreadable.
            with _LOCK:
                run_loop_streaming(body, emit)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-run
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(OSError):
                emit("error", {"message": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Agentic Rubric Loop demo -> http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


def main(argv: t.Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Milestone 1+2 demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Handler", "RecallSpy", "main", "run_loop_streaming", "serve"]
