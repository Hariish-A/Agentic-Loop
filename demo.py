"""One-command launcher for the application.

    python demo.py                 # http://127.0.0.1:8000
    python demo.py --port 8080
    python demo.py --no-browser

Exists so the application is a single command from a fresh checkout: it puts
``src`` on the path, loads ``.env``, and opens a browser. No install step, no
PYTHONPATH, no web framework.

Runs against a live provider only -- there is no simulated mode here. Set
``GROQ_API_KEY`` in ``.env`` first, or start ``ollama serve`` for the local
fallback; the page reports which link in the chain is missing.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    # Importing the server loads .env and records what happened; a blank
    # variable in the shell no longer shadows the real key in the file.
    from agentic_rubric.web.server import ENV_REPORT, ServerBindError, serve

    for note in ENV_REPORT.notes:
        print(f"note: {note}")
    if ENV_REPORT.error:
        print(f"note: {ENV_REPORT.error}")

    url = f"http://{args.host}:{args.port}"

    print("=" * 64)
    print("  Agentic Rubric Loop")
    print(f"  {url}")
    print("  Live providers only. Set GROQ_API_KEY in .env, or run `ollama serve`.")
    print(f"  .env: {ENV_REPORT.path} "
          f"({'found' if ENV_REPORT.exists else 'NOT FOUND'})")
    print("=" * 64)
    def open_browser(ready_url: str) -> None:
        if not args.no_browser:
            # Binding has succeeded, so this can never open a stale server.
            threading.Timer(0.2, lambda: webbrowser.open(ready_url)).start()

    try:
        serve(args.host, args.port, on_ready=open_browser)
    except ServerBindError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
