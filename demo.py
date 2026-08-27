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

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        # Not fatal: the key may already be exported in the environment.
        print("note: python-dotenv is not installed; reading keys from the environment only")

    from agentic_rubric.web.server import serve

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # Fire after the server is listening, but do not block startup on it.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print("=" * 64)
    print("  Agentic Rubric Loop")
    print(f"  {url}")
    print("  Live providers only. Set GROQ_API_KEY in .env, or run `ollama serve`.")
    print("=" * 64)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
