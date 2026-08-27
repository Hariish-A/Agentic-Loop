"""One-command launcher for the browser demo.

    python demo.py                 # http://127.0.0.1:8000
    python demo.py --port 8080
    python demo.py --no-browser

Exists so the demo is a single command from a fresh checkout: it puts ``src`` on
the path, loads ``.env``, and opens a browser. No install step, no PYTHONPATH,
no web framework.
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
        pass  # the mock provider needs no keys

    from agentic_rubric.web.server import serve

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # Fire after the server is listening, but do not block startup on it.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print("=" * 64)
    print("  Agentic Rubric Loop - Milestone 1 & 2 demo")
    print(f"  {url}")
    print("  Provider 'mock' needs no API key. Set GROQ_API_KEY in .env for live runs.")
    print("=" * 64)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
