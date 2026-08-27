"""Download and cache the embedding model into the image at build time.

Without this, the first ``recall`` of the first run in a fresh container pulls
~90 MB over the network -- which makes a demo look broken, makes the image
useless offline, and puts a download inside the wall-clock guardrail's budget.

Exits non-zero when the model cannot be fetched, so the Dockerfile can print a
warning and continue: an image without the embedder still works, it just falls
back to FTS5/BM25 keyword recall.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_rubric.config import load_config  # noqa: E402
from agentic_rubric.memory.embedding import build_embedder  # noqa: E402
from agentic_rubric.memory.factory import MODEL_CACHE_DIRNAME  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "config" / "config.yaml", project_root=ROOT)
    if config.memory.embedder == "none":
        print("embedder is 'none'; nothing to warm")
        return 0

    cache = config.path(MODEL_CACHE_DIRNAME)
    cache.mkdir(parents=True, exist_ok=True)
    embedder, note = build_embedder(
        config.memory.embedder, config.memory.embed_model, cache_dir=cache, warm=True
    )
    print(f"embedder: {note}")

    vector = embedder.embed_one("warm the cache with one real sentence")
    if not vector:
        print("model did not produce a vector", file=sys.stderr)
        return 1
    print(f"cached {config.memory.embed_model} -> {cache} ({len(vector)} dimensions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
