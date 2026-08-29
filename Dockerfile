# ---------------------------------------------------------------------------
# Rubric Forge
#
# Two decisions worth stating up front, because both cost something:
#
# 1. `python:3.12-slim`, not `alpine`. fastembed pulls onnxruntime, which ships
#    manylinux wheels and has no musl build -- on alpine pip falls back to
#    compiling, which needs a toolchain and turns a 30-second build into twenty
#    minutes for a larger image.
#
# 2. The embedding model is baked in at build time (`--warm`). Otherwise the
#    first `recall` of the first run downloads ~90 MB, which makes a demo look
#    broken and makes an air-gapped deployment impossible. It also means the
#    image works with no network at all.
#
# Dependencies are installed before the source is copied, so editing a Python
# file rebuilds one small layer instead of re-downloading onnxruntime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# --- dependency layer (cached until requirements change) -------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- application ------------------------------------------------------------
COPY src/ ./src/
COPY config/ ./config/
COPY samples/ ./samples/
COPY scripts/ ./scripts/

# --- warm the ONNX embedding cache so the first run needs no network --------
RUN python scripts/warm_models.py || \
    echo "WARNING: embedding model not cached; memory will fall back to BM25 keyword recall"

# --- non-root ---------------------------------------------------------------
# The container writes to two places only: data/ (the memory database) and
# runs/ (traces). Both are volume mount points, created and chowned here so a
# bind mount from the host does not land root-owned.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/data /app/runs /app/models \
    && chown -R agent:agent /app
USER agent

VOLUME ["/app/data", "/app/runs"]

# A container that cannot reach its provider should say so and exit non-zero,
# rather than failing three retries into the first run.
HEALTHCHECK --interval=60s --timeout=20s --start-period=5s --retries=2 \
    CMD python scripts/preflight.py > /dev/null || exit 1

ENTRYPOINT ["python", "-m", "agentic_rubric.cli"]
CMD ["--input", "samples/weak_essay.txt", "--provider", "mock"]
