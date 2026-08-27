"""Typed, layered runtime configuration.

Precedence, highest wins::

    CLI overrides  >  environment variables  >  YAML file  >  dataclass defaults

Every knob the challenge asks to be configurable -- model, iteration limits,
token budget, retry settings, memory backend -- lives here. Nothing under
``core/`` imports :mod:`os` or reads YAML; the loop is handed a frozen
:class:`AppConfig`, which is what keeps runtime parameters swappable without
editing loop code.

Environment override convention::

    AGENTIC_<SECTION>__<FIELD>=<value>      # AGENTIC_LOOP__MAX_ITERATIONS=8

Double underscore separates nesting levels, so arbitrarily deep keys work::

    AGENTIC_LLM__PROVIDERS__GROQ__MODEL=openai/gpt-oss-20b
"""

from __future__ import annotations

import os
import types
import typing as t
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path

import yaml

ENV_PREFIX = "AGENTIC_"
NEST_SEP = "__"

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class ConfigError(ValueError):
    """Raised when a config file or override cannot be turned into an AppConfig."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMProviderConfig:
    """One OpenAI-compatible endpoint.

    GroqCloud, Ollama, OpenAI, OpenRouter and others all speak the same
    ``/chat/completions`` dialect, so a single client class plus this record is
    enough to swap providers from a config file.
    """

    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 60.0
    supports_tools: bool = True
    #: Local backends (Ollama) and the mock need no credentials. Declared
    #: explicitly rather than inferred from the URL, so "why is this provider
    #: being skipped?" is answerable from the config file alone.
    requires_key: bool = True
    #: Groq rejects ``messages[].name``. Declared per provider so the client
    #: strips the field instead of the caller having to know which backend it
    #: is talking to.
    supports_message_name: bool = True
    #: Price per 1,000 tokens, for the ``cost_est`` column in the trace. Zero by
    #: default: every provider in the shipped chain has a free path, and a
    #: fabricated price is worse than an honest zero. Set these to your plan's
    #: rates and the estimate becomes real without a code change.
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    def api_key(self, env: t.Mapping[str, str] | None = None) -> str | None:
        """Read this provider key from the environment, or ``None`` if unset."""
        if not self.api_key_env:
            return None
        source = env if env is not None else os.environ
        value = source.get(self.api_key_env, "").strip()
        return value or None


@dataclass(frozen=True)
class LLMConfig:
    primary: str = "mock"
    fallbacks: list[str] = field(default_factory=list)
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)
    temperature: float = 0.3
    max_tokens: int = 2048

    @property
    def chain(self) -> list[str]:
        """Provider names in the order the harness should try them."""
        # dict.fromkeys preserves insertion order and drops duplicates, which
        # is exactly "the chain, without repeating a provider named twice".
        return list(dict.fromkeys([self.primary, *self.fallbacks]))

    def provider(self, name: str) -> LLMProviderConfig:
        try:
            return self.providers[name]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "<none>"
            raise ConfigError(f"unknown LLM provider {name!r}; configured: {known}") from None


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = 6
    target_score: float = 85.0
    min_improvement: float = 1.0
    plateau_patience: int = 2
    revise_candidates: int = 1
    #: Admission floor, used when the rubric does not declare its own. Applied
    #: once before the loop starts, so an ungradeable submission costs no tokens.
    min_input_words: int = 40
    min_input_sentences: int = 2


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    backend: str = "sqlite_vec"
    db_path: str = "data/memory.db"
    embedder: str = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    recall_top_k: int = 5
    recall_min_score: float = 0.25
    #: Lessons outlive a session but belong to the rubric that produced them.
    lesson_scope: str = "rubric"
    episodic_scope: str = "session"
    #: Lessons are few and curated, so they are ranked but not relevance-gated.
    #: The gate applies to episodic recall, which is voluminous and noisy.
    gate_lessons: bool = False
    max_lessons_per_recall: int = 3
    #: Blend weight when both the vector and keyword channels return a hit.
    vector_weight: float = 0.7


@dataclass(frozen=True)
class RetryConfig:
    """Backoff policy. LLM calls and tool calls get separate budgets.

    A tool retry is cheap in latency and expensive in tokens (two of the five
    tools call a model), and a failing tool is far more often deterministic
    than a failing HTTP request. So tools get fewer attempts and a shorter
    base delay than the transport does.
    """

    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: str = "full"
    retry_on_status: list[int] = field(
        default_factory=lambda: [408, 409, 425, 429, 500, 502, 503, 504]
    )
    tool_max_attempts: int = 2
    tool_base_delay_s: float = 0.25
    #: Repair round trips allowed per unparseable response, before the safe
    #: default action is taken. One: a model that cannot produce valid JSON
    #: twice will not produce it on the third try either.
    repair_attempts: int = 1
    #: Cap on ``Retry-After``. A provider asking for a 20-minute wait should
    #: trigger failover, not a run that appears to have hung.
    max_retry_after_s: float = 60.0


@dataclass(frozen=True)
class GuardrailsConfig:
    token_budget: int = 200_000
    token_warn_ratio: float = 0.8
    wall_clock_timeout_s: float = 600.0
    #: How much of the draft is rendered into a prompt. Applied by Perceive,
    #: which is where the prompt view is built; measurements still cover the
    #: whole document.
    max_input_chars: int = 20_000
    #: Hard ingestion cap, applied once before the run starts. A different job
    #: from ``max_input_chars``: this one bounds what the process holds and
    #: diffs at all, so a 40 MB paste cannot make the loop crawl.
    max_document_chars: int = 200_000
    repeat_action_threshold: int = 3
    stuck_score_epsilon: float = 0.5
    #: Scorecards inspected by the stuck detector's plateau signal.
    stuck_score_window: int = 3


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    trace_dir: str = "runs"
    redact_keys: list[str] = field(default_factory=lambda: ["api_key", "authorization"])
    #: Write ``runs/<run_id>/trace.jsonl`` and ``summary.json``. Off makes the
    #: run leave no disk footprint, which the A/B demo script wants.
    trace_enabled: bool = True
    #: Characters of any single string field kept in the trace. Drafts are long
    #: and a trace nobody opens is not observability.
    trace_max_field_chars: int = 2000


@dataclass(frozen=True)
class AppConfig:
    """The single object handed to the loop."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    project_root: str = "."

    def path(self, relative: str) -> Path:
        """Resolve a config-declared relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (Path(self.project_root) / p).resolve()

    def with_overrides(self, **sections: t.Any) -> AppConfig:
        """Return a copy with whole sections replaced (used by tests)."""
        return replace(self, **sections)


# ---------------------------------------------------------------------------
# Generic dataclass hydration
# ---------------------------------------------------------------------------


def _is_union(origin: t.Any) -> bool:
    return origin is t.Union or origin is getattr(types, "UnionType", object())


def _to_bool(value: t.Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"cannot interpret {value!r} as a boolean")


def _coerce(tp: t.Any, value: t.Any, path: str) -> t.Any:
    """Convert a raw YAML/env value into the type the dataclass field declares."""
    origin = t.get_origin(tp)
    args = t.get_args(tp)

    if _is_union(origin):
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _coerce(non_none[0], value, path)

    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _build(tp, value, path)

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        val_t = args[1] if len(args) == 2 else t.Any
        return {k: _coerce(val_t, v, f"{path}.{k}") for k, v in value.items()}

    if origin is list:
        item_t = args[0] if args else t.Any
        # Env vars arrive as comma-separated strings; YAML gives real lists.
        raw = (
            [s.strip() for s in value.split(",") if s.strip()]
            if isinstance(value, str)
            else value
        )
        if not isinstance(raw, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(item_t, v, f"{path}[{i}]") for i, v in enumerate(raw)]

    if tp is bool:
        return _to_bool(value)
    if tp in (int, float, str):
        try:
            return tp(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path}: cannot convert {value!r} to {tp.__name__}") from exc
    return value


def _build(cls: t.Any, data: dict[str, t.Any], path: str = "") -> t.Any:
    hints = t.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {
        f.name: _coerce(hints[f.name], data[f.name], f"{path or cls.__name__}.{f.name}")
        for f in fields(cls)
        if f.name in data
    }
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Layer merging
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, t.Any], overlay: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_path(tree: dict[str, t.Any], parts: t.Sequence[str], value: t.Any) -> None:
    node = tree
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def env_overrides(env: t.Mapping[str, str] | None = None) -> dict[str, t.Any]:
    """Collect ``AGENTIC_*`` variables into a nested override tree."""
    source = env if env is not None else os.environ
    tree: dict[str, t.Any] = {}
    for raw_key, raw_value in source.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in raw_key[len(ENV_PREFIX) :].split(NEST_SEP) if p]
        if parts:
            _set_path(tree, parts, raw_value)
    return tree


def cli_overrides(pairs: t.Mapping[str, t.Any] | None) -> dict[str, t.Any]:
    """Turn ``{"loop.max_iterations": 3}`` into a nested override tree."""
    tree: dict[str, t.Any] = {}
    for dotted, value in (pairs or {}).items():
        if value is None:
            continue  # an unset CLI flag must not clobber the file value
        _set_path(tree, dotted.split("."), value)
    return tree


def load_config(
    path: str | Path | None = None,
    *,
    overrides: t.Mapping[str, t.Any] | None = None,
    env: t.Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> AppConfig:
    """Load, merge and validate configuration from all four layers."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, t.Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_path}: top level must be a mapping")
        raw = loaded
    elif path is not None:
        raise ConfigError(f"config file not found: {config_path}")

    merged = _deep_merge(raw, env_overrides(env))
    merged = _deep_merge(merged, cli_overrides(overrides))

    root = Path(project_root) if project_root else config_path.resolve().parent.parent
    merged.setdefault("project_root", str(root))

    return t.cast("AppConfig", _build(AppConfig, merged))


__all__ = [
    "AppConfig",
    "ConfigError",
    "GuardrailsConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "LoggingConfig",
    "LoopConfig",
    "MemoryConfig",
    "RetryConfig",
    "load_config",
]
