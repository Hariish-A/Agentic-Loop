"""Config layering, coercion and validation."""

from __future__ import annotations

import pytest

from agentic_rubric.config import ConfigError, load_config

CONFIG = "config/config.yaml"


def test_loads_shipped_config() -> None:
    config = load_config(CONFIG)
    assert config.llm.primary == "groq"
    assert config.llm.chain == ["groq", "ollama"]
    assert config.loop.max_iterations == 6
    assert config.guardrails.token_budget == 200_000


def test_missing_file_is_an_error_when_explicitly_requested() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("config/does-not-exist.yaml")


def test_env_overrides_beat_the_file() -> None:
    config = load_config(
        CONFIG,
        env={"AGENTIC_LOOP__MAX_ITERATIONS": "11", "AGENTIC_LLM__PRIMARY": "ollama"},
    )
    assert config.loop.max_iterations == 11  # coerced from str to int
    assert config.llm.chain[0] == "ollama"


def test_env_reaches_arbitrary_nesting_depth() -> None:
    config = load_config(CONFIG, env={"AGENTIC_LLM__PROVIDERS__GROQ__MODEL": "openai/gpt-oss-20b"})
    assert config.llm.provider("groq").model == "openai/gpt-oss-20b"


def test_cli_overrides_beat_env() -> None:
    config = load_config(
        CONFIG,
        env={"AGENTIC_LOOP__MAX_ITERATIONS": "11"},
        overrides={"loop.max_iterations": 3},
    )
    assert config.loop.max_iterations == 3


def test_unset_cli_flags_do_not_clobber_the_file() -> None:
    # argparse hands us None for every flag the user omitted.
    config = load_config(CONFIG, overrides={"loop.max_iterations": None})
    assert config.loop.max_iterations == 6


def test_list_values_survive_env_as_csv() -> None:
    config = load_config(CONFIG, env={"AGENTIC_RETRY__RETRY_ON_STATUS": "429,503"})
    assert config.retry.retry_on_status == [429, 503]


def test_bool_coercion_accepts_human_spellings() -> None:
    assert load_config(CONFIG, env={"AGENTIC_MEMORY__ENABLED": "off"}).memory.enabled is False
    assert load_config(CONFIG, env={"AGENTIC_MEMORY__ENABLED": "yes"}).memory.enabled is True


def test_unknown_key_fails_loudly() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(CONFIG, overrides={"loop.max_iteration": 3})  # typo


def test_unknown_provider_names_the_configured_ones() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ConfigError, match="configured: groq"):
        config.llm.provider("nope")


def test_provider_chain_deduplicates() -> None:
    config = load_config(CONFIG, overrides={"llm.fallbacks": ["ollama", "groq", "ollama"]})
    assert config.llm.chain == ["groq", "ollama"]


def test_api_key_reads_from_the_named_variable() -> None:
    provider = load_config(CONFIG).llm.provider("groq")
    assert provider.api_key({"GROQ_API_KEY": "  abc  "}) == "abc"
    assert provider.api_key({"GROQ_API_KEY": "   "}) is None
    assert provider.api_key({}) is None
