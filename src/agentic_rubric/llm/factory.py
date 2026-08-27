"""Build LLM providers from configuration.

The loop asks the factory for "a provider", never for "a Groq client". That
indirection is what makes `AGENTIC_LLM__PRIMARY=ollama` a complete provider swap,
and it is where the Milestone 3 failover chain plugs in: the harness asks for
the ordered list of *available* providers and walks it on failure.
"""

from __future__ import annotations

import typing as t

from ..config import AppConfig, ConfigError, LLMProviderConfig
from .base import LLMProvider
from .openai_compatible import OpenAICompatibleProvider
from .types import ProviderUnavailableError

MOCK_PROVIDER_NAME = "mock"


def availability(
    provider: LLMProviderConfig, env: t.Mapping[str, str] | None = None
) -> tuple[bool, str]:
    """Can this provider be constructed at all? Returns ``(ok, reason)``.

    Checked before any network call so a missing key costs zero tokens and
    produces a readable reason instead of a 401 three retries later.
    """
    if not provider.base_url:
        return False, "no base_url configured"
    if provider.requires_key and not provider.api_key(env):
        return False, f"environment variable {provider.api_key_env or '<unset>'} is empty"
    return True, "ok"


def build_provider(
    config: AppConfig,
    name: str | None = None,
    *,
    env: t.Mapping[str, str] | None = None,
    mock_provider: LLMProvider | None = None,
) -> LLMProvider:
    """Construct one provider by config key.

    Raises :class:`~.types.ProviderUnavailableError` when the provider is
    configured but unusable, which the harness treats as "try the next one".
    """
    key = name or config.llm.primary
    settings = config.llm.provider(key)

    if key == MOCK_PROVIDER_NAME:
        if mock_provider is None:
            raise ConfigError(
                "provider 'mock' requires an explicit MockProvider instance; "
                "pass mock_provider=... (tests and the offline demo do this)"
            )
        return mock_provider

    ok, reason = availability(settings, env)
    if not ok:
        raise ProviderUnavailableError(f"provider {key!r} unavailable: {reason}", provider=key)

    return OpenAICompatibleProvider(
        name=key,
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key(env),
        timeout_s=settings.timeout_s,
        supports_tools=settings.supports_tools,
        supports_message_name=settings.supports_message_name,
        retry_on_status=config.retry.retry_on_status,
    )


def available_chain(
    config: AppConfig, *, env: t.Mapping[str, str] | None = None
) -> list[tuple[str, bool, str]]:
    """Report the configured failover chain and whether each link is usable.

    Returned rather than filtered so the startup log can show *why* a provider
    was skipped -- the single most common source of "it just doesn't work".
    """
    rows: list[tuple[str, bool, str]] = []
    for key in config.llm.chain:
        try:
            settings = config.llm.provider(key)
        except ConfigError as exc:
            rows.append((key, False, str(exc)))
            continue
        if key == MOCK_PROVIDER_NAME:
            rows.append((key, True, "scripted, offline"))
            continue
        ok, reason = availability(settings, env)
        rows.append((key, ok, reason))
    return rows


__all__ = ["MOCK_PROVIDER_NAME", "availability", "available_chain", "build_provider"]
