"""One HTTP client for every OpenAI-compatible ``/chat/completions`` endpoint.

Gemini (via its OpenAI compatibility layer), xAI/Grok, Ollama, OpenAI,
OpenRouter and Groq all accept the same request shape, so a single ~200-line
client plus a config record covers all of them. Written against ``httpx``
rather than a vendor SDK on purpose: the whole point of this layer is to map
provider-specific HTTP outcomes onto *our* error taxonomy, and going through an
SDK would mean re-deriving that mapping from four different exception trees.

Transport-level retries are disabled here. Retrying is the harness's job
(Milestone 3), so that backoff, jitter and the token budget are decided in one
place instead of two.
"""

from __future__ import annotations

import time
import typing as t

import httpx

from .base import LLMProvider, ToolChoice
from .parsing import parse_tool_arguments
from .types import (
    AuthError,
    BadRequestError,
    LLMParseError,
    LLMResponse,
    LLMTimeoutError,
    Message,
    ProviderUnavailableError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    TransientServerError,
    Usage,
)

DEFAULT_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_AUTH_STATUS = frozenset({401, 403})


class OpenAICompatibleProvider(LLMProvider):
    """Chat completions against any endpoint speaking the OpenAI dialect."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        supports_tools: bool = True,
        retry_on_status: t.Iterable[int] = DEFAULT_RETRY_STATUS,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ProviderUnavailableError(f"provider {name!r} has no base_url", provider=name)
        self.name = name
        self.model = model
        self.supports_tools = supports_tools
        self._retry_status = frozenset(retry_on_status)
        self._owns_client = client is None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout_s,
            # Our own decorator owns retry policy; do not double-retry here.
            transport=httpx.HTTPTransport(retries=0),
        )

    # -- request ------------------------------------------------------------

    def _payload(
        self,
        messages: t.Sequence[Message],
        tools: t.Sequence[ToolSpec] | None,
        tool_choice: ToolChoice,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, t.Any]:
        body: dict[str, t.Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools and self.supports_tools:
            body["tools"] = [tool.to_wire() for tool in tools]
            choice = _normalise_tool_choice(tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        return body

    def complete(
        self,
        messages: t.Sequence[Message],
        *,
        tools: t.Sequence[ToolSpec] | None = None,
        tool_choice: ToolChoice = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        body = self._payload(messages, tools, tool_choice, temperature, max_tokens)
        started = time.perf_counter()
        try:
            response = self._client.post("chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self.name}: request timed out", provider=self.name) from exc
        except httpx.ConnectError as exc:
            # Nothing is listening -- fail over rather than retry this provider.
            raise ProviderUnavailableError(
                f"{self.name}: cannot reach {self._client.base_url}", provider=self.name
            ) from exc
        except httpx.RequestError as exc:
            raise TransientServerError(
                f"{self.name}: transport error: {exc}", provider=self.name
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400:
            raise self._to_error(response)
        return self._to_response(response, body, latency_ms)

    # -- error mapping ------------------------------------------------------

    def _to_error(self, response: httpx.Response) -> Exception:
        status = response.status_code
        detail = _error_detail(response)
        message = f"{self.name}: HTTP {status}: {detail}"

        if status in _AUTH_STATUS:
            return AuthError(message, provider=self.name, status=status)
        if status in self._retry_status:
            if status == 429:
                return RateLimitError(
                    message,
                    provider=self.name,
                    status=status,
                    retry_after_s=_retry_after(response),
                )
            return TransientServerError(
                message, provider=self.name, status=status, retry_after_s=_retry_after(response)
            )
        return BadRequestError(message, provider=self.name, status=status)

    # -- response parsing ---------------------------------------------------

    def _to_response(
        self, response: httpx.Response, request_body: dict[str, t.Any], latency_ms: float
    ) -> LLMResponse:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMParseError(
                f"{self.name}: response body was not JSON", raw=response.text, provider=self.name
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMParseError(
                f"{self.name}: response contained no choices", raw=response.text, provider=self.name
            )
        message = choices[0].get("message") or {}
        text = message.get("content") or ""

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            fn = raw_call.get("function") or {}
            raw_args = fn.get("arguments") or ""
            arguments, error = parse_tool_arguments(raw_args)
            if error:
                # Surface as a parse failure so the repair ladder can act on it;
                # the raw text is preserved for the repair prompt.
                raise LLMParseError(
                    f"{self.name}: tool call {fn.get('name', '?')!r} had unusable arguments: "
                    f"{error}",
                    raw=raw_args,
                    provider=self.name,
                )
            calls.append(
                ToolCall(
                    id=raw_call.get("id") or f"call_{index}",
                    name=fn.get("name", ""),
                    arguments=arguments,
                    raw_arguments=raw_args,
                )
            )

        return LLMResponse(
            text=text,
            tool_calls=tuple(calls),
            usage=_usage(data, request_body, text),
            model=data.get("model") or self.model,
            provider=self.name,
            finish_reason=choices[0].get("finish_reason") or "",
            latency_ms=latency_ms,
            raw=data,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_tool_choice(choice: ToolChoice) -> t.Any:
    """Map our compact tool-choice vocabulary onto the OpenAI wire format."""
    if choice is None:
        return None
    if choice in ("auto", "required", "none"):
        return choice
    # Anything else is read as "force this specific tool".
    return {"type": "function", "function": {"name": choice}}


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:400]
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:400]
    return str(error or payload)[:400]


def _retry_after(response: httpx.Response) -> float | None:
    """Prefer the provider's own backoff hint over our computed delay."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _usage(data: dict[str, t.Any], request_body: dict[str, t.Any], text: str) -> Usage:
    reported = data.get("usage") or {}
    prompt = reported.get("prompt_tokens")
    completion = reported.get("completion_tokens")
    if prompt is None and completion is None:
        # Ollama and some proxies omit usage; estimate so the token-budget
        # guardrail still has a number to work with.
        prompt_chars = sum(len(str(m.get("content") or "")) for m in request_body["messages"])
        return Usage.estimate(prompt_chars, len(text))
    return Usage(input_tokens=int(prompt or 0), output_tokens=int(completion or 0))


__all__ = ["DEFAULT_RETRY_STATUS", "OpenAICompatibleProvider"]
