"""Provider-agnostic LLM access layer.

Public surface: the :class:`~.base.LLMProvider` ABC, the value objects in
:mod:`.types`, and :func:`~.factory.build_provider`. Concrete backends are an
implementation detail the loop never imports directly.
"""

from .base import LLMProvider, ToolChoice
from .factory import availability, available_chain, build_provider
from .mock import MockProvider, MockTurn, tool_call
from .openai_compatible import OpenAICompatibleProvider
from .parsing import parse_tool_arguments, salvage_json
from .types import (
    AuthError,
    BadRequestError,
    LLMError,
    LLMParseError,
    LLMResponse,
    LLMTimeoutError,
    Message,
    ProviderUnavailableError,
    RateLimitError,
    RetryableLLMError,
    TerminalLLMError,
    ToolCall,
    ToolSpec,
    TransientServerError,
    Usage,
    assistant,
    system,
    tool_result,
    user,
)

__all__ = [
    "AuthError",
    "BadRequestError",
    "LLMError",
    "LLMParseError",
    "LLMProvider",
    "LLMResponse",
    "LLMTimeoutError",
    "Message",
    "MockProvider",
    "MockTurn",
    "OpenAICompatibleProvider",
    "ProviderUnavailableError",
    "RateLimitError",
    "RetryableLLMError",
    "TerminalLLMError",
    "ToolCall",
    "ToolChoice",
    "ToolSpec",
    "TransientServerError",
    "Usage",
    "assistant",
    "availability",
    "available_chain",
    "build_provider",
    "parse_tool_arguments",
    "salvage_json",
    "system",
    "tool_call",
    "tool_result",
    "user",
]
