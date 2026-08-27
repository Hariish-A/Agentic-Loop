"""LLM layer: JSON salvage, error mapping, wire format, and the mock provider."""

from __future__ import annotations

import json

import httpx
import pytest

from agentic_rubric.config import load_config
from agentic_rubric.llm import (
    AuthError,
    BadRequestError,
    LLMParseError,
    MockProvider,
    MockTurn,
    OpenAICompatibleProvider,
    ProviderUnavailableError,
    RateLimitError,
    ToolSpec,
    TransientServerError,
    Usage,
    assistant,
    build_provider,
    salvage_json,
    system,
    tool_call,
    user,
)

CONFIG = "config/config.yaml"

SPEC = ToolSpec(
    name="score_against_rubric",
    description="Score text against a rubric.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here you go:\n{"a": 1}\nHope that helps.', {"a": 1}),
        ('{"a": 1,}', {"a": 1}),
        ('{"a": "}"}', {"a": "}"}),  # brace inside a string must not end the span
        ("[1, 2]", [1, 2]),
    ],
)
def test_salvage_json_recovers_common_model_output(raw: str, expected: object) -> None:
    assert salvage_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{unclosed"])
def test_salvage_json_gives_up_cleanly(raw: str) -> None:
    assert salvage_json(raw) is None


# --- wire format -----------------------------------------------------------


def test_assistant_tool_call_serialises_with_null_content() -> None:
    call = tool_call("finalize", reason="done")
    wire = assistant(tool_calls=[call]).to_wire()
    assert wire["content"] is None
    assert wire["tool_calls"][0]["function"]["name"] == "finalize"


def test_tool_spec_wire_shape() -> None:
    wire = SPEC.to_wire()
    assert wire["type"] == "function"
    assert wire["function"]["parameters"]["required"] == ["text"]


def test_tool_call_signature_is_order_independent() -> None:
    a = tool_call("revise_text", focus="thesis", tone="formal")
    b = tool_call("revise_text", tone="formal", focus="thesis")
    assert a.signature() == b.signature()  # the stuck-loop detector relies on this


def test_usage_addition_and_estimation() -> None:
    total = Usage(10, 5) + Usage(3, 2)
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (13, 7, 20)
    assert total.estimated is False
    assert Usage.estimate(400, 40).estimated is True


# --- HTTP client -----------------------------------------------------------


def _provider(handler: object, **kwargs: object) -> OpenAICompatibleProvider:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        base_url="https://example.test/v1",
    )
    return OpenAICompatibleProvider(
        name="test", base_url="https://example.test/v1", model="m", client=client, **kwargs
    )


def _ok_body(**overrides: object) -> dict:
    body = {
        "model": "m",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }
    body.update(overrides)
    return body


def test_successful_completion_is_normalised() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_ok_body()))
    response = provider.complete([user("hi")])
    assert response.text == "hello"
    assert response.usage.total_tokens == 15
    assert response.provider == "test"
    assert response.latency_ms >= 0


def test_tool_calls_are_parsed_into_value_objects() -> None:
    body = _ok_body(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "score_against_rubric",
                                "arguments": '{"text": "abc"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))
    call = provider.complete([user("hi")], tools=[SPEC]).first_tool_call()
    assert call.name == "score_against_rubric"
    assert call.arguments == {"text": "abc"}


def test_unparseable_tool_arguments_raise_a_parse_error_carrying_the_raw_text() -> None:
    body = _ok_body(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "revise_text", "arguments": "{oops"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))
    with pytest.raises(LLMParseError) as excinfo:
        provider.complete([user("hi")], tools=[SPEC])
    assert excinfo.value.raw == "{oops"  # the repair prompt needs this


def test_missing_choices_is_a_parse_error_not_a_crash() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMParseError, match="no choices"):
        provider.complete([user("hi")])


def test_prose_when_a_tool_was_required_is_a_parse_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_ok_body()))
    with pytest.raises(LLMParseError, match="expected a tool call"):
        provider.complete([user("hi")], tools=[SPEC], tool_choice="required").first_tool_call()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RateLimitError),
        (503, TransientServerError),
        (500, TransientServerError),
        (401, AuthError),
        (403, AuthError),
        (400, BadRequestError),
        (404, BadRequestError),
    ],
)
def test_status_codes_map_onto_the_error_taxonomy(status: int, expected: type) -> None:
    provider = _provider(
        lambda request: httpx.Response(status, json={"error": {"message": "boom"}})
    )
    with pytest.raises(expected):
        provider.complete([user("hi")])


def test_retryable_statuses_come_from_config_not_hardcoding() -> None:
    # 418 is not retryable by default; declaring it in config makes it so.
    provider = _provider(lambda request: httpx.Response(418, text="teapot"))
    with pytest.raises(BadRequestError):
        provider.complete([user("hi")])

    provider = _provider(lambda request: httpx.Response(418, text="teapot"), retry_on_status=[418])
    with pytest.raises(TransientServerError):
        provider.complete([user("hi")])


def test_rate_limit_carries_the_provider_retry_after_hint() -> None:
    provider = _provider(
        lambda request: httpx.Response(429, headers={"retry-after": "7.5"}, text="slow down")
    )
    with pytest.raises(RateLimitError) as excinfo:
        provider.complete([user("hi")])
    assert excinfo.value.retry_after_s == 7.5
    assert excinfo.value.retryable is True


def test_connection_failure_is_unavailable_so_the_harness_fails_over() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _provider(refuse).complete([user("hi")])


def test_timeout_is_retryable() -> None:
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(Exception) as excinfo:
        _provider(stall).complete([user("hi")])
    assert excinfo.value.retryable is True  # type: ignore[attr-defined]


def test_forcing_a_named_tool_sends_the_openai_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_body())

    _provider(handler).complete([user("hi")], tools=[SPEC], tool_choice="score_against_rubric")
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "score_against_rubric"},
    }


def test_missing_usage_falls_back_to_an_estimate() -> None:
    body = _ok_body()
    del body["usage"]
    provider = _provider(lambda request: httpx.Response(200, json=body))
    usage = provider.complete([user("hi" * 100)]).usage
    assert usage.estimated is True
    assert usage.input_tokens > 0


# --- mock provider ---------------------------------------------------------


def test_mock_replays_its_script_in_order() -> None:
    provider = MockProvider([MockTurn(text="one"), MockTurn(text="two")])
    assert provider.complete([user("a")]).text == "one"
    assert provider.complete([user("b")]).text == "two"
    assert provider.call_count == 2


def test_mock_can_inject_a_failure_at_a_chosen_turn() -> None:
    provider = MockProvider(
        [MockTurn(text="fine"), MockTurn(raises=RateLimitError("429", provider="mock"))]
    )
    provider.complete([user("a")])
    with pytest.raises(RateLimitError):
        provider.complete([user("b")])


def test_mock_responder_can_react_to_conversation_state() -> None:
    def responder(call):  # noqa: ANN001 - test-local closure
        last = call.messages[-1].content
        return MockTurn(text=f"saw:{last}")

    provider = MockProvider(responder=responder)
    assert provider.complete([system("s"), user("hello")]).text == "saw:hello"


def test_exhausted_script_fails_loudly() -> None:
    provider = MockProvider([MockTurn(text="only")])
    provider.complete([user("a")])
    with pytest.raises(ProviderUnavailableError, match="exhausted"):
        provider.complete([user("b")])


# --- factory ---------------------------------------------------------------


def test_factory_refuses_a_provider_with_no_key() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ProviderUnavailableError, match="GROQ_API_KEY"):
        build_provider(config, "groq", env={})


def test_factory_builds_a_keyed_provider() -> None:
    config = load_config(CONFIG)
    provider = build_provider(config, "groq", env={"GROQ_API_KEY": "gsk-test"})
    try:
        assert provider.name == "groq"
        assert provider.describe() == "groq:openai/gpt-oss-120b"
    finally:
        provider.close()


def test_local_provider_needs_no_key() -> None:
    config = load_config(CONFIG)
    provider = build_provider(config, "ollama", env={})
    try:
        assert provider.name == "ollama"
    finally:
        provider.close()


def test_factory_accepts_an_injected_mock() -> None:
    config = load_config(CONFIG)
    mock = MockProvider([MockTurn(text="hi")])
    assert build_provider(config, "mock", mock_provider=mock) is mock

def test_an_empty_truncated_completion_is_a_parse_error() -> None:
    """A reasoning model can return 200 OK with nothing in it.

    Groq's gpt-oss spends the output budget on an internal `reasoning` field
    before emitting content, so a tight max_tokens yields an empty `content`
    and finish_reason=length. Returning that as a valid response would let the
    reviser replace the user's draft with an empty string.
    """
    body = _ok_body(
        choices=[
            {
                "message": {"role": "assistant", "content": "", "reasoning": "hmm"},
                "finish_reason": "length",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))
    with pytest.raises(LLMParseError) as caught:
        provider.complete([user("hi")])
    assert "truncated" in str(caught.value)
    assert "max_tokens" in str(caught.value)


def test_a_truncated_completion_with_content_is_still_returned() -> None:
    """Partial text is the caller's problem to judge, not the client's."""
    body = _ok_body(
        choices=[
            {
                "message": {"role": "assistant", "content": "half an ans"},
                "finish_reason": "length",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))
    response = provider.complete([user("hi")])
    assert response.text == "half an ans"
    assert response.finish_reason == "length"
