"""Tool registry, argument validation and dispatch.

Three responsibilities, kept apart from the loop on purpose:

1. **Own the schemas.** One :class:`~..llm.types.ToolSpec` per tool drives the
   wire format, the argument validation and the documentation. There is no
   second place where a tool's parameters are described.
2. **Validate before dispatch.** A hand-rolled JSON Schema subset checks the
   model's arguments *before* a handler runs, so a malformed call produces a
   precise, actionable error string rather than a ``KeyError`` from inside a
   handler. That error goes back to the agent as an observation.
3. **Never crash the loop.** Every failure mode -- unknown tool, bad arguments,
   handler exception -- becomes an ``ActionResult(ok=False)``. A broken tool
   call is data the agent reacts to, not an exception that ends the run.

Validation is hand-rolled rather than pulled from ``jsonschema`` because the
subset needed here is small, and the error messages are better when written for
this audience: the next thing that reads them is an LLM deciding what to do.
"""

from __future__ import annotations

import time
import typing as t
from dataclasses import dataclass, field, replace

from ..config import AppConfig
from ..core.rubric import Rubric
from ..core.state import ActionResult, Decision, ErrorKind, Workspace
from ..llm.base import LLMProvider
from ..llm.types import LLMParseError, RetryableLLMError, ToolSpec, Usage

#: Every tool schema carries this. See :class:`~..core.state.Decision`.
THOUGHT_FIELD = "thought"

THOUGHT_PROPERTY = {
    "type": "string",
    "description": (
        "One or two sentences: why this action, right now, given the current "
        "scores. Required on every call."
    ),
}


class ToolError(Exception):
    """A handler failed in a way the agent should be told about, not shielded from."""

    def __init__(self, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable


@dataclass
class ToolContext:
    """Everything a handler is allowed to touch.

    Handlers receive this rather than the loop state, so a tool cannot decide
    what happens next -- it can only read config, read the rubric, edit the
    workspace, and call the model.
    """

    config: AppConfig
    rubric: Rubric
    workspace: Workspace
    llm: LLMProvider
    iteration: int = 0
    #: Token usage accumulated by LLM-backed tools, drained by the loop.
    usage: Usage = field(default_factory=Usage)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage


Handler = t.Callable[[dict[str, t.Any], ToolContext], "ToolOutput"]


@dataclass
class ToolOutput:
    """What a handler returns: a payload for the trace and a one-line summary.

    The summary is what lands in the ReAct scratchpad, so it must be short and
    information-dense -- it is the agent's only memory of what a tool did.
    """

    summary: str
    payload: dict[str, t.Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Minimal JSON Schema validation
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_errors(value: t.Any, spec: t.Mapping[str, t.Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = spec.get("type")
    if expected:
        allowed = _TYPE_MAP.get(expected, ())
        # bool is a subclass of int; do not let True satisfy "integer".
        if expected in ("number", "integer") and isinstance(value, bool):
            return [f"{path}: expected {expected}, got boolean"]
        if allowed and not isinstance(value, allowed):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]

    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: {value!r} is not one of {spec['enum']}")

    if expected == "array":
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(value):
                errors.extend(_type_errors(item, item_spec, f"{path}[{index}]"))
        if "minItems" in spec and len(value) < spec["minItems"]:
            errors.append(f"{path}: needs at least {spec['minItems']} item(s)")

    if expected in ("number", "integer"):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {spec['maximum']}")

    if expected == "string" and "minLength" in spec and len(value) < spec["minLength"]:
        errors.append(f"{path}: must be at least {spec['minLength']} character(s)")

    return errors


def validate_arguments(
    schema: t.Mapping[str, t.Any], arguments: t.Mapping[str, t.Any]
) -> list[str]:
    """Check arguments against a JSON Schema subset. Returns readable errors."""
    errors: list[str] = []
    properties: dict[str, t.Any] = schema.get("properties", {})

    for name in schema.get("required", []):
        if name not in arguments or arguments[name] in (None, ""):
            errors.append(f"missing required argument {name!r}")

    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                errors.append(
                    f"unknown argument {name!r}; accepted: {sorted(properties) or 'none'}"
                )

    for name, value in arguments.items():
        spec = properties.get(name)
        if isinstance(spec, dict) and value is not None:
            errors.extend(_type_errors(value, spec, name))

    return errors


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_exception(exc: BaseException) -> ErrorKind:
    """Decide what the harness may do about an exception a handler raised.

    Two of the five tools call an LLM, so a rate limit inside a handler is a
    *tool* failure that a retry would fix. The registry contains the exception
    to keep the loop alive, and containment destroys the type -- so the verdict
    is taken here, while the exception object is still in hand.
    """
    if isinstance(exc, RetryableLLMError):
        return ErrorKind.TRANSIENT
    if isinstance(exc, LLMParseError):
        # The model produced something unusable; a fresh sample often differs.
        return ErrorKind.TRANSIENT
    if isinstance(exc, (KeyError, TypeError, AttributeError)):
        return ErrorKind.TERMINAL  # a bug in the handler, not bad luck
    return ErrorKind.TERMINAL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def dispatch_schema(spec: ToolSpec) -> dict[str, t.Any]:
    """The spec's schema minus ``thought``.

    Reason consumes ``thought`` onto the Decision before dispatch, so the
    handler never sees it. Validating against the unmodified schema would then
    reject every well-formed call for a missing required argument -- the schema
    the model is asked to satisfy and the schema the arguments are checked
    against are two different contracts, and this is the conversion between them.
    """
    schema = dict(spec.parameters)
    properties = {
        k: v for k, v in schema.get("properties", {}).items() if k != THOUGHT_FIELD
    }
    required = [name for name in schema.get("required", []) if name != THOUGHT_FIELD]
    return {**schema, "properties": properties, "required": required}


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Handler
    terminal: bool = False  # ends the loop when it succeeds

    @property
    def validation_schema(self) -> dict[str, t.Any]:
        return dispatch_schema(self.spec)


class ToolRegistry:
    """Holds the tool set the Reason step may choose from."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Handler, *, terminal: bool = False) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler, terminal=terminal)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Tool schemas in registration order, for the LLM request."""
        return [entry.spec for entry in self._tools.values()]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def is_terminal(self, name: str) -> bool:
        entry = self._tools.get(name)
        return bool(entry and entry.terminal)

    def describe(self) -> str:
        """Compact catalogue rendered into the Reason prompt."""
        lines = []
        for entry in self._tools.values():
            required = [
                arg
                for arg in entry.spec.parameters.get("required", [])
                if arg != THOUGHT_FIELD
            ]
            args = ", ".join(required) or "no required arguments"
            summary = entry.spec.description.strip().splitlines()[0]
            lines.append(f"- {entry.spec.name}({args}): {summary}")
        return "\n".join(lines)

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, decision: Decision, ctx: ToolContext) -> ActionResult:
        """Run the chosen tool. Never raises: every failure becomes a result."""
        started = time.perf_counter()
        entry = self._tools.get(decision.action)

        if entry is None:
            return ActionResult.failure(
                decision.action,
                decision.arguments,
                f"unknown tool {decision.action!r}; available tools are {self.names}",
                kind=ErrorKind.UNKNOWN_TOOL,
            )

        # `thought` is captured on the Decision; handlers never see it.
        arguments = {k: v for k, v in decision.arguments.items() if k != THOUGHT_FIELD}

        errors = validate_arguments(entry.validation_schema, arguments)
        if errors:
            return ActionResult.failure(
                decision.action,
                arguments,
                "invalid arguments: " + "; ".join(errors),
                kind=ErrorKind.VALIDATION,
            )

        try:
            output = entry.handler(arguments, ctx)
        except ToolError as exc:
            result = ActionResult.failure(
                decision.action,
                arguments,
                str(exc),
                kind=ErrorKind.RECOVERABLE if exc.recoverable else ErrorKind.TERMINAL,
            )
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
            result = ActionResult.failure(
                decision.action,
                arguments,
                f"{type(exc).__name__}: {exc}",
                kind=classify_exception(exc),
            )
        else:
            result = ActionResult(
                action=decision.action,
                arguments=arguments,
                ok=True,
                output=output.payload,
                summary=output.summary,
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        return replace(result, duration_ms=duration_ms)


def build_spec(
    name: str,
    description: str,
    properties: dict[str, t.Any],
    required: list[str] | None = None,
) -> ToolSpec:
    """Assemble a ToolSpec with the shared ``thought`` field attached.

    Centralised so no tool can be added that forgets to capture its reasoning.
    """
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {THOUGHT_FIELD: THOUGHT_PROPERTY, **properties},
            "required": [THOUGHT_FIELD, *(required or [])],
            "additionalProperties": False,
        },
    )


__all__ = [
    "THOUGHT_FIELD",
    "classify_exception",
    "Handler",
    "RegisteredTool",
    "ToolContext",
    "ToolError",
    "ToolOutput",
    "ToolRegistry",
    "build_spec",
    "validate_arguments",
]
