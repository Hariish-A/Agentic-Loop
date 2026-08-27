"""Prompt templates, one module per LLM-touching step.

Kept out of the step implementations so a prompt can be inspected, diffed and
reasoned about on its own. Each module exposes `build_messages(...)` and, where
the step needs structured output, the `ToolSpec` that forces it.
"""

from . import reason, reflect, revise, score

__all__ = ["reason", "reflect", "revise", "score"]
