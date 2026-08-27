"""Loading ``.env``, and being able to say what happened.

This module exists because of one silent failure that cost a real debugging
session. ``python-dotenv``'s ``load_dotenv`` does **not** overwrite a variable
that is already present in the environment -- which is correct, since an
explicit ``export`` should beat a file. But it makes no distinction between
"already set to a real value" and "already set to an empty string".

A blank ``GROQ_API_KEY`` left over in a shell, a Windows user-environment entry,
or a sourced ``.env.example`` therefore *shadows* the real key sitting in
``.env``. Everything then reports, perfectly accurately and completely
uselessly, that the variable is empty -- and the advice ("put your key in
``.env``") describes something the user has already done.

Two rules follow:

1. **An empty variable carries no information, so it does not shadow the file.**
   A non-empty one still wins: a deliberate override stays a deliberate
   override.
2. **Loading reports what it did.** Which file, whether it existed, which keys
   it defined, and which values it had to fill in because the environment's were
   blank. Callers surface that instead of guessing.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

ENV_FILENAME = ".env"


@dataclass(frozen=True)
class EnvFileReport:
    """What one attempt to load a ``.env`` file actually did."""

    path: Path
    exists: bool
    #: Keys the file defines, whatever became of them.
    keys: tuple[str, ...] = ()
    #: Keys taken from the file because the process had no usable value.
    filled: tuple[str, ...] = ()
    #: Keys the file defines but which a non-empty environment value overrode.
    overridden: tuple[str, ...] = ()
    #: Keys the file defines with an empty value.
    blank: tuple[str, ...] = ()
    #: Keys that had a usable effective value after this load. Kept internal to
    #: the report; the browser summary deliberately does not expose it.
    available: tuple[str, ...] = ()
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def describes(self, variable: str) -> str:
        """One sentence about ``variable``, for an error a human has to act on.

        This is the whole point of the module: when a key is missing, say which
        of the several quite different reasons applies.
        """
        if not variable:
            return "no environment variable is configured for this provider"
        if variable in self.available:
            source = "the .env file" if variable in self.filled else "the environment"
            return f"{variable} is set from {source}"
        if not self.exists:
            return (
                f"{variable} is not set, and no {ENV_FILENAME} file exists at {self.path}. "
                f"Copy {ENV_FILENAME}.example to {ENV_FILENAME} and put the key in it."
            )
        if variable in self.blank:
            return f"{variable} is present in {self.path} but its value is empty."
        if variable not in self.keys:
            return (
                f"{variable} is not set, and {self.path} does not define it "
                f"(it defines: {', '.join(self.keys) or 'nothing'})."
            )
        return f"{variable} could not be read from {self.path}."


def _parse(path: Path) -> tuple[dict[str, str], str]:
    """Read a ``.env`` into a mapping without touching ``os.environ``."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}, (
            "python-dotenv is not installed, so .env was not read; "
            "export the keys in your shell, or pip install -r requirements.txt"
        )
    try:
        values = dotenv_values(path)
    except OSError as exc:
        return {}, f"could not read {path}: {exc}"
    return {k: (v or "") for k, v in values.items()}, ""


def load_env_file(
    path: str | Path | None = None, *, environ: dict[str, str] | None = None
) -> EnvFileReport:
    """Load ``.env`` into the environment, and report exactly what happened.

    Unlike a bare ``load_dotenv``, a variable that is present but **blank** is
    treated as absent and filled from the file. See the module docstring for the
    debugging session that motivated it.
    """
    target = Path(path) if path is not None else Path(ENV_FILENAME)
    env: t.MutableMapping[str, str]
    if environ is None:
        import os

        env = os.environ
    else:
        env = environ

    def available_keys() -> tuple[str, ...]:
        return tuple(sorted(key for key, value in env.items() if value.strip()))

    if not target.is_file():
        return EnvFileReport(path=target, exists=False, available=available_keys())

    values, error = _parse(target)
    if error:
        return EnvFileReport(
            path=target,
            exists=True,
            error=error,
            available=available_keys(),
        )

    filled: list[str] = []
    overridden: list[str] = []
    blank: list[str] = []
    #: Keys that were present in the environment but empty -- the shadowing case
    #: this module exists to fix. Recorded *before* the value is filled in,
    #: because afterwards every filled key looks present.
    unshadowed: list[str] = []

    for key, value in values.items():
        if not value.strip():
            blank.append(key)
            continue
        current = env.get(key, "")
        if current.strip():
            if current != value:
                overridden.append(key)
            continue
        if key in env:
            unshadowed.append(key)
        env[key] = value
        filled.append(key)

    notes: list[str] = []
    if unshadowed:
        # A plain first load is unremarkable and gets no line. This one is not:
        # something in the environment was actively hiding a working key.
        notes.append(
            f"{', '.join(sorted(unshadowed))} was set but empty in the environment, "
            f"which would have hidden the value in {target}; the file's value is being used"
        )
    if overridden:
        notes.append(
            f"the environment already sets {', '.join(sorted(overridden))}, "
            f"so the value in {target} was ignored"
        )

    return EnvFileReport(
        path=target,
        exists=True,
        keys=tuple(values),
        filled=tuple(filled),
        overridden=tuple(overridden),
        blank=tuple(blank),
        available=available_keys(),
        notes=notes,
    )


def summary(report: EnvFileReport) -> dict[str, t.Any]:
    """A JSON-friendly view, for the application's Setup panel."""
    return {
        "path": str(report.path),
        "exists": report.exists,
        "keys": list(report.keys),
        "filled": list(report.filled),
        "overridden": list(report.overridden),
        "blank": list(report.blank),
        "error": report.error,
        "notes": list(report.notes),
    }


__all__ = ["ENV_FILENAME", "EnvFileReport", "load_env_file", "summary"]
