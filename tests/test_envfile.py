"""Regression tests for deterministic .env loading and UI port ownership."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_rubric.envfile import load_env_file
from agentic_rubric.web.server import ExclusiveThreadingHTTPServer, Handler


def test_blank_environment_value_does_not_shadow_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=gsk_from_file\n", encoding="utf-8")
    environment = {"GROQ_API_KEY": ""}

    report = load_env_file(path, environ=environment)

    assert environment["GROQ_API_KEY"] == "gsk_from_file"
    assert report.filled == ("GROQ_API_KEY",)
    assert report.overridden == ()
    assert report.notes and "set but empty" in report.notes[0]


def test_nonempty_environment_value_still_overrides_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=gsk_from_file\n", encoding="utf-8")
    environment = {"GROQ_API_KEY": "gsk_from_shell"}

    report = load_env_file(path, environ=environment)

    assert environment["GROQ_API_KEY"] == "gsk_from_shell"
    assert report.filled == ()
    assert report.overridden == ("GROQ_API_KEY",)


def test_missing_and_blank_file_values_are_reported(tmp_path: Path) -> None:
    missing = load_env_file(tmp_path / "missing.env", environ={})
    assert not missing.exists
    assert "no .env file exists" in missing.describes("GROQ_API_KEY")

    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=\n", encoding="utf-8")
    blank = load_env_file(path, environ={})
    assert blank.blank == ("GROQ_API_KEY",)
    assert "value is empty" in blank.describes("GROQ_API_KEY")


def test_http_server_owns_its_address_exclusively() -> None:
    first = ExclusiveThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = first.server_address[1]
    try:
        with pytest.raises(OSError):
            second = ExclusiveThreadingHTTPServer(("127.0.0.1", port), Handler)
            second.server_close()
    finally:
        first.server_close()


def test_exclusive_server_disables_reuse() -> None:
    assert ExclusiveThreadingHTTPServer.allow_reuse_address is False
    if os.name == "nt":
        assert hasattr(__import__("socket"), "SO_EXCLUSIVEADDRUSE")
