"""Tests for repo .env path loading."""

import os

from shared.env import load_project_env, path_from_env


def test_path_from_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("SPDMX_OUTPUT_DIR", raising=False)
    assert path_from_env("SPDMX_OUTPUT_DIR", "/default/out") == "/default/out"


def test_path_from_env_expands_user(monkeypatch):
    monkeypatch.setenv("SPDMX_OUTPUT_DIR", "~/spdmx-out")
    assert path_from_env("SPDMX_OUTPUT_DIR", "/default") == os.path.expanduser(
        "~/spdmx-out"
    )


def test_load_project_env_reads_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SPDMX_OUTPUT_DIR=/from/dotenv\n", encoding="utf-8")
    monkeypatch.delenv("SPDMX_OUTPUT_DIR", raising=False)
    monkeypatch.setattr("shared.env", "_ENV_LOADED", False)
    load_project_env(repo_root=tmp_path)
    assert path_from_env("SPDMX_OUTPUT_DIR", "/default") == "/from/dotenv"
