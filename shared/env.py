"""Load repo ``.env`` and resolve filesystem path settings."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # e.g. isolated .venv-ddsp TF worker
    load_dotenv = None

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def repo_root() -> Path:
    return _REPO_ROOT


def load_project_env(*, repo_root: Path | None = None) -> None:
    """Load ``.env`` from the repo root once per process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    root = repo_root or _REPO_ROOT
    if load_dotenv is not None:
        load_dotenv(root / ".env")
    _ENV_LOADED = True


def path_from_env(name: str, default: str) -> str:
    """Read a filesystem path from the environment (after ``load_project_env``)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(Path(os.path.expanduser(str(raw).strip())))
