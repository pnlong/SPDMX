"""stable-audio-tools custom_metadata hook: path → prompt from dataset_meta.json."""

from __future__ import annotations

import json
from pathlib import Path


_META: dict[str, dict] | None = None


def _load(meta_json: str) -> dict[str, dict]:
    global _META
    if _META is None:
        with open(meta_json) as f:
            raw = json.load(f)
        # Resolve symlinks so lookup by real path works.
        _META = {}
        for path, info in raw.items():
            _META[path] = info
            try:
                _META[str(Path(path).resolve())] = info
            except OSError:
                pass
    return _META


def get_custom_metadata(info: dict, audio: object) -> dict:
    """Called by stable-audio-tools; ``info['path']`` is the audio file path."""
    args = info.get("custom_metadata_args") or {}
    meta_json = args.get("meta_json")
    if not meta_json:
        return {"prompt": "instrumental music"}
    meta = _load(meta_json)
    path = str(info.get("path") or "")
    resolved = path
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        pass
    entry = meta.get(path) or meta.get(resolved) or {}
    return {"prompt": entry.get("prompt") or "instrumental music"}
