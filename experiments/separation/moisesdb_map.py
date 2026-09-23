"""Locate MoisesDB tracks and BDGP stem WAVs on disk.

Official zip unpacks to::

    <root>/moisesdb/moisesdb_v0.1/<track-uuid>/
        data.json
        bass/*.wav
        drums/*.wav
        ...

``SPDMX_MOISESDB_ROOT`` may point at ``<root>``, ``<root>/moisesdb``, or
``moisesdb_v0.1`` itself. Stem folders already use Moises top-level names
(``bass``, ``drums``, ``guitar``, ``piano``, …); BDGP uses those four when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

MOISES_BDGP = ("bass", "drums", "guitar", "piano")


def resolve_moises_version_root(moises_root: Path) -> Path | None:
    """Return the ``moisesdb_v0.1`` directory, or None if not found/empty."""
    if not moises_root.is_dir():
        return None
    candidates: list[Path] = []
    if moises_root.name == "moisesdb_v0.1":
        candidates.append(moises_root)
    candidates.extend(
        [
            moises_root / "moisesdb_v0.1",
            moises_root / "moisesdb" / "moisesdb_v0.1",
        ]
    )
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if not cand.is_dir():
            continue
        try:
            if any(cand.iterdir()):
                return cand
        except OSError:
            continue
    return None


def iter_moises_track_dirs(version_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(track_id, track_dir)`` for dirs that contain ``data.json``."""
    try:
        children = sorted(p for p in version_root.iterdir() if p.is_dir())
    except OSError:
        return
    for track_dir in children:
        if (track_dir / "data.json").is_file():
            yield track_dir.name, track_dir


def _stem_wavs_from_meta(track_dir: Path, stem_name: str, meta: dict) -> list[Path]:
    """Resolve WAV paths for one top-level stem via ``data.json``, else glob."""
    paths: list[Path] = []
    for stem in meta.get("stems") or []:
        if not isinstance(stem, dict):
            continue
        if str(stem.get("stemName") or "") != stem_name:
            continue
        for track in stem.get("tracks") or []:
            if not isinstance(track, dict):
                continue
            tid = track.get("id")
            ext = track.get("extension") or "wav"
            if not tid:
                continue
            paths.append(track_dir / stem_name / f"{tid}.{ext}")
    if paths:
        return paths
    stem_dir = track_dir / stem_name
    if stem_dir.is_dir():
        return sorted(stem_dir.glob("*.wav"))
    return []


def all_stem_names(meta: dict) -> list[str]:
    names: list[str] = []
    for stem in meta.get("stems") or []:
        if not isinstance(stem, dict):
            continue
        name = stem.get("stemName")
        if name and name not in names:
            names.append(str(name))
    return names


def load_track_data_json(track_dir: Path) -> dict | None:
    path = track_dir / "data.json"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def bdgp_stem_wavs(track_dir: Path, meta: dict) -> dict[str, list[Path]]:
    """Return present BDGP targets → existing WAV paths (skip missing files)."""
    out: dict[str, list[Path]] = {}
    for target in MOISES_BDGP:
        paths = [p for p in _stem_wavs_from_meta(track_dir, target, meta) if p.is_file()]
        if paths:
            out[target] = paths
    return out


def mixture_stem_wavs(track_dir: Path, meta: dict) -> list[Path]:
    """All stem WAVs for rebuilding the mixture (no dedicated mix file)."""
    paths: list[Path] = []
    for name in all_stem_names(meta):
        paths.extend(p for p in _stem_wavs_from_meta(track_dir, name, meta) if p.is_file())
    if paths:
        return paths
    # Fallback: every wav under stem subfolders.
    for child in sorted(p for p in track_dir.iterdir() if p.is_dir()):
        paths.extend(sorted(child.glob("*.wav")))
    return paths
