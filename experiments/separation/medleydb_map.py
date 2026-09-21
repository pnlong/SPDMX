"""Map MedleyDB stem instrument labels to BDGP targets for SI-SDR eval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

# Case-insensitive instrument → BDGP. Unlisted instruments are ignored.
# Aligns with common MedleyDB → 6-stem (VDBO+GP) mappings.
_INSTRUMENT_TO_TARGET: dict[str, str] = {
    # bass
    "electric bass": "bass",
    "double bass": "bass",
    "acoustic bass": "bass",
    "bass synthesizer": "bass",
    # drums (kit / unpitched kit pieces that appear as stem labels)
    "drum set": "drums",
    "drum machine": "drums",
    "kick drum": "drums",
    "snare drum": "drums",
    "toms": "drums",
    "cymbal": "drums",
    "hi-hat": "drums",
    # guitar
    "acoustic guitar": "guitar",
    "clean electric guitar": "guitar",
    "distorted electric guitar": "guitar",
    "lap steel guitar": "guitar",
    # piano
    "piano": "piano",
    "electric piano": "piano",
    "tack piano": "piano",
}

_SKIP_INSTRUMENTS = frozenset({"main system", "unlabeled"})


def normalize_instrument(label: str | None) -> str:
    return str(label or "").strip().lower()


def instrument_to_target(label: str | None) -> str | None:
    key = normalize_instrument(label)
    if not key or key in _SKIP_INSTRUMENTS:
        return None
    return _INSTRUMENT_TO_TARGET.get(key)


def _stem_instruments(stem_info: dict) -> list[str]:
    raw = stem_info.get("instrument")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def load_track_metadata(metadata_dir: Path, track_id: str) -> dict | None:
    path = metadata_dir / f"{track_id}_METADATA.yaml"
    if not path.is_file():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def bdgp_stem_files(meta: dict) -> dict[str, list[str]]:
    """Return BDGP target → list of stem filenames (relative names)."""
    out: dict[str, list[str]] = {t: [] for t in ("bass", "drums", "guitar", "piano")}
    stems = meta.get("stems") or {}
    for stem_info in stems.values():
        if not isinstance(stem_info, dict):
            continue
        filename = stem_info.get("filename")
        if not filename:
            continue
        instruments = _stem_instruments(stem_info)
        # Prefer component=bass when present (MedleyDB bass annotation).
        component = normalize_instrument(stem_info.get("component"))
        targets: set[str] = set()
        if component == "bass":
            targets.add("bass")
        for inst in instruments:
            t = instrument_to_target(inst)
            if t:
                targets.add(t)
        # One stem should not map to multiple BDGP classes; take first hit in priority order.
        for t in ("bass", "drums", "guitar", "piano"):
            if t in targets:
                out[t].append(str(filename))
                break
    return {k: v for k, v in out.items() if v}


def iter_medleydb_track_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (track_id, track_dir) from V1/ and V2/ (or a flat root of track folders)."""
    subdirs = [root / "V1", root / "V2"]
    roots = [d for d in subdirs if d.is_dir()]
    if not roots and root.is_dir():
        roots = [root]
    seen: set[str] = set()
    for base in roots:
        for track_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            track_id = track_dir.name
            if track_id in seen:
                continue
            seen.add(track_id)
            yield track_id, track_dir
