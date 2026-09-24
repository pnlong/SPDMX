"""Copy donor ablation stems into DDSP conditions (NFS-safe; no symlinks/hardlinks)."""

from __future__ import annotations

import shutil
from pathlib import Path

from shared.config import (
    FALLBACK_DONOR,
    RENDER_MODE_DDSP_BASIC,
    RENDER_MODE_DDSP_SLAKH,
    RENDER_MODE_SLAKH,
    RENDER_MODES,
)
from synthesis.audio import stem_path
from synthesis.paths import ablation_raw_dir, condition_name


def uses_ddsp(render_mode: str) -> bool:
    return render_mode in (RENDER_MODE_DDSP_BASIC, RENDER_MODE_DDSP_SLAKH)


def uses_slakh_recipes(render_mode: str) -> bool:
    """True when per-category locked soundfont/FX recipes apply (B1 and CB fallbacks)."""
    return render_mode in (RENDER_MODE_SLAKH, RENDER_MODE_DDSP_SLAKH)


def fallback_donor_mode(render_mode: str) -> str | None:
    return FALLBACK_DONOR.get(render_mode)


def reused_source_label(donor_mode: str) -> str:
    return f"reused:{donor_mode}"


def is_reused_source(source: str | None) -> bool:
    return bool(source) and str(source).startswith("reused:")


def donor_mode_from_source(source: str | None) -> str | None:
    if not is_reused_source(source):
        return None
    donor = str(source).split(":", 1)[1]
    return donor if donor in RENDER_MODES else None


def copy_stem(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` with metadata (NFS-safe duplicate file)."""
    dst = Path(dst)
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Donor stem missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    return dst


def song_rel_under_data(song_dir: Path | str, ablation_dir: Path | str) -> str:
    """Return path of ``song_dir`` relative to ``{ablation}/data/``."""
    song_dir = Path(song_dir)
    data_root = Path(ablation_dir) / "data"
    try:
        return str(song_dir.resolve().relative_to(data_root.resolve()))
    except ValueError:
        # Fallback: strip a ``.../data/`` segment from the absolute path.
        parts = song_dir.parts
        if "data" in parts:
            idx = parts.index("data")
            return str(Path(*parts[idx + 1 :]))
        raise


def donor_raw_stem_path(
    output_dir: str,
    donor_mode: str,
    song_rel: str,
    track: int,
    audio_format: str,
) -> Path:
    donor_dir = Path(ablation_raw_dir(output_dir, donor_mode))
    return stem_path(donor_dir / "data" / song_rel, track, audio_format)


def donor_condition_name(donor_mode: str) -> str:
    return condition_name(donor_mode)
