"""RMS-gated active duration helpers for hours-per-instrument analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.gm_programs import DRUM_GM_ID, gm_program_paper_label
from shared.config import SAMPLE_RATE

DEFAULT_MIN_RMS = 0.01
DEFAULT_HOP_SECONDS = 0.25


def active_seconds_from_mono(
    mono: np.ndarray,
    *,
    sample_rate: int,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
) -> tuple[float, float]:
    """Return ``(wall_seconds, active_seconds)`` using short-hop RMS gating."""
    n = int(mono.shape[0])
    if n <= 0 or sample_rate <= 0:
        return 0.0, 0.0
    wall = float(n) / float(sample_rate)
    hop = max(1, int(round(hop_seconds * sample_rate)))
    n_hops = n // hop
    if n_hops <= 0:
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        return wall, wall if rms >= min_rms else 0.0

    segment = mono[: n_hops * hop].reshape(n_hops, hop)
    hop_rms = np.sqrt(np.mean(np.square(segment, dtype=np.float64), axis=1))
    active_hops = int(np.count_nonzero(hop_rms >= min_rms))
    # Attribute any trailing remainder to the last hop's activity.
    rem = n - n_hops * hop
    active = active_hops * (hop / sample_rate)
    if rem > 0 and hop_rms[-1] >= min_rms:
        active += rem / sample_rate
    return wall, float(active)


def measure_stem_active_seconds(
    flac_path: Path,
    *,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    min_rms: float = DEFAULT_MIN_RMS,
) -> tuple[float, float] | None:
    """Decode one stem and return wall/active seconds, or ``None`` on failure."""
    try:
        audio, sr = sf.read(str(flac_path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        return None
    if audio.size == 0:
        return None
    sample_rate = int(sr) if sr else SAMPLE_RATE
    mono = np.mean(np.asarray(audio, dtype=np.float32), axis=1)
    return active_seconds_from_mono(
        mono,
        sample_rate=sample_rate,
        hop_seconds=hop_seconds,
        min_rms=min_rms,
    )


def gm_id_from_stem_row(program: object, is_drum: object) -> int:
    if bool(is_drum):
        return DRUM_GM_ID
    try:
        return int(program)
    except (TypeError, ValueError):
        return -1


def aggregate_hours_by_program(stem_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-stem wall/active seconds into hours by GM program."""
    if stem_rows.empty:
        return pd.DataFrame(
            columns=[
                "gm_id",
                "label",
                "n_stems",
                "wall_hours",
                "active_hours",
                "active_frac",
            ]
        )

    grouped = (
        stem_rows.groupby("gm_id", sort=False)
        .agg(
            n_stems=("active_seconds", "size"),
            wall_seconds=("wall_seconds", "sum"),
            active_seconds=("active_seconds", "sum"),
        )
        .reset_index()
    )
    grouped["label"] = grouped["gm_id"].map(lambda g: gm_program_paper_label(int(g)))
    grouped["wall_hours"] = grouped["wall_seconds"] / 3600.0
    grouped["active_hours"] = grouped["active_seconds"] / 3600.0
    grouped["active_frac"] = np.where(
        grouped["wall_seconds"] > 0,
        grouped["active_seconds"] / grouped["wall_seconds"],
        0.0,
    )
    return grouped.sort_values("active_hours", ascending=False).reset_index(drop=True)
