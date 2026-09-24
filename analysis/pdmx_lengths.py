"""Load and summarize PDMX song lengths from metadata."""

from __future__ import annotations

import pandas as pd

from shared.config import (
    MAX_STEM_DURATION,
    PDMX_FILEPATH,
)

SONG_LENGTH_COLUMN = "song_length.seconds"
PERCENTILE_LEVELS = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)


def load_pdmx_song_lengths(
    dataset_filepath: str = PDMX_FILEPATH,
    *,
    valid_only: bool = True,
) -> pd.Series:
    """Return effective song durations in seconds (capped at synthesis max)."""
    df = pd.read_csv(dataset_filepath, usecols=["subset:all_valid", SONG_LENGTH_COLUMN])
    if valid_only:
        df = df[df["subset:all_valid"]]
    durations = pd.to_numeric(df[SONG_LENGTH_COLUMN], errors="coerce").dropna()
    return durations.clip(upper=MAX_STEM_DURATION).reset_index(drop=True)


def _percentile_key(level: float) -> str:
    pct = round(level * 100, 1)
    if pct == int(pct):
        return f"p{int(pct)}"
    return f"p{str(pct).replace('.', '_')}"


def percentile_table(durations: pd.Series) -> dict[str, float]:
    return {
        _percentile_key(level): round(float(durations.quantile(level)), 2)
        for level in PERCENTILE_LEVELS
    }
