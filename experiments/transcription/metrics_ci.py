"""Bootstrap CIs over per-track YourMT3 metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_per_track_rows(path: Path | Iterable[Path]) -> pd.DataFrame:
    paths = [path] if isinstance(path, Path) else list(path)
    rows: list[dict] = []
    for p in paths:
        if not p.is_file():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["track_id", "onset_f", "multi_f", "offset_f"])
    return pd.DataFrame(rows)


def _finite(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return vals[np.isfinite(vals)]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 43,
) -> dict[str, float | int]:
    """Mean and percentile bootstrap CI; NaNs already dropped."""
    n = int(values.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "ci_half": float("nan"),
            "n": 0,
        }
    if n == 1:
        m = float(values[0])
        return {"mean": m, "ci_lo": m, "ci_hi": m, "ci_half": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    m = float(values.mean())
    return {
        "mean": m,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_half": float(0.5 * (hi - lo)),
        "n": n,
    }


def summarize_per_track(
    df: pd.DataFrame,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 43,
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for col in ("onset_f", "multi_f", "offset_f"):
        if col not in df.columns:
            continue
        out[col] = bootstrap_mean_ci(_finite(df[col]), n_boot=n_boot, alpha=alpha, seed=seed)
    return out


def merge_rank_jsonls(per_track_dir: Path) -> Path:
    """Merge ``per_track_rank*.jsonl`` into ``per_track.jsonl`` (dedupe by track_id)."""
    parts = sorted(per_track_dir.glob("per_track_rank*.jsonl"))
    existing = per_track_dir / "per_track.jsonl"
    if not parts and existing.is_file():
        return existing
    df = load_per_track_rows(parts)
    if "track_id" in df.columns and not df.empty:
        df = df.drop_duplicates(subset=["track_id"], keep="last")
    out = per_track_dir / "per_track.jsonl"
    with open(out, "w") as f:
        for row in df.to_dict(orient="records"):
            clean = {}
            for k, v in row.items():
                if isinstance(v, float) and not np.isfinite(v):
                    clean[k] = None
                elif pd.isna(v):
                    clean[k] = None
                else:
                    clean[k] = v
            f.write(json.dumps(clean) + "\n")
    return out
