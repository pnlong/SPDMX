"""Unit tests for transcription bootstrap CI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.transcription.metrics_ci import (
    bootstrap_mean_ci,
    load_per_track_rows,
    merge_rank_jsonls,
    summarize_per_track,
)


def test_bootstrap_mean_ci_basic():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.25, 0.05, size=200)
    out = bootstrap_mean_ci(vals, n_boot=2000, seed=0)
    assert out["n"] == 200
    assert abs(out["mean"] - vals.mean()) < 1e-9
    assert out["ci_lo"] < out["mean"] < out["ci_hi"]
    assert out["ci_half"] > 0


def test_merge_and_summarize(tmp_path: Path):
    for rank, tracks in enumerate((["a", "b"], ["b", "c"])):
        p = tmp_path / f"per_track_rank{rank}.jsonl"
        with open(p, "w") as f:
            for i, tid in enumerate(tracks):
                f.write(json.dumps({"track_id": tid, "onset_f": 0.1 * (i + 1), "multi_f": 0.2}) + "\n")
    merged = merge_rank_jsonls(tmp_path)
    df = load_per_track_rows(merged)
    assert set(df["track_id"]) == {"a", "b", "c"}
    summary = summarize_per_track(df, n_boot=500, seed=1)
    assert summary["onset_f"]["n"] == 3
    assert abs(summary["multi_f"]["mean"] - 0.2) < 1e-9
