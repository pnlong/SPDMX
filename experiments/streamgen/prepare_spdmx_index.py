"""Build a Slakh-like SPDMX index for stream-music-gen adapters.

Writes per-song JSONL with stem FLAC paths, GM program, and mix duration so an
upstream ``Slakh2100``-style Dataset can be mirrored without forking their repo
until clone time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.streamgen.paths import REPO_ROOT, SPDMX_ROOT, load_config, resolve_dev_dir
from experiments.transcription.prepare_manifest import (
    _split_songs,
    hour_matched_subsample,
)


def build_spdmx_index(
    spdmx_root: Path,
    *,
    min_stems: int = 2,
    seed: int = 43,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    max_songs: int | None = None,
) -> list[dict]:
    songs = pd.read_csv(spdmx_root / "songs.csv")
    stems_path = spdmx_root / "stems.csv"
    if not stems_path.is_file():
        raise FileNotFoundError(stems_path)
    header = pd.read_csv(stems_path, nrows=0).columns.tolist()
    usecols = [c for c in ["song_id", "track", "program", "is_drum", "path"] if c in header]
    stems = pd.read_csv(stems_path, usecols=usecols)
    songs["n_tracks"] = pd.to_numeric(songs["n_tracks"], errors="coerce")
    songs["song_length"] = pd.to_numeric(songs.get("song_length"), errors="coerce")
    songs = songs[songs["n_tracks"] >= int(min_stems)].copy()
    if max_songs is not None:
        songs = songs.head(int(max_songs))

    keep_ids = set(songs["song_id"].astype(str))
    stems = stems[stems["song_id"].astype(str).isin(keep_ids)]
    stems_by_song = {sid: g for sid, g in stems.groupby(stems["song_id"].astype(str), sort=False)}

    splits = _split_songs(
        songs["song_id"].astype(str).tolist(),
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    split_of = {sid: sp for sp, ids in splits.items() for sid in ids}

    records = []
    for _, row in songs.iterrows():
        song_id = str(row["song_id"])
        mix = ""
        if pd.notna(row.get("chunk")):
            cand = spdmx_root / f"chunk_{int(row['chunk'])}" / song_id / "mix.flac"
            if cand.is_file():
                mix = str(cand)
        if not mix and "mix" in row and pd.notna(row["mix"]):
            mix = str(row["mix"])
        song_stems = stems_by_song.get(song_id)
        stem_list = []
        if song_stems is not None:
            for _, s in song_stems.iterrows():
                track = int(s["track"]) if "track" in s and pd.notna(s["track"]) else None
                audio = None
                if "path" in s and pd.notna(s["path"]):
                    p = Path(str(s["path"]))
                    if p.is_file():
                        audio = p
                if audio is None and track is not None and pd.notna(row.get("chunk")):
                    base = spdmx_root / f"chunk_{int(row['chunk'])}" / song_id
                    cand = base / f"{track}.flac"
                    if cand.is_file():
                        audio = cand
                stem_list.append(
                    {
                        "track": track,
                        "program": int(s["program"]) if pd.notna(s.get("program")) else None,
                        "is_drum": bool(s["is_drum"])
                        if "is_drum" in s and pd.notna(s["is_drum"])
                        else False,
                        "audio_path": str(audio) if audio else "",
                    }
                )
        records.append(
            {
                "corpus": "spdmx",
                "split": split_of.get(song_id, "train"),
                "song_id": song_id,
                "mix_path": mix,
                "n_stems": int(row["n_tracks"]),
                "duration_sec": float(row["song_length"])
                if pd.notna(row.get("song_length"))
                else None,
                "stems": stem_list,
            }
        )
    return records


def _resolve_out_dir(cfg: dict) -> Path:
    try:
        out = resolve_dev_dir(cfg) / "spdmx_index"
        out.mkdir(parents=True, exist_ok=True)
        return out
    except OSError:
        out = REPO_ROOT / "analysis" / "paper_data" / "streamgen_spdmx_index"
        out.mkdir(parents=True, exist_ok=True)
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--spdmx-root", type=Path, default=None)
    parser.add_argument("--max-songs", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--hour-matched", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = _resolve_out_dir(cfg)
    spdmx_root = Path(args.spdmx_root or cfg.get("spdmx_root") or SPDMX_ROOT)

    print(f"building SPDMX StreamGen index from {spdmx_root} …")
    records = build_spdmx_index(
        spdmx_root,
        min_stems=int(cfg.get("min_stems", 2)),
        seed=int(cfg.get("seed", 43)),
        val_frac=float(cfg.get("val_fraction", 0.05)),
        test_frac=float(cfg.get("test_fraction", 0.05)),
        max_songs=args.max_songs,
    )
    index_path = out / "spdmx_multitrack.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary = {
        "n_songs": len(records),
        "by_split": pd.Series([r["split"] for r in records]).value_counts().to_dict(),
        "future_visibility": cfg.get("future_visibility", 0),
        "train_steps": cfg.get("train_steps", 100000),
        "upstream": cfg.get("upstream_repo"),
        "preprocess_notes": [
            "Clone lukewys/stream-music-gen",
            "Add dataset adapter reading this JSONL (mirror slakh2100.py)",
            "Run extract_causal_dac_32k → extract_rms → dump_audio_mixdown",
            "Train dec_online with future_visibility=0 under matched steps",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index_path} ({len(records)} songs)")

    if args.hour_matched or cfg.get("hour_matched_hours"):
        df = pd.DataFrame(
            [
                {
                    "split": r["split"],
                    "song_id": r["song_id"],
                    "duration_sec": r["duration_sec"],
                    "corpus": "spdmx",
                    "mix_path": "",
                    "midi_path": "",
                    "n_stems": 2,
                }
                for r in records
            ]
        )
        sub = hour_matched_subsample(
            df,
            target_hours=float(cfg.get("hour_matched_hours", 145.0)),
            seed=int(cfg.get("seed", 43)),
        )
        keep = set(sub["song_id"].astype(str))
        matched = [r for r in records if r["song_id"] in keep]
        matched_path = out / "spdmx_hour_matched.jsonl"
        with matched_path.open("w", encoding="utf-8") as f:
            for rec in matched:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {matched_path} ({len(matched)} songs)")

    (out / "ADAPTER.md").write_text(
        "\n".join(
            [
                "# SPDMX adapter for stream-music-gen",
                "",
                "1. `git clone https://github.com/lukewys/stream-music-gen`",
                "2. Copy or symlink `spdmx_multitrack.jsonl` into the upstream data root.",
                "3. Implement `stream_music_gen/dataset/spdmx.py` mirroring `slakh2100.py`:",
                "   - group by `song_id`",
                "   - load stem FLACs from `stems[].audio_path`",
                "   - map `program` / `is_drum` → upstream instrument class ids",
                "4. Register `spdmx` in extract_causal_dac_32k / extract_rms / dump_audio_mixdown.",
                "5. Train: `scripts/train_dec_online.py` with `future_visibility: 0`.",
                "6. Eval: `scripts/gen_pred/gen_and_evaluate.py` on Slakh test.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
