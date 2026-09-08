"""Evaluate trained Hybrid Demucs checkpoints with SI-SDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.separation.audio_io import load_mono, mono_to_stereo
from experiments.separation.paths import (
    MUSDB_ROOT,
    TARGETS,
    load_config,
    resolve_dev_dir,
)
from experiments.separation.sisdr import si_sdr
from experiments.separation.train import build_model


@torch.no_grad()
def separate_track(
    model: torch.nn.Module,
    mix_mono: np.ndarray,
    *,
    sample_rate: int,
    device: torch.device,
    segment: int = 441000,
) -> dict[str, np.ndarray]:
    """Run the model over a mono mix in overlapping chunks; return mono stems."""
    model.eval()
    hop = segment
    n = mix_mono.shape[0]
    sources = list(getattr(model, "sources", TARGETS))
    acc = {t: np.zeros(n, dtype=np.float64) for t in sources}
    weight = np.zeros(n, dtype=np.float64)
    for start in range(0, max(n, 1), hop):
        end = min(start + segment, n)
        chunk = mix_mono[start:end]
        if chunk.shape[0] < segment:
            chunk = np.pad(chunk, (0, segment - chunk.shape[0]))
        mix_t = torch.from_numpy(mono_to_stereo(chunk)).unsqueeze(0).to(device)
        est = model(mix_t)[0].detach().cpu().numpy()  # (S, 2, T)
        for i, t in enumerate(sources):
            mono = est[i].mean(axis=0)[: end - start]
            acc[t][start:end] += mono
        weight[start:end] += 1.0
    weight = np.maximum(weight, 1e-8)
    return {t: (acc[t] / weight).astype(np.float32) for t in sources}


def eval_slakh_test(
    model: torch.nn.Module,
    test_csv: Path,
    packs_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    df = pd.read_csv(test_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="eval:slakh"):
        song_dir = packs_root / row["path"]
        refs = {}
        for t in TARGETS:
            refs[t], _ = load_mono(song_dir / f"{t}.flac", sample_rate=sample_rate)
        mix, _ = load_mono(song_dir / "mix.flac", sample_rate=sample_rate)
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        for t in TARGETS:
            rows.append(
                {
                    "test_set": "slakh2100",
                    "song_id": row["song_id"],
                    "target": t,
                    "si_sdr": si_sdr(est[t], refs[t]),
                    "mixture_si_sdr": si_sdr(mix, refs[t]),
                }
            )
    return pd.DataFrame(rows)


def eval_musdb_bass_drums(
    model: torch.nn.Module,
    musdb_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
) -> pd.DataFrame:
    """Cross-domain: only Bass/Drums (shared with MUSDB)."""
    test_dir = musdb_root / "test"
    if not test_dir.is_dir():
        # musdb18hq layout sometimes flat under test/
        alt = musdb_root / "musdb18hq" / "test"
        test_dir = alt if alt.is_dir() else test_dir
    if not test_dir.is_dir():
        print(f"MUSDB test not found at {musdb_root}; skipping")
        return pd.DataFrame()

    rows = []
    tracks = sorted(p for p in test_dir.iterdir() if p.is_dir())
    for track in tqdm(tracks, desc="eval:musdb"):
        bass_p = track / "bass.wav"
        drums_p = track / "drums.wav"
        mix_p = track / "mixture.wav"
        if not (bass_p.is_file() and drums_p.is_file() and mix_p.is_file()):
            continue
        bass, _ = load_mono(bass_p, sample_rate=sample_rate)
        drums, _ = load_mono(drums_p, sample_rate=sample_rate)
        mix, _ = load_mono(mix_p, sample_rate=sample_rate)
        est = separate_track(model, mix, sample_rate=sample_rate, device=device)
        for t, ref in (("bass", bass), ("drums", drums)):
            rows.append(
                {
                    "test_set": "musdb18",
                    "song_id": track.name,
                    "target": t,
                    "si_sdr": si_sdr(est[t], ref),
                    "mixture_si_sdr": si_sdr(mix, ref),
                }
            )
    return pd.DataFrame(rows)


def load_checkpoint(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sources = list(blob.get("sources") or TARGETS)
    sample_rate = int(blob.get("sample_rate") or 44100)
    model = build_model(sources, sample_rate)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.sources = sources  # type: ignore[attr-defined]
    return model, sample_rate, blob.get("arm", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("slakh", "spdmx_matched", "spdmx_full", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--musdb-root", type=Path, default=None)
    parser.add_argument(
        "--write-paper",
        action="store_true",
        help="Also write submission/data/separation_sisdr.csv for make_figures.py",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    packs_root = root / "packs"
    manifests = root / "manifests"
    out_dir = root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    musdb_root = args.musdb_root or Path(cfg.get("musdb_root") or MUSDB_ROOT)

    arms = ("slakh", "spdmx_matched", "spdmx_full") if args.arm == "all" else (args.arm,)
    all_rows: list[pd.DataFrame] = []
    for arm in arms:
        ckpt_dir = root / "checkpoints" / arm
        # Prefer best val, then best train, then last.
        for name in ("best_val.ckpt", "best_train.ckpt", "last.ckpt", "best.pt", "final.pt"):
            cand = ckpt_dir / name
            if cand.is_file():
                ckpt = cand
                break
        else:
            print(f"missing checkpoint for {arm} under {ckpt_dir}")
            continue
        print(f"{arm}: evaluating {ckpt.name}")
        model, sr, _ = load_checkpoint(ckpt, device)
        test_csv = manifests / arm / "test.csv"
        if test_csv.is_file():
            df = eval_slakh_test(model, test_csv, packs_root, sample_rate=sr, device=device)
            df["train_arm"] = arm
            all_rows.append(df)
        mus = eval_musdb_bass_drums(model, musdb_root, sample_rate=sr, device=device)
        if len(mus):
            mus["train_arm"] = arm
            all_rows.append(mus)

    if not all_rows:
        raise SystemExit("no eval results; train checkpoints first")
    full = pd.concat(all_rows, ignore_index=True)
    full_path = out_dir / "si_sdr_per_track.csv"
    full.to_csv(full_path, index=False)

    summary = (
        full.groupby(["train_arm", "test_set", "target"], as_index=False)["si_sdr"]
        .mean()
        .rename(columns={"si_sdr": "si_sdr_mean"})
    )
    summary_path = out_dir / "si_sdr_summary.csv"
    summary.to_csv(summary_path, index=False)
    paper_csv = Path(__file__).resolve().parents[2] / "submission" / "data" / "separation_sisdr.csv"
    if args.write_paper:
        paper_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(paper_csv, index=False)
    print(
        json.dumps(
            {
                "per_track": str(full_path),
                "summary": str(summary_path),
                "paper": str(paper_csv) if args.write_paper else None,
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
