"""Evaluate multistem stem/other HTDemucs with SI-SDR on the held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.separation.audio_io import load_mono
from experiments.separation.dataset_multistem import STEM_OTHER_SOURCES, parse_tracks
from experiments.separation.eval import load_checkpoint, separate_track
from experiments.separation.freeze_multistem import MULTISTEM_ARM
from experiments.separation.paths import SPDMX_ROOT, load_config, resolve_dev_dir
from experiments.separation.sisdr import si_sdr
from experiments.separation.spdmx_io import resolve_spdmx_mix, resolve_spdmx_stem


@torch.no_grad()
def eval_multistem_manifest(
    model: torch.nn.Module,
    manifest_csv: Path,
    spdmx_root: Path,
    *,
    sample_rate: int,
    device: torch.device,
    track_mode: str = "first",
) -> pd.DataFrame:
    """SI-SDR for stem/other on each song (fixed first track, or mean over tracks)."""
    rows = []
    df = pd.read_csv(manifest_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="eval:multistem"):
        song_id = str(row["song_id"])
        tracks = parse_tracks(row.get("tracks"))
        if len(tracks) < 2:
            continue
        mix_path = resolve_spdmx_mix(spdmx_root, row, song_id=song_id)
        if mix_path is None:
            continue
        mix, _ = load_mono(mix_path, sample_rate=sample_rate)
        eval_tracks = tracks if track_mode == "all" else tracks[:1]
        for track in eval_tracks:
            stem_path = resolve_spdmx_stem(
                spdmx_root, row, song_id=song_id, track=int(track),
            )
            if stem_path is None:
                continue
            stem, _ = load_mono(stem_path, sample_rate=sample_rate)
            n = min(mix.shape[0], stem.shape[0])
            if n <= 0:
                continue
            mix_n = mix[:n]
            stem_n = stem[:n]
            other_n = mix_n - stem_n
            est = separate_track(model, mix_n, sample_rate=sample_rate, device=device)
            for name, ref in (("stem", stem_n), ("other", other_n)):
                rows.append(
                    {
                        "test_set": "spdmx_multistem",
                        "song_id": song_id,
                        "track": int(track),
                        "target": name,
                        "si_sdr": si_sdr(est[name], ref),
                        "mixture_si_sdr": si_sdr(mix_n, ref),
                        "train_arm": MULTISTEM_ARM,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_multistem.yaml",
    )
    parser.add_argument("--ckpt", type=Path, default=None, help="Default: best_val then last")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--track-mode", choices=("first", "all"), default="first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-paper", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    spdmx_root = Path(cfg.get("spdmx_root") or SPDMX_ROOT)
    manifest = root / "manifests" / MULTISTEM_ARM / f"{args.split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    ckpt_dir = root / "checkpoints" / MULTISTEM_ARM
    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        best = ckpt_dir / "best_val.ckpt"
        last = ckpt_dir / "last.ckpt"
        ckpt = best if best.is_file() else last
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    device = torch.device(args.device)
    model, sr, _arm = load_checkpoint(ckpt, device)
    # Ensure source names match stem/other even if older ckpt omitted them.
    if list(getattr(model, "sources", [])) != list(STEM_OTHER_SOURCES):
        model.sources = list(STEM_OTHER_SOURCES)  # type: ignore[attr-defined]

    per_track = eval_multistem_manifest(
        model,
        manifest,
        spdmx_root,
        sample_rate=sr,
        device=device,
        track_mode=args.track_mode,
    )
    out_dir = root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "si_sdr_per_track_multistem.csv"
    per_track.to_csv(per_path, index=False)
    summary = (
        per_track.groupby(["train_arm", "test_set", "target"], as_index=False)["si_sdr"]
        .mean()
        .rename(columns={"si_sdr": "si_sdr_mean"})
    )
    sum_path = out_dir / "si_sdr_summary_multistem.csv"
    summary.to_csv(sum_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {per_path}")
    print(f"wrote {sum_path}")
    if args.write_paper:
        paper = out_dir / "separation_sisdr_multistem.csv"
        summary.to_csv(paper, index=False)
        print(f"wrote {paper}")


if __name__ == "__main__":
    main()
