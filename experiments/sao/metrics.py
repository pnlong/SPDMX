"""Compute FAD (OpenL3) and CLAP scores for SAO generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.sao.paths import ARMS, load_config, resolve_dev_dir


def _clap_scores(manifest: list[dict]) -> float:
    """Mean CLAP audio-text similarity; falls back to NaN if deps missing."""
    try:
        import laion_clap
        import torch
        import torchaudio
    except ImportError:
        print("laion-clap / torchaudio not installed; CLAP=NaN")
        return float("nan")

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    scores = []
    for row in manifest:
        path = row["path"]
        prompt = row["prompt"]
        wav, sr = torchaudio.load(path)
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        audio_emb = model.get_audio_embedding_from_data(x=wav, use_tensor=True)
        text_emb = model.get_text_embedding([prompt], use_tensor=True)
        sim = torch.nn.functional.cosine_similarity(audio_emb, text_emb).item()
        scores.append(sim)
    return float(np.mean(scores)) if scores else float("nan")


def _fad_score(gen_dir: Path, ref_dir: Path | None) -> float:
    """OpenL3 FAD via frechet_audio_distance if available."""
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError:
        print("frechet_audio_distance not installed; FAD=NaN")
        return float("nan")
    if ref_dir is None or not ref_dir.is_dir():
        print(f"reference dir missing ({ref_dir}); FAD=NaN")
        return float("nan")
    fad = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False,
    )
    # Some versions use openl3 — try attribute fallback.
    try:
        return float(fad.score(str(ref_dir), str(gen_dir)))
    except Exception as exc:  # noqa: BLE001
        print(f"FAD failed: {exc}")
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Directory of reference wav/flac for FAD (e.g. held-out mixes)",
    )
    parser.add_argument(
        "--write-paper",
        action="store_true",
        help=(
            "Write sao_metrics.csv under SPDMX_OUTPUT_DIR "
            "(dev/experiments/sao/metrics/); no repo-local copy."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    ref_dir = args.ref_dir
    rows = []
    for arm in ARMS:
        man_path = root / "generations" / arm / "manifest.json"
        if not man_path.is_file():
            print(f"skip {arm}: no manifest")
            continue
        with open(man_path) as f:
            manifest = json.load(f)
        gen_dir = root / "generations" / arm
        clap = _clap_scores(manifest)
        fad = _fad_score(gen_dir, ref_dir)
        rows.append({"train_arm": arm, "fad": fad, "clap": clap, "n": len(manifest)})

    if not rows:
        raise SystemExit("no generations to score")
    df = pd.DataFrame(rows)
    out = root / "metrics" / "sao_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
