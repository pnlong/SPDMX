"""Compute FAD (OpenL3) and CLAP scores for SAO generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.sao.paths import ARMS, load_config, resolve_dev_dir


def _clap_device() -> str:
    """Pick a CLAP device; fall back to CPU when the GPU arch is unsupported."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    major, _minor = torch.cuda.get_device_capability(0)
    # Current torch wheels often stop at sm_90; Blackwell is sm_120.
    if major >= 12:
        print(
            f"CLAP: CUDA capability sm_{major}xx unsupported by this PyTorch; using CPU"
        )
        return "cpu"
    return "cuda:0"


def _load_clap() -> Any | None:
    """Load LAION-CLAP; returns None if deps are missing."""
    try:
        import laion_clap
        import torch
    except ImportError:
        print("laion-clap / torch not installed; CLAP=NaN")
        return None

    # laion_clap calls torch.load without weights_only=; PyTorch 2.6+ defaults True.
    _orig_load = torch.load

    def _load_ckpt(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    model = laion_clap.CLAP_Module(enable_fusion=False, device=_clap_device())
    torch.load = _load_ckpt  # type: ignore[assignment]
    try:
        model.load_ckpt()
    finally:
        torch.load = _orig_load  # type: ignore[assignment]
    return model


def _clap_scores(manifest: list[dict], model: Any | None = None) -> float:
    """Mean CLAP audio-text similarity; falls back to NaN if deps missing."""
    try:
        import torch
        import torchaudio
    except ImportError:
        print("torchaudio not installed; CLAP=NaN")
        return float("nan")

    if model is None:
        model = _load_clap()
    if model is None:
        return float("nan")

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
    clap_model = _load_clap()
    rows = []
    for arm in ARMS:
        man_path = root / "generations" / arm / "manifest.json"
        if not man_path.is_file():
            print(f"skip {arm}: no manifest")
            continue
        with open(man_path) as f:
            manifest = json.load(f)
        gen_dir = root / "generations" / arm
        clap = _clap_scores(manifest, model=clap_model)
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
