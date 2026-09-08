"""Train Hybrid Demucs (HTDemucs) on a frozen manifest arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.separation.dataset import StemPackDataset
from experiments.separation.paths import TARGETS, load_config, resolve_dev_dir


def build_model(sources: list[str], sample_rate: int):
    try:
        from demucs.htdemucs import HTDemucs
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "demucs is required for training. Install with: uv pip install demucs"
        ) from exc
    return HTDemucs(sources=sources, samplerate=sample_rate)


def train_arm(
    arm: str,
    *,
    cfg: dict,
    packs_root: Path,
    manifests_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
) -> Path:
    train_csv = manifests_dir / arm / "train.csv"
    val_csv = manifests_dir / arm / "val.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(train_csv)

    sample_rate = int(cfg.get("sample_rate", 44100))
    segment = float(cfg.get("segment_seconds", 10.0))
    channels = int(cfg.get("channels", 2))
    batch_size = int(cfg.get("batch_size", 4))
    max_steps = int(cfg.get("max_steps", 50_000))
    lr = float(cfg.get("lr", 3e-4))
    num_workers = int(cfg.get("num_workers", 4))
    sources = list(cfg.get("sources") or TARGETS)

    ds = StemPackDataset(
        train_csv,
        packs_root,
        sample_rate=sample_rate,
        segment_seconds=segment,
        channels=channels,
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    model = build_model(sources, sample_rate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    model.train()
    pbar = tqdm(total=max_steps, desc=f"train:{arm}")
    while step < max_steps:
        for batch in loader:
            mix = batch["mix"].to(device)
            sources_t = batch["sources"].to(device)
            # demucs forward: (B, C, T) → (B, S, C, T)
            estimate = model(mix)
            loss = torch.nn.functional.l1_loss(estimate, sources_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            pbar.update(1)
            if step % 1000 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "arm": arm,
                        "sources": sources,
                        "sample_rate": sample_rate,
                        "cfg": cfg,
                    },
                    ckpt_dir / f"step_{step:06d}.pt",
                )
            if step >= max_steps:
                break
    pbar.close()

    final_path = ckpt_dir / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "arm": arm,
            "sources": sources,
            "sample_rate": sample_rate,
            "cfg": cfg,
            "val_manifest": str(val_csv) if val_csv.is_file() else None,
        },
        final_path,
    )
    with open(ckpt_dir / "train_meta.json", "w") as f:
        json.dump({"arm": arm, "steps": step, "ckpt": str(final_path)}, f, indent=2)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("slakh", "spdmx_matched", "spdmx_full", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    packs_root = root / "packs"
    manifests = root / "manifests"
    device = torch.device(args.device)
    arms = ("slakh", "spdmx_matched", "spdmx_full") if args.arm == "all" else (args.arm,)
    for arm in arms:
        ckpt_dir = root / "checkpoints" / arm
        path = train_arm(
            arm,
            cfg=cfg,
            packs_root=packs_root,
            manifests_dir=manifests,
            ckpt_dir=ckpt_dir,
            device=device,
        )
        print(f"{arm}: wrote {path}")


if __name__ == "__main__":
    main()
