"""Train Hybrid Demucs (HTDemucs) on a frozen manifest arm.

Resumes automatically from ``last.ckpt`` when present (use ``--reset`` to
start fresh). Checkpoints store model + optimizer + step + best metrics.

Progress UI: one tqdm bar per train segment of ``val_every`` steps, then a
short val line, then a new bar for the next segment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
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


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        mix = batch["mix"].to(device)
        sources_t = batch["sources"].to(device)
        estimate = model(mix)
        loss = torch.nn.functional.l1_loss(estimate, sources_t)
        total += float(loss.detach().cpu())
        n += 1
    model.train()
    return total / max(n, 1)


class LossLogger:
    """JSONL + CSV loss history (truncate or append for resume)."""

    def __init__(self, ckpt_dir: Path, *, append: bool) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = ckpt_dir / "losses.jsonl"
        self.csv_path = ckpt_dir / "losses.csv"
        mode = "a" if append else "w"
        self._jsonl = open(self.jsonl_path, mode)
        new_csv = not append or not self.csv_path.is_file() or self.csv_path.stat().st_size == 0
        self._csv = open(self.csv_path, "a" if append and not new_csv else "w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=("step", "split", "loss"))
        if new_csv:
            self._writer.writeheader()
            self._csv.flush()

    def log(self, *, step: int, split: str, loss: float) -> None:
        row = {"step": int(step), "split": split, "loss": float(loss)}
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()
        self._writer.writerow(row)
        self._csv.flush()

    def close(self) -> None:
        self._jsonl.close()
        self._csv.close()


def _iter_batches(loader: DataLoader):
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


def train_arm(
    arm: str,
    *,
    cfg: dict,
    packs_root: Path,
    manifests_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
    resume: bool = True,
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
    log_every = int(cfg.get("log_every", 50))
    val_every = max(1, int(cfg.get("val_every", 1000)))
    raw_val_max = cfg.get("val_max_batches")
    val_max_batches = int(raw_val_max) if raw_val_max is not None else None

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

    val_loader: DataLoader | None = None
    if val_csv.is_file() and val_csv.stat().st_size > 0 and len(pd.read_csv(val_csv)) > 0:
        val_ds = StemPackDataset(
            val_csv,
            packs_root,
            sample_rate=sample_rate,
            segment_seconds=segment,
            channels=channels,
            train=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    model = build_model(sources, sample_rate).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.ckpt"
    best_val_path = ckpt_dir / "best_val.ckpt"
    best_train_path = ckpt_dir / "best_train.ckpt"

    best_val = float("inf")
    best_train = float("inf")
    step = 0
    resumed = False

    if not resume:
        for p in (last_path, best_val_path, best_train_path):
            if p.is_file():
                p.unlink()
        print(f"{arm}: --reset (cleared checkpoints)")
    elif last_path.is_file():
        blob = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if blob.get("optimizer") is not None:
            opt.load_state_dict(blob["optimizer"])
        step = int(blob.get("step") or 0)
        if blob.get("best_val") is not None:
            best_val = float(blob["best_val"])
        if blob.get("best_train") is not None:
            best_train = float(blob["best_train"])
        resumed = True
        print(f"{arm}: resume from {last_path} at step {step}")

    if step >= max_steps:
        print(f"{arm}: already at step {step} >= max_steps {max_steps}; skip")
        return last_path

    logger = LossLogger(ckpt_dir, append=resumed)
    batch_iter = _iter_batches(loader)

    def _save(path: Path, *, extra: dict | None = None) -> None:
        blob = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "arm": arm,
            "sources": sources,
            "sample_rate": sample_rate,
            "cfg": cfg,
            "best_val": best_val if best_val < float("inf") else None,
            "best_train": best_train if best_train < float("inf") else None,
            "val_manifest": str(val_csv) if val_loader is not None else None,
        }
        if extra:
            blob.update(extra)
        torch.save(blob, path)

    def _run_val(*, label: str) -> float | None:
        nonlocal best_val
        if val_loader is None:
            print(f"  [{label}] step {step}/{max_steps}  (no val set)")
            return None
        val_loss = run_validation(
            model, val_loader, device, max_batches=val_max_batches,
        )
        logger.log(step=step, split="val", loss=val_loss)
        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            _save(best_val_path, extra={"val_loss": val_loss})
            improved = "  *best*"
        train_note = f"  best_train={best_train:.4f}" if best_train < float("inf") else ""
        print(
            f"  [{label}] step {step}/{max_steps}  "
            f"val_l1={val_loss:.4f}  best_val={best_val:.4f}{train_note}{improved}"
        )
        return val_loss

    model.train()
    seg_idx = step // val_every
    try:
        while step < max_steps:
            seg_start = step
            seg_end = min(seg_start + val_every, max_steps)
            seg_len = seg_end - seg_start
            seg_idx += 1
            n_segs = (max_steps + val_every - 1) // val_every
            running = 0.0
            running_n = 0
            last_loss = 0.0

            pbar = tqdm(
                total=seg_len,
                desc=f"{arm} seg {seg_idx}/{n_segs} [{seg_start}→{seg_end})",
                leave=True,
            )
            try:
                for _ in range(seg_len):
                    batch = next(batch_iter)
                    mix = batch["mix"].to(device)
                    sources_t = batch["sources"].to(device)
                    estimate = model(mix)
                    loss = torch.nn.functional.l1_loss(estimate, sources_t)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    step += 1
                    last_loss = float(loss.detach().cpu())
                    running += last_loss
                    running_n += 1
                    pbar.set_postfix(loss=f"{last_loss:.4f}")
                    pbar.update(1)

                    if step % log_every == 0:
                        avg = running / max(running_n, 1)
                        logger.log(step=step, split="train", loss=avg)
                        running = 0.0
                        running_n = 0
                        _save(last_path)
                        if avg < best_train:
                            best_train = avg
                            _save(best_train_path, extra={"train_loss": avg})
            finally:
                pbar.close()

            # Flush remaining train window into the log.
            if running_n > 0:
                avg = running / running_n
                logger.log(step=step, split="train", loss=avg)
                _save(last_path)
                if avg < best_train:
                    best_train = avg
                    _save(best_train_path, extra={"train_loss": avg})

            _run_val(label=f"val after seg {seg_idx}/{n_segs}")
    finally:
        logger.close()

    _save(last_path)

    with open(ckpt_dir / "train_meta.json", "w") as f:
        json.dump(
            {
                "arm": arm,
                "steps": step,
                "resumed": resumed,
                "last_ckpt": str(last_path),
                "best_val_ckpt": str(best_val_path) if best_val_path.is_file() else None,
                "best_train_ckpt": str(best_train_path) if best_train_path.is_file() else None,
                "best_val": best_val if best_val < float("inf") else None,
                "best_train": best_train if best_train < float("inf") else None,
                "losses_csv": str(ckpt_dir / "losses.csv"),
                "losses_jsonl": str(ckpt_dir / "losses.jsonl"),
            },
            f,
            indent=2,
        )
    return last_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("slakh", "spdmx", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore last.ckpt and start training from step 0",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    packs_root = root / "packs"
    manifests = root / "manifests"
    device = torch.device(args.device)
    arms = ("slakh", "spdmx") if args.arm == "all" else (args.arm,)
    for arm in arms:
        ckpt_dir = root / "checkpoints" / arm
        path = train_arm(
            arm,
            cfg=cfg,
            packs_root=packs_root,
            manifests_dir=manifests,
            ckpt_dir=ckpt_dir,
            device=device,
            resume=not args.reset,
        )
        print(f"{arm}: wrote {path}")


if __name__ == "__main__":
    main()
