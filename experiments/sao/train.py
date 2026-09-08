"""Launch SAO fine-tunes via stable-audio-tools (three arms, matched steps)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.sao.paths import ARMS, SAO_DIR, load_config, resolve_dev_dir


def _find_train_py() -> Path | None:
    """Locate stable-audio-tools train.py (venv site-packages or sibling clone)."""
    which = shutil.which("stable-audio-tools-train")
    if which:
        return Path(which)
    # Common clone location
    for candidate in (
        SAO_DIR / "stable-audio-tools" / "train.py",
        SAO_DIR.parents[1] / "stable-audio-tools" / "train.py",
        Path.home() / "stable-audio-tools" / "train.py",
    ):
        if candidate.is_file():
            return candidate
    try:
        import stable_audio_tools

        pkg = Path(stable_audio_tools.__file__).resolve().parent
        # train.py often lives at repo root, not inside the package
        repo_train = pkg.parent / "train.py"
        if repo_train.is_file():
            return repo_train
    except ImportError:
        pass
    return None


def train_arm(
    arm: str,
    *,
    cfg: dict,
    pretrained_ckpt: Path,
    name: str | None = None,
) -> None:
    root = resolve_dev_dir(cfg)
    ds_cfg = root / "datasets" / arm / "dataset_config.json"
    if not ds_cfg.is_file():
        raise FileNotFoundError(f"missing {ds_cfg}; run prepare_dataset first")
    model_cfg = SAO_DIR / (cfg.get("model_config") or "model_config.json")
    train_py = _find_train_py()
    if train_py is None:
        raise SystemExit(
            "stable-audio-tools train.py not found. Clone "
            "https://github.com/Stability-AI/stable-audio-tools into "
            f"{SAO_DIR / 'stable-audio-tools'} or `uv pip install stable-audio-tools`."
        )

    save_dir = root / "checkpoints" / arm
    save_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg.get("max_steps", 10_000))
    batch_size = int(cfg.get("batch_size", 2))
    run_name = name or f"sao_{arm}"

    cmd = [
        sys.executable,
        str(train_py),
        "--model-config",
        str(model_cfg),
        "--dataset-config",
        str(ds_cfg),
        "--pretrained-ckpt-path",
        str(pretrained_ckpt),
        "--name",
        run_name,
        "--save-dir",
        str(save_dir),
        "--batch-size",
        str(batch_size),
        "--checkpoint-every",
        "2000",
        "--num-workers",
        "4",
    ]
    # stable-audio-tools uses Lightning; pass max_steps via env file sidecar.
    meta = {
        "arm": arm,
        "max_steps": max_steps,
        "cmd": cmd,
        "pretrained_ckpt": str(pretrained_ckpt),
    }
    with open(save_dir / "launch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Launching:", " ".join(cmd))
    print(f"Stop at ~{max_steps} optimizer steps (monitor Lightning progress).")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pretrained-ckpt", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt = args.pretrained_ckpt or cfg.get("pretrained_ckpt")
    if not ckpt:
        raise SystemExit(
            "Pass --pretrained-ckpt /path/to/sao.ckpt "
            "(unwrapped Stable Audio Open 1.0 checkpoint)."
        )
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        train_arm(arm, cfg=cfg, pretrained_ckpt=ckpt)


if __name__ == "__main__":
    main()
