"""Generate audio from fine-tuned SAO checkpoints for a fixed prompt set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from experiments.sao.paths import ARMS, load_config, resolve_dev_dir
from experiments.sao.prompts import select_prompts

_STEP_RE = re.compile(r"step=(\d+)")


def _ckpt_step(path: Path) -> int | None:
    m = _STEP_RE.search(path.stem)
    return int(m.group(1)) if m else None


def resolve_arm_ckpt(arm_dir: Path, *, prefer_steps: int | None = None) -> Path | None:
    """Pick a checkpoint: final/last, else exact step match, else highest step."""
    for name in ("final.ckpt", "last.ckpt"):
        p = arm_dir / name
        if p.is_file():
            return p
    stepped = [p for p in arm_dir.glob("epoch=*-step=*.ckpt") if p.is_file()]
    if stepped:
        if prefer_steps is not None:
            matched = [p for p in stepped if _ckpt_step(p) == prefer_steps]
            if matched:
                return matched[0]
        return max(stepped, key=lambda p: _ckpt_step(p) or -1)
    plain = sorted(p for p in arm_dir.glob("*.ckpt") if p.is_file())
    return plain[-1] if plain else None


def _is_lightning_ckpt(ckpt: Path) -> bool:
    """Lightning Trainer writes last.ckpt / epoch=*-step=*.ckpt."""
    return ckpt.name == "last.ckpt" or (
        ckpt.name.startswith("epoch=") and "step=" in ckpt.name
    )


def _load_unwrapped_state(ckpt: Path) -> dict:
    try:
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        state = load_ckpt_state_dict(str(ckpt))
    except Exception:
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) else blob
    if state and next(iter(state)).startswith("diffusion."):
        state = {
            k[len("diffusion.") :]: v
            for k, v in state.items()
            if k.startswith("diffusion.")
        }
    return state


def _load_sao_model(ckpt: Path, model_config: Path, device: torch.device):
    try:
        from stable_audio_tools.models.factory import create_model_from_config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install stable-audio-tools to generate: "
            'uv pip install -e "experiments/sao/stable-audio-tools[train]"'
        ) from exc

    with open(model_config) as f:
        model_cfg = json.load(f)
    model = create_model_from_config(model_cfg)

    # Filename heuristic avoids a second full read of ~17GB Lightning ckpts.
    if _is_lightning_ckpt(ckpt):
        try:
            from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Lightning checkpoints need pytorch-lightning: "
                'uv pip install -e "experiments/sao/stable-audio-tools[train]"'
            ) from exc
        training_config = model_cfg.get("training") or {}
        wrapper = DiffusionCondTrainingWrapper.load_from_checkpoint(
            str(ckpt),
            model=model,
            use_ema=training_config.get("use_ema", True),
            lr=training_config.get("learning_rate", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            strict=False,
            map_location="cpu",
        )
        if getattr(wrapper, "diffusion_ema", None) is not None:
            wrapper.diffusion.model = wrapper.diffusion_ema.ema_model
        model = wrapper.diffusion
    else:
        model.load_state_dict(_load_unwrapped_state(ckpt), strict=False)

    model.to(device).eval()
    return model, model_cfg


@torch.no_grad()
def generate_arm(
    arm: str,
    *,
    cfg: dict,
    ckpt: Path,
    device: torch.device,
) -> Path:
    try:
        from stable_audio_tools.inference.generation import generate_diffusion_cond
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-audio-tools inference unavailable") from exc

    root = resolve_dev_dir(cfg)
    out_dir = root / "generations" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    model_config = Path(__file__).resolve().parent / (
        cfg.get("model_config") or "model_config.json"
    )
    model, model_cfg = _load_sao_model(ckpt, model_config, device)
    sample_size = int(model_cfg.get("sample_size") or cfg.get("sample_size") or 2097152)
    sample_rate = int(model_cfg.get("sample_rate") or cfg.get("sample_rate") or 44100)
    seconds_total = sample_size / float(sample_rate)
    prompts = select_prompts(int(cfg.get("n_prompts", 50)))
    n_seeds = int(cfg.get("n_seeds_per_prompt", 2))
    steps = int(cfg.get("gen_steps", 100))
    cfg_scale = float(cfg.get("guidance_scale", 7.0))

    manifest = []
    for pi, prompt in enumerate(tqdm(prompts, desc=f"gen:{arm}")):
        for seed in range(n_seeds):
            out_path = out_dir / f"p{pi:03d}_s{seed}.wav"
            if out_path.is_file():
                manifest.append({"prompt": prompt, "seed": seed, "path": str(out_path)})
                continue
            torch.manual_seed(seed + 1000 * pi)
            audio = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=[
                    {
                        "prompt": prompt,
                        "seconds_start": 0,
                        "seconds_total": seconds_total,
                    }
                ],
                sample_size=sample_size,
                device=str(device),
            )
            # audio: (B, C, T)
            wav = audio[0].detach().cpu()
            torchaudio.save(str(out_path), wav, sample_rate)
            manifest.append({"prompt": prompt, "seed": seed, "path": str(out_path)})

    man_path = out_dir / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return man_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None, help="Single ckpt (applies to --arm)")
    parser.add_argument(
        "--prefer-steps",
        type=int,
        default=None,
        help="Prefer epoch=*-step=N.ckpt (default: config max_steps when set)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    device = torch.device(args.device)
    prefer_steps = args.prefer_steps
    if prefer_steps is None and cfg.get("max_steps") is not None:
        prefer_steps = int(cfg["max_steps"])
    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        if args.ckpt is not None:
            ckpt = Path(args.ckpt)
        else:
            ckpt = resolve_arm_ckpt(
                root / "checkpoints" / arm, prefer_steps=prefer_steps
            )
            if ckpt is None:
                print(f"no checkpoint for {arm}")
                continue
        print(f"{arm}: loading {ckpt}")
        path = generate_arm(arm, cfg=cfg, ckpt=Path(ckpt), device=device)
        print(f"{arm}: {path}")


if __name__ == "__main__":
    main()
