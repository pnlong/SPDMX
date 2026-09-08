"""Generate audio from fine-tuned SAO checkpoints for a fixed prompt set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.sao.paths import ARMS, load_config, resolve_dev_dir
from experiments.sao.prompts import select_prompts


def _load_sao_model(ckpt: Path, model_config: Path, device: torch.device):
    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install stable-audio-tools to generate: uv pip install stable-audio-tools"
        ) from exc

    with open(model_config) as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    state = load_ckpt_state_dict(str(ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg


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
    model_config = Path(__file__).resolve().parent / (cfg.get("model_config") or "model_config.json")
    model, model_cfg = _load_sao_model(ckpt, model_config, device)
    sample_size = int(model_cfg.get("sample_size") or cfg.get("sample_size") or 2097152)
    sample_rate = int(model_cfg.get("sample_rate") or cfg.get("sample_rate") or 44100)
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
                conditioning=[{"prompt": prompt}],
                sample_size=sample_size,
                device=str(device),
            )
            # audio: (B, C, T)
            wav = audio[0].detach().cpu()
            import torchaudio

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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_dev_dir(cfg)
    device = torch.device(args.device)
    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        ckpt = args.ckpt or (root / "checkpoints" / arm / "final.ckpt")
        # Lightning may write last.ckpt
        if not Path(ckpt).is_file():
            candidates = sorted((root / "checkpoints" / arm).glob("**/*.ckpt"))
            if not candidates:
                print(f"no checkpoint for {arm}")
                continue
            ckpt = candidates[-1]
        path = generate_arm(arm, cfg=cfg, ckpt=Path(ckpt), device=device)
        print(f"{arm}: {path}")


if __name__ == "__main__":
    main()
