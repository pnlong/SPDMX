# Lite Stable Audio Open fine-tune PoC

Three arms (**Slakh**, **sPDMX-matched**, **sPDMX-full**) under a matched step budget.
Eval: OpenL3/VGGish **FAD** + **CLAP** (no listening).

## Setup

```bash
# Clone training code (recommended)
git clone https://github.com/Stability-AI/stable-audio-tools experiments/sao/stable-audio-tools
# [train] pulls pytorch_lightning and other training deps (not in the base package)
uv pip install -e "experiments/sao/stable-audio-tools[train]"
uv pip install laion-clap frechet_audio_distance  # metrics

# Download unwrapped SAO 1.0 checkpoint (HF: stabilityai/stable-audio-open-1.0)
# export or pass --pretrained-ckpt /path/to/model.ckpt
```

Depends on separation manifests for hour-matched / full song lists:

```bash
uv run python -m experiments.separation.prepare_stems
uv run python -m experiments.separation.freeze_manifests
```

## Pipeline

```bash
uv run python -m experiments.sao.prepare_dataset
uv run python -m experiments.sao.train --arm all --pretrained-ckpt /path/to/sao.ckpt
uv run python -m experiments.sao.generate --arm all
uv run python -m experiments.sao.metrics --ref-dir /path/to/heldout_mixes
```

Outputs: `{SPDMX_OUTPUT_DIR}/dev/experiments/sao/`. Paper CSV: `submission/data/sao_metrics.csv`.

See also [`CHECKPOINT.md`](CHECKPOINT.md) for downloading SAO 1.0 weights.
