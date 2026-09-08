#!/usr/bin/env bash
# Chain ICASSP PoC steps after packs exist. Requires a GPU host for train/eval.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="${SPDMX_OUTPUT_DIR:-/deepfreeze/share/SPDMX}"
SEP="$ROOT/dev/experiments/separation"

echo "== prepare stems (skip if packs already built) =="
uv run python -m experiments.separation.prepare_stems \
  --corpus both \
  --spdmx-root "${SPDMX_DATASET_ROOT:-$ROOT/SPDMX}"

echo "== freeze manifests =="
uv run python -m experiments.separation.freeze_manifests

echo "== train Demucs (GPU) =="
uv run python -m experiments.separation.train --arm all

echo "== eval → submission/data/separation_sisdr.csv =="
uv run python -m experiments.separation.eval --arm all --write-paper

echo "== SAO (optional; needs SAO ckpt) =="
if [[ -n "${SAO_PRETRAINED_CKPT:-}" ]]; then
  uv run python -m experiments.sao.prepare_dataset
  uv run python -m experiments.sao.train --arm all --pretrained-ckpt "$SAO_PRETRAINED_CKPT"
  uv run python -m experiments.sao.generate --arm all
  uv run python -m experiments.sao.metrics --write-paper \
    --ref-dir "${SAO_REF_DIR:-$SEP/../sao/ref_mixes}"
else
  echo "skip SAO: set SAO_PRETRAINED_CKPT to enable"
fi

echo "== regenerate paper figures =="
uv run python -m submission.make_figures --only downstream

echo "Done. Recompile submission/main.tex"
