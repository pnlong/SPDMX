#!/usr/bin/env bash
# Multistem v2: backup v1 ckpts, reset-train, eval, regenerate panel (b).
# Usage:
#   CUDA_VISIBLE_DEVICES=1 experiments/separation/launch_multistem_v2.sh
# Prefer a 24GB GPU for batch_size 8; on 11GB cards lower batch in config first.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT_ROOT="${SPDMX_OUTPUT_DIR:-/deepfreeze/share/SPDMX}/dev/experiments/separation/checkpoints"
mkdir -p "$CKPT_ROOT"

if [[ -d "$CKPT_ROOT/multistem" && ! -d "$CKPT_ROOT/multistem_v1_baseline" ]]; then
  mv "$CKPT_ROOT/multistem" "$CKPT_ROOT/multistem_v1_baseline"
  echo "Backed up v1 → $CKPT_ROOT/multistem_v1_baseline"
fi

echo "Training multistem v2 (96k steps, weighted L1 + SI-SDR)…"
uv run python -m experiments.separation.train \
  --arm multistem \
  --config experiments/separation/config_multistem.yaml \
  --reset

echo "Evaluating…"
uv run python -m experiments.separation.eval_multistem --write-paper

echo "Regenerating separation figures…"
uv run python -m submission.make_figures --only separation

echo "Done. Check stem SI-SDR vs mixture_si_sdr in:"
echo "  $CKPT_ROOT/../eval/si_sdr_summary_multistem.csv"
