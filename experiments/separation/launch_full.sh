#!/usr/bin/env bash
# Full separation PoC on production SPDMX_OUTPUT_DIR / Slakh paths.
# Usage: experiments/separation/launch_full.sh [arm]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM="${1:-all}"
echo "Preparing packs (this can take many hours)…"
uv run python -m experiments.separation.prepare_stems --corpus both
uv run python -m experiments.separation.freeze_manifests
echo "Training arm=${ARM} (matched max_steps from config.yaml)…"
uv run python -m experiments.separation.train --arm "$ARM"
uv run python -m experiments.separation.eval --arm "$ARM" --write-paper
echo "Regenerate paper figure:"
uv run python -m submission.make_figures --only separation
