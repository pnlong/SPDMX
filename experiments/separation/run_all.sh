#!/usr/bin/env bash
# End-to-end separation PoC: prepare → manifests → train → eval.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM="${1:-all}"
uv run python -m experiments.separation.prepare_stems --corpus both
uv run python -m experiments.separation.freeze_manifests
uv run python -m experiments.separation.train --arm "$ARM"
uv run python -m experiments.separation.eval --arm "$ARM"
