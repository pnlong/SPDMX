#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT="${1:?usage: run_all.sh /path/to/sao.ckpt}"
uv run python -m experiments.sao.prepare_dataset
uv run python -m experiments.sao.train --arm all --pretrained-ckpt "$CKPT"
uv run python -m experiments.sao.generate --arm all
uv run python -m experiments.sao.metrics --write-paper
