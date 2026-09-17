#!/usr/bin/env bash
# Prepare StreamGen training data (Slakh and/or SPDMX) in one shot.
#
# Steps:
#   0. Wire DAC weights, Slakh symlink, SPDMX adapter + JSONL index
#   1. extract_causal_dac_32k  (GPU; fixed windows)
#   2. extract_rms
#   3. dump_audio_mixdown      (train / valid / test)
#
# Usage (from anywhere)::
#
#   # Slakh baseline only (default)
#   bash experiments/streamgen/prepare_streamgen_data.sh
#
#   # SPDMX arm (rebuilds full multitrack index first)
#   bash experiments/streamgen/prepare_streamgen_data.sh --datasets spdmx
#
#   # Both
#   bash experiments/streamgen/prepare_streamgen_data.sh --datasets slakh2100,spdmx
#
#   # Smoke: tiny SPDMX index + skip mixdown
#   bash experiments/streamgen/prepare_streamgen_data.sh --datasets spdmx --max-songs 50 --skip-dump
#
#   # Tune DAC GPU packing (fixed windows packed across stems; default batch 8)
#   bash experiments/streamgen/prepare_streamgen_data.sh --gpu 1 --win-duration 5 --chunk-batch-size 8
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="${REPO_ROOT}/experiments/streamgen/stream-music-gen"
ADAPTER_SRC="${REPO_ROOT}/experiments/streamgen/adapters/spdmx.py"
PAPER_INDEX="${REPO_ROOT}/analysis/paper_data/streamgen_spdmx_index"
DATA_ROOT="${STREAMGEN_DATA_ROOT:-/deepfreeze/share/SPDMX/dev/experiments/streamgen/stream_music_gen_data}"
SLAKH_ROOT="${SPDMX_SLAKH_ROOT:-/deepfreeze/share/pnlong/slakh2100_flac_redux}"
SPDMX_AUDIO="${SPDMX_DATASET_ROOT:-/deepfreeze/share/SPDMX/SPDMX}"
DAC_SRC="${DAC_WEIGHTS:-/home/pnlong/jazz-standard-dataset/experiments/live/stream-music-gen/pretrained_models/250121_stemmix_dac_weights_400k_steps.pth}"

DATASETS="slakh2100"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
WIN_DURATION=5
CHUNK_BATCH=8
NUM_WORKERS=8
MAX_SONGS=""
SKIP_DAC=0
SKIP_RMS=0
SKIP_DUMP=0
REBUILD_INDEX=0
TRAIN_MAX=1000000
VALID_MAX=10000
TEST_MAX=10000

usage() {
  sed -n '2,25p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --win-duration) WIN_DURATION="$2"; shift 2 ;;
    --chunk-batch-size) CHUNK_BATCH="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-songs) MAX_SONGS="$2"; shift 2 ;;
    --rebuild-index) REBUILD_INDEX=1; shift ;;
    --skip-dac) SKIP_DAC=1; shift ;;
    --skip-rms) SKIP_RMS=1; shift ;;
    --skip-dump) SKIP_DUMP=1; shift ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
cd "$REPO_ROOT"

run_uv() {
  (cd "$UPSTREAM" && CUDA_VISIBLE_DEVICES="$GPU" \
    uv run --project "$REPO_ROOT" "$@")
}

echo "=== StreamGen data prep ==="
echo "  repo:     $REPO_ROOT"
echo "  upstream: $UPSTREAM"
echo "  data:     $DATA_ROOT"
echo "  datasets: ${DATASET_ARR[*]}"
echo "  GPU:      $GPU"
echo "  DAC win:  ${WIN_DURATION}s × batch ${CHUNK_BATCH}"

[[ -d "$UPSTREAM" ]] || {
  echo "Missing upstream clone. Run: uv run python -m experiments.streamgen.setup_streamgen" >&2
  exit 1
}

# --- 0a. DAC weights ---
mkdir -p "$UPSTREAM/pretrained_models"
if [[ ! -e "$UPSTREAM/pretrained_models/250121_stemmix_dac_weights_400k_steps.pth" ]]; then
  [[ -f "$DAC_SRC" ]] || { echo "DAC weights not found: $DAC_SRC" >&2; exit 1; }
  ln -sfn "$DAC_SRC" "$UPSTREAM/pretrained_models/250121_stemmix_dac_weights_400k_steps.pth"
  echo "linked DAC weights"
fi

# --- 0b. data root + symlink into upstream ---
mkdir -p "$DATA_ROOT"
ln -sfn "$DATA_ROOT" "$UPSTREAM/stream_music_gen_data"

# --- 0c. Slakh layout (reuse deepfreeze 44.1 kHz FLAC; SAMPLE_RATE patched in clone) ---
if printf '%s\n' "${DATASET_ARR[@]}" | grep -qx 'slakh2100'; then
  mkdir -p "$DATA_ROOT/slakh2100/original"
  ln -sfn "$SLAKH_ROOT" "$DATA_ROOT/slakh2100/original/slakh2100_redux_16k"
  echo "Slakh → $SLAKH_ROOT"
fi

# --- 0d. SPDMX adapter + index ---
if printf '%s\n' "${DATASET_ARR[@]}" | grep -qx 'spdmx'; then
  cp "$ADAPTER_SRC" "$UPSTREAM/stream_music_gen/dataset/spdmx.py"

  # Ensure registrations (idempotent grep/insert is heavy; adapter copy is enough if
  # setup already patched constants — re-assert spdmx in DATASET_SPLITS via python).
  uv run --project "$REPO_ROOT" python - <<'PY'
from pathlib import Path
root = Path("experiments/streamgen/stream-music-gen/stream_music_gen")
# constants
c = root / "constants.py"
text = c.read_text()
if '"spdmx"' not in text:
    needle = '    "slakh2100": {\n        "train": "train",\n        "valid": "validation",\n        "test": "test",\n    },'
    insert = needle + '\n    "spdmx": {\n        "train": "train",\n        "valid": "validation",\n        "test": "test",\n    },'
    if needle not in text:
        raise SystemExit("could not patch constants.py for spdmx")
    c.write_text(text.replace(needle, insert, 1))
    print("patched constants.py")
for script in ("dataset/extract_causal_dac_32k.py", "dataset/extract_rms.py"):
    p = root / script
    t = p.read_text()
    if "from stream_music_gen.dataset.spdmx import Spdmx" not in t:
        t = t.replace(
            "from stream_music_gen.dataset.slakh2100 import Slakh2100\n",
            "from stream_music_gen.dataset.slakh2100 import Slakh2100\n"
            "from stream_music_gen.dataset.spdmx import Spdmx\n",
        )
        t = t.replace(
            '    "slakh2100": {\n        "class": Slakh2100,\n    },\n}',
            '    "slakh2100": {\n        "class": Slakh2100,\n    },\n'
            '    "spdmx": {\n        "class": Spdmx,\n    },\n}',
        )
        p.write_text(t)
        print(f"patched {script}")
dump = root / "dataset/dump_audio_mixdown.py"
dt = dump.read_text()
if '"spdmx"' not in dt:
    dump.write_text(
        dt.replace(
            'choices=["cocochorales", "moisesdb", "musdb", "slakh2100"]',
            'choices=["cocochorales", "moisesdb", "musdb", "slakh2100", "spdmx"]',
        )
    )
    print("patched dump_audio_mixdown.py")
PY

  INDEX_OUT="$PAPER_INDEX/spdmx_multitrack.jsonl"
  if [[ "$REBUILD_INDEX" -eq 1 || ! -f "$INDEX_OUT" || -n "$MAX_SONGS" ]]; then
    echo "building SPDMX JSONL index…"
    EXTRA=()
    [[ -n "$MAX_SONGS" ]] && EXTRA+=(--max-songs "$MAX_SONGS")
    (cd "$REPO_ROOT" && uv run python -m experiments.streamgen.prepare_spdmx_index "${EXTRA[@]}")
  fi
  mkdir -p "$DATA_ROOT/spdmx"
  ln -sfn "$INDEX_OUT" "$DATA_ROOT/spdmx/spdmx_multitrack.jsonl"
  ln -sfn "$SPDMX_AUDIO" "$DATA_ROOT/spdmx/audio"
  # drop stale parquet if index changed
  rm -f "$DATA_ROOT/spdmx"/pt_dataset_metadata_*.parquet
  echo "SPDMX index → $INDEX_OUT"
  echo "SPDMX audio → $SPDMX_AUDIO"
fi

cd "$UPSTREAM"

# --- 1. DAC ---
if [[ "$SKIP_DAC" -eq 0 ]]; then
  echo "=== 1/3  DAC extract ==="
  run_uv python stream_music_gen/dataset/extract_causal_dac_32k.py \
    --datasets "${DATASET_ARR[@]}" \
    --dataset_root stream_music_gen_data \
    --output_dir stream_music_gen_data/causal_dac_codes_32khz \
    --audio_dir stream_music_gen_data \
    --num_workers "$NUM_WORKERS" \
    --win-duration "$WIN_DURATION" \
    --chunk-batch-size "$CHUNK_BATCH"
else
  echo "=== 1/3  DAC extract (skipped) ==="
fi

# --- 2. RMS ---
if [[ "$SKIP_RMS" -eq 0 ]]; then
  echo "=== 2/3  RMS extract ==="
  run_uv python stream_music_gen/dataset/extract_rms.py \
    --datasets "${DATASET_ARR[@]}" \
    --dataset_root stream_music_gen_data \
    --output_dir stream_music_gen_data/rms_50hz \
    --num_workers "$NUM_WORKERS"
else
  echo "=== 2/3  RMS extract (skipped) ==="
fi

# --- 3. Mixdown dumps ---
if [[ "$SKIP_DUMP" -eq 0 ]]; then
  echo "=== 3/3  mixdown dump ==="
  for ds in "${DATASET_ARR[@]}"; do
    for split in train valid test; do
      case "$split" in
        train) max="$TRAIN_MAX"; extra=() ;;
        *) max="$VALID_MAX"; extra=(--save_audio) ;;
      esac
      echo "--- dump $ds $split (max=$max) ---"
      run_uv python stream_music_gen/dataset/dump_audio_mixdown.py \
        --dataset "$ds" \
        --split "$split" \
        --max_examples "$max" \
        --output_dir stream_music_gen_data/precompute_audio_mixdown_20s \
        --audio_duration 20 \
        --data_base_dir stream_music_gen_data/causal_dac_codes_32khz \
        --rms_base_dir stream_music_gen_data/rms_50hz \
        --audio_base_dir stream_music_gen_data/ \
        --num_workers "$NUM_WORKERS" \
        "${extra[@]}"
    done
  done
else
  echo "=== 3/3  mixdown dump (skipped) ==="
fi

echo
echo "Done. Data under: $DATA_ROOT"
echo "Train example:"
echo "  cd $UPSTREAM"
echo "  CUDA_VISIBLE_DEVICES=$GPU uv run --project $REPO_ROOT python scripts/train_dec_online.py \\"
echo "    --args.load configs/online/decoder_online_future_visibility_0.yml \\"
echo "    --save_dir /deepfreeze/share/SPDMX/dev/experiments/streamgen/logs/dec_online_fv0 \\"
echo "    --batch_size 8 --precision 16-mixed"
