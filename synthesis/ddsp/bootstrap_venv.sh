#!/usr/bin/env bash
# Bootstrap the isolated TF/DDSP venv for ablation B3 (Linux x86_64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Neural DDSP Track C is Linux x86_64 only (midi-ddsp pip)." >&2
  exit 1
fi

uv venv .venv-ddsp --python 3.10
PY=(uv pip install --python .venv-ddsp --no-build-isolation)

# 1) Build tooling + binary-friendly numeric stack (avoid ancient llvmlite builds).
"${PY[@]}" "setuptools>=65,<81" wheel "numpy<2" "llvmlite>=0.40" "numba>=0.57"

# 2) TensorFlow + shared audio deps.
# Runtime deps that midi-ddsp/ddsp need but are skipped by --no-deps below.
"${PY[@]}" "tensorflow>=2.11,<2.16" "tensorflow-probability==0.23.0" \
  soundfile gin-config "protobuf<=3.20" absl-py pretty_midi "librosa>=0.10" matplotlib \
  hmmlearn tensorflow-datasets music21 pandas \
  cloudml-hypertune google-cloud-storage dill future pydub "mir-eval" resampy

# 2b) CUDA 12 + cuDNN 8 pip wheels (TF 2.15). Host driver may be newer (e.g. 13.x);
# synthesis.ddsp.env prepends these to LD_LIBRARY_PATH for the worker.
"${PY[@]}" \
  "nvidia-cublas-cu12" "nvidia-cuda-cupti-cu12" "nvidia-cuda-nvrtc-cu12" \
  "nvidia-cuda-runtime-cu12" "nvidia-cudnn-cu12==8.9.7.29" \
  "nvidia-cufft-cu12" "nvidia-curand-cu12" "nvidia-cusolver-cu12" \
  "nvidia-cusparse-cu12" "nvidia-nccl-cu12" "nvidia-nvtx-cu12"

# 3) DDSP stack without letting pip downgrade numba/llvmlite via note-seq pins.
# midi-ddsp declares ddsp==3.2.0; we use ddsp==3.7.0 (needed by DDSP-Piano) via --no-deps.
"${PY[@]}" --no-deps "ddsp==3.7.0" "note-seq==0.0.3" crepe
"${PY[@]}" --no-deps midi-ddsp

# 4) DDSP-Piano checkout (uses ddsp already installed).
if [[ ! -d synthesis/ddsp/third_party/ddsp-piano/.git ]]; then
  git clone --depth 1 https://github.com/lrenault/ddsp-piano.git \
    synthesis/ddsp/third_party/ddsp-piano
fi

echo "DDSP venv ready: .venv-ddsp"
echo "GPU: CUDA-12 pip libs installed; worker uses GPU 0 by default (SPDMX_DDSP_FORCE_CPU=1 to disable)."
echo "Next: .venv-ddsp/bin/midi_ddsp_download_model_weights"
echo "Spot-listen: SPDMX_DDSP_PYTHON=\$PWD/.venv-ddsp/bin/python uv run python -m synthesis.ddsp.spot_listen_piano"
