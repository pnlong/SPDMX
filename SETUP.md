# spdmx setup

Step-by-step guide to set up spdmx on a new machine. Everything Python-related lives in **`~/spdmx/.venv`** via [uv](https://docs.astral.sh/uv/). Python **3.10** is required (see `.python-version`).

---

## What you get

| Track | Capabilities | Needs |
|-------|----------------|-------|
| **A — Synthesis + analysis** | MIDI → FLAC stems, song-length analysis, tests | uv, fluidsynth, soundfont |
| **B — + Realify (SA3)** | Audio-to-audio realification | Everything in A + GPU, SA3 submodule, flash-attn, Hugging Face login |

Do **Track A** first. Add **Track B** when you need `--realify`.

---

## 0. Prerequisites

### Clone the repo

```bash
git clone <your-spdmx-repo-url> ~/spdmx
cd ~/spdmx
cp .env.example .env
# edit .env for your PDMX / output / soundfont paths
```

### Install uv (user-local, not system Python)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Add that `export` to your shell rc file (`~/.bashrc`, etc.).

### Install Python 3.10 via uv

```bash
uv python install 3.10
```

uv stores this in its own cache; it does not replace system Python.

### fluidsynth (required for synthesis)

spdmx calls the **`fluidsynth` executable** on your PATH. uv cannot install this inside `.venv`.

**Option 1 — system package (simplest):**

```bash
sudo apt install fluidsynth
fluidsynth --version
```

**Option 2 — project-local (no sudo):** if you use mamba/conda:

```bash
cd ~/spdmx
mamba create -p .tools/fluidsynth -c conda-forge fluidsynth -y
export PATH="$PWD/.tools/fluidsynth/bin:$PATH"
fluidsynth --version
```

Add the `export PATH=...` line to your shell when working on spdmx.

### Data paths (edit for your machine)

Copy the example env file and set paths for your storage layout:

```bash
cp .env.example .env
# edit .env — SPDMX_PDMX_FILEPATH, SPDMX_OUTPUT_DIR, SPDMX_SOUNDFONT_DIR
```

| Variable | Purpose |
|----------|---------|
| `SPDMX_PDMX_FILEPATH` | PDMX metadata CSV |
| `SPDMX_OUTPUT_DIR` | All spdmx outputs |
| `SPDMX_SOUNDFONT_DIR` | Soundfont library directory |
| `SPDMX_SOUNDFONT_PATH` | (optional) Override default `SGM-V2.01.sf2` path |

Python code still imports `PDMX_FILEPATH`, `OUTPUT_DIR`, etc. from [`shared/config.py`](shared/config.py); those names read from `.env` at import time. Non-path constants stay in `config.py`.

### GPU (Track B only)

- NVIDIA driver + CUDA (driver must support CUDA 12.6 for our default PyTorch wheels)
- ~6+ GB VRAM recommended for SA3 **medium**

---

## Track A — Synthesis and analysis

### Step A1. Create the project venv

From repo root:

```bash
cd ~/spdmx
uv sync --group dev
```

This creates `.venv/` and installs spdmx, torch, analysis deps, pytest, etc.

### Step A2. Symlink dev outputs into the repo

After editing paths in `shared/config.py` if needed:

```bash
uv run python -m shared.setup_symlinks
```

Creates gitignored symlinks:

- `analysis/output/` → `{OUTPUT_DIR}/dev/analysis/`
- `synthesis/ablations_output/` → `{OUTPUT_DIR}/dev/ablations/`

Re-run when you change `OUTPUT_DIR` or set up a fresh clone. Individual CLIs (`synthesize`, `analyze_song_lengths`) also refresh their symlinks.

### Step A3. Verify Track A

```bash
uv run python --version          # Python 3.10.x
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest -q
```

Optional — song-length analysis (reads PDMX CSV, no synthesis):

```bash
uv run python -m analysis.analyze_song_lengths
```

Outputs go to `{OUTPUT_DIR}/dev/analysis/song_lengths/` and symlink to `analysis/output/` in the repo (gitignored).

### Step A4. Run synthesis (smoke test)

```bash
uv run python -m synthesis.synthesize --render-mode basic
```

Default: 100-song ablation sample → `{OUTPUT_DIR}/dev/ablations/basic/`. Browsable in-repo via gitignored symlink `synthesis/ablations_output/` (from Step A2 or `synthesize`).

---

## Track B — Stable Audio 3 realify

Complete Track A first.

### Step B1. Clone the SA3 submodule

```bash
cd ~/spdmx
git submodule update --init --depth 1 synthesis/realify/stable-audio-3
```

Confirm upstream code is present:

```bash
test -f synthesis/realify/stable-audio-3/pyproject.toml && echo "SA3 submodule ok"
```

#### Submodule troubleshooting

If `git submodule update --init` fails with *pathspec did not match*:

The submodule was never registered correctly. Fix:

```bash
cd ~/spdmx

# Remove any wrongly-tracked stub files
git rm -f synthesis/realify/stable-audio-3/README.md 2>/dev/null || true

# If .gitmodules has a stale entry but no gitlink:
git config -f .gitmodules --remove-section submodule.synthesis/realify/stable-audio-3 2>/dev/null || true
git add .gitmodules

# Add the real submodule
git submodule add https://github.com/Stability-AI/stable-audio-3.git synthesis/realify/stable-audio-3
```

If `git submodule add` says *already exists in the index*, run `git rm` on any files under that path first, then retry.

After cloning on a fresh machine with a correct repo:

```bash
git clone --recurse-submodules <repo-url> ~/spdmx
```

### Step B2. Install SA3 into the spdmx venv

Still from repo root — one venv for everything:

```bash
uv pip install -e synthesis/realify/stable-audio-3
```

This installs SA3 and its Python dependencies into `.venv`. You do **not** need to run `uv sync` inside the submodule for normal spdmx realify.

Verify:

```bash
uv run python -c "from stable_audio_3 import StableAudioModel; print('SA3 import ok')"
```

**CUDA note:** On Linux x86_64, `pyproject.toml` installs `torch==2.7.1` from the PyTorch **cu126** index. For other CUDA versions see [SA3 CUDA docs](https://github.com/Stability-AI/stable-audio-3/blob/main/README.md#cuda-version).

Check your versions:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Step B3. Install flash-attention (required for SA3 medium)

SA3 **medium** (our default from song-length analysis) needs flash-attn. It is **not** in `pyproject.toml` — install a prebuilt wheel matching your CUDA, torch, and Python.

**Example — CUDA 12.6, torch 2.7, Python 3.10** (matches default spdmx env):

```bash
uv pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
```

For other combinations, browse [flash-attention-prebuild-wheels releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) or see [SA3 flash-attention docs](https://github.com/Stability-AI/stable-audio-3/blob/main/README.md#flash-attention).

Verify:

```bash
uv run python -c "import flash_attn; print('flash-attn ok', flash_attn.__version__)"
```

**Important:** After installing flash-attn or SA3, if you run `uv sync` again, use:

```bash
uv sync --inexact --group dev
```

Otherwise uv may remove packages not listed in `pyproject.toml`.

### Step B4. Hugging Face login

Model weights download from Hugging Face on first realify run.

```bash
uv run hf auth login
```

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access). Verify:

```bash
uv run hf auth whoami
```

We use **[stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)** (see song-length analysis report).

### Step B5. Realify smoke test

Requires completed A1 stems (`Step A4`). Realify is GPU-only and does **not** re-synthesize:

```bash
uv run python -m synthesis.synthesize --render-mode basic --realify --realify-limit 2
```

If `basic/` is incomplete, this prints an error with the command to run first.

Or standalone (captions generated in memory):

```bash
uv run python -m synthesis.realify.realify \
  --source-dir /path/to/dev/ablations/basic \
  --output-dir /path/to/dev/ablations/basic_realify \
  --limit 2
```

Preview captions without running realify:

```bash
uv run python -m synthesis.realify.captions.generate \
  --dataset_dir /path/to/dev/ablations/basic
```

Preset notebook: `synthesis/realify/tests/explore_presets.ipynb`

---

## Track C — Neural DDSP (ablations CA/CB: `ddsp_basic` / `ddsp_slakh`)

MIDI-DDSP + DDSP-Piano run in a **separate TensorFlow venv** so they do not fight
PyTorch CUDA in `.venv`. **Linux x86_64 only** — `midi-ddsp` does not install on
Apple Silicon.

### Step C1. Create the TF/DDSP venv

```bash
cd ~/spdmx
bash synthesis/ddsp/bootstrap_venv.sh
# Or manually:
#   uv venv .venv-ddsp --python 3.10
#   uv pip install --python .venv-ddsp -r synthesis/ddsp/requirements-tf.txt
#   uv pip install --python .venv-ddsp --no-deps midi-ddsp
```

Install `midi-ddsp` with `--no-deps` after the requirements file so it does not pull
an ancient `note-seq`→`librosa`→`numba`→`llvmlite` chain that fails to build on
Python 3.10.

Optional override: `export SPDMX_DDSP_VENV=/path/to/venv` or
`export SPDMX_DDSP_PYTHON=/path/to/python`.

**GPU (default):** `bootstrap_venv.sh` installs CUDA 12 / cuDNN 8 pip wheels into
`.venv-ddsp`. The synthesizer starts a **persistent worker pool** (one long-lived
`worker serve` process per GPU; models stay loaded). Useful knobs:

| Variable | Meaning |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPU id(s); one serve worker each (default `0`; e.g. `0,1`) |
| `SPDMX_DDSP_FORCE_CPU=1` | Single CPU serve worker (`CUDA_VISIBLE_DEVICES=-1`) |
| `SPDMX_DDSP_ONESHOT=1` | Legacy per-stem subprocess + flock (debug) |
| `SPDMX_DDSP_CHUNK_SEC` | MIDI-DDSP + piano chunk length in seconds (default `12`) |
| `SPDMX_DDSP_CHUNK_OVERLAP_SEC` | Chunk crossfade overlap in seconds (default `2.0`) |

TF 2.15 needs CUDA **12.x** + cuDNN **8**; a host driver advertising CUDA 13 is fine
as long as the pip CUDA-12 libs are on `LD_LIBRARY_PATH` (handled automatically).

### Step C2. MIDI-DDSP weights

```bash
# Preferred (entrypoint from the midi-ddsp package):
.venv-ddsp/bin/midi_ddsp_download_model_weights

# Or manual zip → extract under {OUTPUT_DIR}/models/midi_ddsp (or set MIDI_DDSP_WEIGHTS_DIR):
# https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip
```

Default weight dir: `{OUTPUT_DIR}/models/midi_ddsp` (see `synthesis/ddsp/config.py`).

### Step C3. Clone DDSP-Piano

```bash
git clone https://github.com/lrenault/ddsp-piano.git synthesis/ddsp/third_party/ddsp-piano
# Bundled default checkpoint: ddsp_piano/model_weights/dafx22
# Gin: ddsp_piano/configs/dafx22.gin
```

Install any extra deps the upstream README requires into `.venv-ddsp` (gin-config is
already in `requirements-tf.txt`).

### Step C4. Spot-listen piano (quality gate)

```bash
SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python \
  uv run python -m synthesis.ddsp.spot_listen_piano --out /tmp/ddsp_piano_spot.wav
```

Listen before large B3 batches. Provenance notes: [`THIRD_PARTY.md`](THIRD_PARTY.md).

### Step C5. Run B3 / optional B4

```bash
# B3 — neural DDSP + slakh soundfont fallback (no SA3)
# Song-level -j stays 1; neural stems within a song fan out across the GPU pool.
SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python \
  CUDA_VISIBLE_DEVICES=0,1 \
  uv run python -m synthesis.synthesize --render-mode ddsp_slakh

# B4 — optional SA3 after B3 (requires completed B3 stems)
SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python \
  uv run python -m synthesis.synthesize --render-mode ddsp_slakh --realify
```

Coverage stats (program-only, then optional monophony pass):

```bash
uv run python -m analysis.ddsp_coverage --subset rated_deduplicated
uv run python -m analysis.ddsp_coverage --subset rated_deduplicated --check-monophony -n 200
```

---

## Quick reference — copy/paste (full setup)

```bash
# --- prerequisites ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10
sudo apt install fluidsynth   # or use project-local mamba — see above

# --- clone ---
git clone --recurse-submodules <repo-url> ~/spdmx
cd ~/spdmx

# --- Track A ---
uv sync --group dev
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"

# --- Track B ---
git submodule update --init --depth 1 synthesis/realify/stable-audio-3
uv pip install -e synthesis/realify/stable-audio-3
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
uv run hf auth login
uv run python -c "import flash_attn; from stable_audio_3 import StableAudioModel; print('realify ok')"
```

Edit `shared/config.py` paths before synthesis.

---

## Optional: SA3 standalone env

To run SA3's Gradio UI or upstream tests in isolation:

```bash
cd ~/spdmx/synthesis/realify/stable-audio-3
uv sync
uv run python run_gradio.py --model medium
```

This uses a **separate** `.venv` inside the submodule. For spdmx `--realify`, prefer the single root venv (Track B above).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv init` fails — project already initialized | Use `uv sync`, not `uv init` |
| `stable-audio-3 does not appear to be a Python project` on `uv sync` | Expected — base `uv sync` does not need SA3. Install SA3 separately (Step B2) |
| `git submodule add` — already exists in index | `git rm` stub files under `stable-audio-3/`, then re-add submodule |
| `git submodule update` — pathspec not known | Submodule gitlink missing; see Step B1 troubleshooting |
| `hf login` — no such command | Use `uv run hf auth login` |
| Realify outputs static noise | flash-attn not installed correctly — re-run Step B3 verify |
| `fluidsynth` not found | Install fluidsynth (Step 0); ensure it is on `PATH` |

---

## Day-to-day commands

Always run from repo root:

```bash
cd ~/spdmx
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m analysis.analyze_song_lengths
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest -q
```

No need to `source .venv/bin/activate` — `uv run` uses the project venv automatically.
