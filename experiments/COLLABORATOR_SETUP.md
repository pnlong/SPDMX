# ICASSP pilots — collaborator machine setup

Copy everything below the line into Slack/email. Assumes **Deep Freeze is
mounted** and the shared trees already exist:

- `/deepfreeze/share/SPDMX` (SPDMX release + `dev/`)
- `/deepfreeze/share/pnlong/slakh2100_flac_redux`

Goal: get **YourMT3 (transcription)** and **stream-music-gen (StreamGen)**
cloned and importable with one command after clone + `uv sync`.

---

## Slack message (copy from here)

**SPDMX — set up transcription + StreamGen pilots**

Same Deep Freeze layout as synthesis. Clone the repo, sync the venv, run one
setup command. Read the “do not” list at the bottom.

**Prerequisites:** Deep Freeze mounted, NVIDIA GPU (`nvidia-smi`), Linux x86_64,
git, git-lfs (`sudo apt install git-lfs`).

---

**1. Clone and paths**

```bash
git clone git@github.com:pnlong/SPDMX.git ~/spdmx
cd ~/spdmx
cp .env.example .env
```

`.env.example` already points at the shared trees — only edit if your mount
paths differ:

```
SPDMX_OUTPUT_DIR="/deepfreeze/share/SPDMX"
SPDMX_SLAKH_ROOT="/deepfreeze/share/pnlong/slakh2100_flac_redux"
SPDMX_PDMX_FILEPATH="/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
```

---

**2. Install uv + Python 3.10 + project venv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# add that export to ~/.bashrc if needed

uv python install 3.10
cd ~/spdmx
uv sync --group dev
```

---

**3. One-command pilot setup**

```bash
cd ~/spdmx
uv run python -m experiments.setup_pilots
```

That will:

1. Keep / create `.env`
2. Verify SPDMX `songs.csv` / `stems.csv` and Slakh are visible
3. Create in-repo deepfreeze symlinks
4. Clone + install **YourMT3** → `experiments/transcription/YourMT3`
5. Clone + install **stream-music-gen** → `experiments/streamgen/stream-music-gen`
6. Write `experiments/PILOTS_READY.md` with next steps

Optional (large downloads — only when you are ready to train/dump):

```bash
uv run python -m experiments.setup_pilots --with-weights
```

Quick path check only:

```bash
uv run python -m experiments.setup_pilots --only check
```

One pilot only:

```bash
uv run python -m experiments.setup_pilots --only yourmt3
uv run python -m experiments.setup_pilots --only streamgen
```

---

**4. Confirm**

```bash
ls experiments/transcription/YOURMT3_READY.md
ls experiments/streamgen/STREAMGEN_READY.md
ls /deepfreeze/share/SPDMX/SPDMX/songs.csv
head -1 analysis/paper_data/transcription_manifests/manifest_spdmx.csv
head -1 analysis/paper_data/streamgen_spdmx_index/spdmx_multitrack.jsonl
```

---

**Do not**

- Do not change `SPDMX_OUTPUT_DIR` away from `/deepfreeze/share/SPDMX` unless
  you know you are on a personal tree
- Do not `git push --force` or rewrite shared deepfreeze data
- Do not expect YourMT3 and StreamGen `transformers` pins to coexist forever in
  one venv — if one breaks, re-run that pilot’s setup with `--skip-clone`, or
  use a separate venv

When setup finishes, open `experiments/PILOTS_READY.md` and the per-pilot
`*_READY.md` files for train commands.
