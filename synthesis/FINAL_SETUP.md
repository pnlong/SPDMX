# Join sPDMX final synthesis (new GPU machine)

Copy everything below the line into Slack and send. Assumes Deep Freeze is
mounted and production output already lives at **`/deepfreeze/share/SPDMX`**
(layout, MIDI index, and partial renders are already there).

Your only job: **MIDI-DDSP rendering** (`--only-pass midi_ddsp`).

---

## Slack message (copy from here)

**sPDMX — set up a new machine for MIDI-DDSP rendering**

Production data is on shared Deep Freeze at `/deepfreeze/share/SPDMX`. Install the repo, point at that tree, pick a shard index, and start rendering. Please read the “do not” list at the bottom.

**Prerequisites:** Deep Freeze mounted, NVIDIA GPU (`nvidia-smi` works), Linux x86_64, git.

---

**1. Clone and paths**

```bash
git clone git@github.com:pnlong/SPDMX.git ~/spdmx
cd ~/spdmx
cp .env.example .env
```

Edit `~/spdmx/.env`:

```
SPDMX_PDMX_FILEPATH="/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
SPDMX_OUTPUT_DIR="/deepfreeze/share/SPDMX"
```

---

**2. Install uv + Python 3.10 + project venv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# add that export to ~/.bashrc if needed

uv python install 3.10
cd ~/spdmx
uv sync
uv run python -m shared.setup_symlinks
uv run python -c "import synthesis.final; print('spdmx ok')"
```

---

**3. MIDI-DDSP GPU venv**

```bash
cd ~/spdmx
bash synthesis/ddsp/bootstrap_venv.sh
nvidia-smi
```

Weights should already be on Deep Freeze. If missing:

```bash
.venv-ddsp/bin/midi_ddsp_download_model_weights
```

---

**4. Confirm you can write the shared tree**

```bash
OUTPUT=/deepfreeze/share/SPDMX
touch "$OUTPUT/dev/final/.write_test" && rm "$OUTPUT/dev/final/.write_test"
mkdir -p "$OUTPUT/SPDMX/raw/.write_test" && rmdir "$OUTPUT/SPDMX/raw/.write_test"
echo "write access ok"
```

If that fails, ping me — you need write access to `dev/final/` and `SPDMX/raw/`.

---

**5. Coordinate your shard**

We split songs across machines with `--shard-count` and `--shard-index`:

- **`--shard-count N`** — total number of machines (ask me what N is today)
- **`--shard-index k`** — your machine, **0-based**, must be **unique** per box

Example for **3 machines**:

| Machine | `--shard-index` | GPUs (example) |
|---------|-----------------|----------------|
| A | 0 | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| B | 1 | `CUDA_VISIBLE_DEVICES=0,1` |
| C | 2 | `CUDA_VISIBLE_DEVICES=0,1` |

GPU count on your box only affects speed — not how many songs you own.

---

**6. Start MIDI-DDSP**

Replace `N` and `k` with the agreed shard count and your index:

```bash
cd ~/spdmx
export SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python
export CUDA_VISIBLE_DEVICES=0,1   # use the GPUs on YOUR box only

uv run python -m synthesis.final --only-pass midi_ddsp \
  --shard-count N --shard-index k --refresh-every 10
```

Example — **machine B, 2 GPUs, 3 machines total:**

```bash
cd ~/spdmx
export SPDMX_DDSP_PYTHON=$PWD/.venv-ddsp/bin/python
export CUDA_VISIBLE_DEVICES=0,1

uv run python -m synthesis.final --only-pass midi_ddsp \
  --shard-count 3 --shard-index 1 --refresh-every 10
```

Progress log: `/deepfreeze/share/SPDMX/dev/final/synthesis.midi_ddsp.log`

Resume is automatic — stems already in `stem_recipe.midi_ddsp.csv` are skipped
(CSV-only by default). After all shards finish and rsync completes, one machine
runs `verify` then `mix`.

---

**Do not**

- Do **not** run `--only-pass layout`, `prepare_synthesis`, or `--reset`
- Do **not** use the same `--shard-index` as another running machine
- Do **not** use a different `--shard-count` than the rest of the fleet

---

**If something breaks**

- Permission denied → need write access on `/deepfreeze/share/SPDMX`
- Same songs as another box → duplicate shard index; stop and re-coordinate
- `midi_index.csv` missing → ping me (layout not done on shared tree)

## End of Slack message
