# shared

Cross-cutting configuration and constants used by synthesis and analysis.

## Files

| File | Description |
|------|-------------|
| `config.py` | Ablation sample settings, table column schemas, audio/render constants, SA3 duration limits; re-exports path settings from `.env` |
| `env.py` | Load repo `.env` and resolve `SPDMX_*` path variables |
| `repo_symlinks.py` | In-repo symlink helpers for `analysis/output/` and `synthesis/ablations_output/` |
| `setup_symlinks.py` | CLI: `python -m shared.setup_symlinks` — run after clone on a new machine |

Set machine paths in repo-root `.env` (see `.env.example`). Key paths imported from `shared.config`:

- `OUTPUT_DIR` — from `SPDMX_OUTPUT_DIR`
- `{OUTPUT_DIR}/dev/` — development artifacts (ablations, analysis, interim stems)
- `{OUTPUT_DIR}/SPDMX_dev/` — flat production render (`LICENSE`, `README.md`, `stems.csv`, `raw/`, `audio/`, `mid/`). Untouched by packaging.
- `{OUTPUT_DIR}/SPDMX/` — chunked distributable from `build_spdmx` (`chunk_N/`, packaged `stems.csv` with `chunk`, `chunks.csv`).
- `{OUTPUT_DIR}/dev/final/` — production synthesis tables (`data.csv`, `stems.csv`, per-pass `stems.<engine>.csv`)
