# SPDMX release indexes

Gzipped copies of the public corpus tables so you can browse and filter without
downloading the ~1.5 TB audio archives.

| File | Rows | Description |
|------|------|-------------|
| [`stems.csv.gz`](stems.csv.gz) | one per stem | `song_id`, `track`, `program`, `chunk`, paths, … |
| [`songs.csv.gz`](songs.csv.gz) | one per song | subsets (`subset:bdgp`, `subset:multitrack`, …), `programs`, … |
| [`stem_soundfonts.csv.gz`](stem_soundfonts.csv.gz) | one per stem | render backend + soundfont provenance |

Schema and joins: [`synthesis/spdmx_release/README.md`](../synthesis/spdmx_release/README.md).

### `stem_soundfonts.csv` notes

| Column | Meaning |
|--------|---------|
| `soundfont_id` | Bank actually used when known (`provenance=recorded` / `basic_default` / neural empty) |
| `soundfont_id_locked` | Per-category listening winner from `winners_locked.yaml` (slakh pool primary) |
| `provenance` | `recorded`, `basic_default`, `neural`, `slakh_unlogged`, … |

Most production stems used the **varied (slakh) soundfont pool**. The v1 render did not log the per-stem pick, and the old RNG seed used process-randomized `hash()` under `spawn` workers, so those rows are `slakh_unlogged` with empty `soundfont_id`. Use `soundfont_id_locked` + category for the tuned primary bank; pool members are in [`experiments/patch_sweep/winners_locked.yaml`](../experiments/patch_sweep/winners_locked.yaml). New renders write `soundfont_id` into `stem_recipe.csv`.

```python
import pandas as pd

songs = pd.read_csv("songs.csv.gz")
stems = pd.read_csv("stems.csv.gz")
sfs = pd.read_csv("stem_soundfonts.csv.gz")

# Example: songs that include GM choir / voice programs
voiceish = songs[
    songs["programs"].str.contains(r"(^|\|)(5[2-5]|85)(\||$)", regex=True)
]
```

Regenerate from a packaged tree (lab):

```bash
uv run python -m synthesis.export_release_indexes \
  --spdmx-root /deepfreeze/share/SPDMX/SPDMX \
  --recipe /deepfreeze/share/SPDMX/dev/final/stem_recipe.csv \
  --out-dir indexes
```
