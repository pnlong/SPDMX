"""Per-engine CSV shards for hybrid synthesis; merged before mix/realify."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.config import (
    DATA_DIR_NAME,
    NA_STRING,
    SONGS_TABLE_COLUMNS,
    STEMS_FILE_NAME,
    STEMS_TABLE_COLUMNS,
)
from synthesis.ddsp.config import DDSP_ROUTING_COLUMNS, DDSP_ROUTING_FILE_NAME
from synthesis.paths import MIDI_INDEX_FILE_NAME
from synthesis.recipe import STEM_RECIPE_COLUMNS, STEM_RECIPE_FILE_NAME


def canonical_stems_csv(tables_dir: str | Path) -> Path:
    return Path(tables_dir) / f"{STEMS_FILE_NAME}.csv"


def canonical_recipe_csv(tables_dir: str | Path) -> Path:
    return Path(tables_dir) / STEM_RECIPE_FILE_NAME


def drop_canonical_tables(tables_dir: str | Path) -> None:
    """Remove merge outputs. Render progress lives only in per-pass shards."""
    root = Path(tables_dir)
    for path in (
        canonical_stems_csv(root),
        canonical_recipe_csv(root),
        root / f"{DATA_DIR_NAME}.csv",
        root / DDSP_ROUTING_FILE_NAME,
    ):
        path.unlink(missing_ok=True)
        Path(str(path) + ".lock").unlink(missing_ok=True)


RENDER_PASSES = ("fluidsynth", "ddsp_piano", "midi_ddsp")


def pass_stems_csv(tables_dir: str | Path, pass_name: str) -> Path:
    return Path(tables_dir) / f"{STEMS_FILE_NAME}.{pass_name}.csv"


def pass_recipe_csv(tables_dir: str | Path, pass_name: str) -> Path:
    return Path(tables_dir) / f"stem_recipe.{pass_name}.csv"


def pass_routing_csv(tables_dir: str | Path, pass_name: str) -> Path:
    return Path(tables_dir) / f"ddsp_routing.{pass_name}.csv"


def _song_id_from_audio_dir(path: str) -> str:
    text = str(path).replace("\\", "/")
    for marker in ("/raw/", "/audio/", "/audio_summable/"):
        if marker in text:
            return text.split(marker, 1)[1].strip("/")
    return Path(path).name


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _concat_dedup(paths: list[Path], key_cols: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = _read_csv(path)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not key_cols or not set(key_cols) <= set(out.columns):
        return out
    return out.drop_duplicates(key_cols, keep="last")


def merge_pass_tables(tables_dir: str | Path) -> dict[str, int]:
    """Write canonical stems/recipe/routing/data CSVs from per-pass shards.

    Shards are read-only inputs: mix/merge never deletes or rewrites
    ``stems.<pass>.csv`` / ``stem_recipe.<pass>.csv`` / ``ddsp_routing.<pass>.csv``,
    so a later re-render or recipe change can still append to them.
    Returns row counts written (stems, recipes, songs).
    """
    root = Path(tables_dir)
    stem_paths = [pass_stems_csv(root, name) for name in RENDER_PASSES]
    stems = _concat_dedup(stem_paths, ["path", "track"])
    if not len(stems):
        stems = pd.DataFrame(columns=STEMS_TABLE_COLUMNS)
    elif set(STEMS_TABLE_COLUMNS) <= set(stems.columns):
        stems = stems[STEMS_TABLE_COLUMNS]
    stems.to_csv(canonical_stems_csv(root), index=False, na_rep=NA_STRING)

    recipe_paths = [pass_recipe_csv(root, name) for name in RENDER_PASSES]
    recipes = _concat_dedup(recipe_paths, ["path", "track"])
    if not len(recipes):
        recipes = pd.DataFrame(columns=STEM_RECIPE_COLUMNS)
    else:
        from synthesis.recipe import BACKEND_PENDING_MIDI_DDSP

        # Bookkeeping-only rows: Fluidsynth deferred true MIDI-DDSP stems (no audio).
        if "backend" in recipes.columns:
            recipes = recipes[
                recipes["backend"].astype(str) != BACKEND_PENDING_MIDI_DDSP
            ]
        for col in STEM_RECIPE_COLUMNS:
            if col not in recipes.columns:
                recipes[col] = pd.NA
        recipes = recipes[STEM_RECIPE_COLUMNS]
    recipes.to_csv(canonical_recipe_csv(root), index=False, na_rep=NA_STRING)

    routing_paths = [pass_routing_csv(root, name) for name in RENDER_PASSES]
    routing = _concat_dedup(routing_paths, ["path", "track"])
    routing_out = root / DDSP_ROUTING_FILE_NAME
    if len(routing):
        if set(DDSP_ROUTING_COLUMNS) <= set(routing.columns):
            routing = routing[DDSP_ROUTING_COLUMNS]
        routing.to_csv(routing_out, index=False, na_rep=NA_STRING)

    n_songs = 0
    data_csv = root / f"{DATA_DIR_NAME}.csv"
    rows = []
    index_path = root / MIDI_INDEX_FILE_NAME
    if len(stems) and "path" in stems.columns and index_path.is_file():
        index = pd.read_csv(index_path, usecols=["song_id", "n_tracks"])
        need = index.drop_duplicates("song_id").set_index("song_id")["n_tracks"]
        counts = stems.groupby("path")["track"].nunique()
        for path, n in counts.items():
            sid = _song_id_from_audio_dir(str(path))
            if sid not in need.index:
                continue
            want = int(need.loc[sid])
            if n >= want:
                rows.append({"path": path, "n_tracks": want})
    songs = pd.DataFrame(rows)
    n_songs = len(songs)
    for col in SONGS_TABLE_COLUMNS:
        if col not in songs.columns:
            songs[col] = pd.NA
    songs[SONGS_TABLE_COLUMNS].to_csv(data_csv, index=False, na_rep=NA_STRING)
    print(
        f"Merged pass tables: {len(stems)} stems, {len(recipes)} recipes, "
        f"{n_songs} complete songs → {root} (pass shards kept)",
        flush=True,
    )
    return {"stems": len(stems), "recipes": len(recipes), "songs": n_songs}
