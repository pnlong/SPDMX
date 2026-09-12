"""Per-engine CSV shards for hybrid synthesis; merged before mix/realify."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.config import (
    DATA_DIR_NAME,
    NA_STRING,
    SONGS_TABLE_COLUMNS,
    SPDMX_RAW_DIR_NAME,
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


def rewrite_raw_path_to_media(path: str, media_dir: str | Path) -> str:
    """Map any ``…/raw/<song_id>`` path to ``{media_dir}/raw/<song_id>``."""
    sid = _song_id_from_audio_dir(path)
    if not sid:
        return str(path)
    return str(Path(media_dir) / SPDMX_RAW_DIR_NAME / sid)


def rewrite_dataframe_raw_paths(
    df: pd.DataFrame,
    media_dir: str | Path,
    *,
    column: str = "path",
) -> pd.DataFrame:
    """Rewrite ``column`` song dirs onto ``media_dir/raw`` (no-op if column missing)."""
    if df is None or not len(df) or column not in df.columns:
        return df
    out = df.copy()
    media = Path(media_dir)
    out[column] = [
        rewrite_raw_path_to_media(str(p), media) for p in out[column].tolist()
    ]
    return out


def rewrite_tables_raw_paths(
    tables_dir: str | Path,
    media_dir: str | Path,
    *,
    include_shards: bool = True,
) -> dict[str, int]:
    """Rewrite ``path`` columns in final tables (and optional pass shards) to ``media_dir/raw``.

    Dedupes ``(path, track)`` with last-row-wins after rewrite so legacy
    ``…/SPDMX/raw/…`` and newer ``…/SPDMX_dev/raw/…`` rows collapse.
    """
    root = Path(tables_dir)
    media = Path(media_dir)
    rewritten: dict[str, int] = {}

    targets: list[tuple[Path, list[str] | None]] = [
        (root / f"{DATA_DIR_NAME}.csv", None),
        (canonical_stems_csv(root), ["path", "track"]),
        (canonical_recipe_csv(root), ["path", "track"]),
        (root / DDSP_ROUTING_FILE_NAME, ["path", "track"]),
    ]
    if include_shards:
        for name in RENDER_PASSES:
            targets.append((pass_stems_csv(root, name), ["path", "track"]))
            targets.append((pass_recipe_csv(root, name), ["path", "track"]))
            targets.append((pass_routing_csv(root, name), ["path", "track"]))

    for path, key_cols in targets:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        df = _read_csv(path)
        if df.empty or "path" not in df.columns:
            continue
        df = rewrite_dataframe_raw_paths(df, media)
        if key_cols and set(key_cols) <= set(df.columns):
            df = df.drop_duplicates(key_cols, keep="last")
        elif path.name == f"{DATA_DIR_NAME}.csv":
            df = df.drop_duplicates(["path"], keep="last")
        df.to_csv(path, index=False, na_rep=NA_STRING)
        rewritten[path.name] = len(df)
        print(
            f"Rewrote paths → {media / SPDMX_RAW_DIR_NAME}: {path.name} "
            f"({len(df)} rows)",
            flush=True,
        )
    return rewritten


def _normalize_stems_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``zero_duration_notes`` to real True/False (missing → False)."""
    if df is None or not len(df):
        return df
    col = "zero_duration_notes"
    if col not in df.columns:
        # Legacy one-shot column name before rename.
        if "zero_duration" in df.columns:
            df = df.rename(columns={"zero_duration": col})
        else:
            df[col] = False
            return df
    raw = df[col]
    truthy = {"true", "1", "yes"}
    falsy = {"false", "0", "no", "", "nan", "none", "na", "<na>"}
    out = []
    for value in raw.tolist():
        if value is True or value is False:
            out.append(bool(value))
            continue
        if value is None or (isinstance(value, float) and value != value):
            out.append(False)
            continue
        text = str(value).strip().lower()
        if text in truthy:
            out.append(True)
        elif text in falsy:
            out.append(False)
        else:
            out.append(bool(value))
    df = df.copy()
    df[col] = out
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    # low_memory=False avoids chunked dtype guesses on large stem tables.
    return pd.read_csv(path, low_memory=False)


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


def merge_pass_tables(
    tables_dir: str | Path,
    *,
    media_dir: str | Path | None = None,
) -> dict[str, int]:
    """Write canonical stems/recipe/routing/data CSVs from per-pass shards.

    Shards are read-only inputs: mix/merge never deletes or rewrites
    ``stems.<pass>.csv`` / ``stem_recipe.<pass>.csv`` / ``ddsp_routing.<pass>.csv``,
    so a later re-render or recipe change can still append to them.

    When ``media_dir`` is set, ``path`` columns are rewritten to
    ``{media_dir}/raw/<song_id>`` so canonical tables match the live tree
    (``SPDMX_dev``) even if shards still say ``SPDMX/raw``.
    Returns row counts written (stems, recipes, songs).
    """
    root = Path(tables_dir)
    stem_paths = [pass_stems_csv(root, name) for name in RENDER_PASSES]
    stems = _concat_dedup(stem_paths, ["path", "track"])
    if media_dir is not None and len(stems):
        stems = rewrite_dataframe_raw_paths(stems, media_dir)
        stems = stems.drop_duplicates(["path", "track"], keep="last")
    if not len(stems):
        stems = pd.DataFrame(columns=STEMS_TABLE_COLUMNS)
    else:
        stems = _normalize_stems_bool_columns(stems)
        for col in STEMS_TABLE_COLUMNS:
            if col not in stems.columns:
                stems[col] = False if col == "zero_duration_notes" else pd.NA
        stems = stems[STEMS_TABLE_COLUMNS]
    stems.to_csv(canonical_stems_csv(root), index=False, na_rep=NA_STRING)

    recipe_paths = [pass_recipe_csv(root, name) for name in RENDER_PASSES]
    recipes = _concat_dedup(recipe_paths, ["path", "track"])
    if media_dir is not None and len(recipes):
        recipes = rewrite_dataframe_raw_paths(recipes, media_dir)
        recipes = recipes.drop_duplicates(["path", "track"], keep="last")
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
    if media_dir is not None and len(routing):
        routing = rewrite_dataframe_raw_paths(routing, media_dir)
        routing = routing.drop_duplicates(["path", "track"], keep="last")
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
