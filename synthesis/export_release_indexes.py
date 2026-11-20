"""Export gzipped release indexes (stems / songs / stem_soundfonts) for GitHub.

Lab usage::

    uv run python -m synthesis.export_release_indexes \\
      --spdmx-root /deepfreeze/share/SPDMX/SPDMX \\
      --recipe /deepfreeze/share/SPDMX/dev/final/stem_recipe.csv \\
      --out-dir indexes
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import pandas as pd

from shared.config import ABLATION_SAMPLE_SEED, SOUNDFONT_PATH

STEM_SOUNDFONTS_COLUMNS = [
    "song_id",
    "track",
    "category",
    "ablation",
    "method",
    "backend",
    "soundfont_id",
    "soundfont_id_locked",
    "fx_profile",
    "provenance",
]


def _song_id_from_recipe_path(path: str) -> str:
    """Map ``…/raw/<song_id>`` (or already-relative) recipe paths to ``song_id``."""
    text = str(path).replace("\\", "/").rstrip("/")
    parts = text.split("/")
    if "raw" in parts:
        i = parts.index("raw")
        return "/".join(parts[i + 1 :])
    # Already song_id-shaped (a/b/Qm…).
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return text


def _soundfont_id_for_filepath(soundfont_filepath: str | None) -> str | None:
    if not soundfont_filepath:
        return None
    from experiments.patch_sweep.config import load_combined_soundfont_catalog

    name = Path(soundfont_filepath).name
    for entry in load_combined_soundfont_catalog().get("candidates", []):
        if Path(entry.get("file", "")).name == name:
            return str(entry["id"])
    return Path(soundfont_filepath).stem


def _basic_soundfont_id() -> str:
    return _soundfont_id_for_filepath(SOUNDFONT_PATH) or "sgm_v2"


def _gzip_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copy2(src, dest)
        return
    with open(src, "rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout)


def build_stem_soundfonts(
    stems: pd.DataFrame,
    recipe: pd.DataFrame,
    *,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    reconstruct_slakh: bool = False,
) -> pd.DataFrame:
    """Join release stems to production ``stem_recipe`` and resolve soundfont ids.

    Historical ``slakh`` / varied-soundfont stems did **not** log the picked bank,
    and the old ``hash()``-based seed was process-randomized under ``spawn``
    workers — exact recovery is impossible. Those rows get
    ``provenance=slakh_unlogged`` and an empty ``soundfont_id`` unless
    ``reconstruct_slakh`` is set (stable sha256 replay; **not** bit-exact to the
    original render).
    """
    from synthesis.patches import SLAKH_CATEGORY_RENDER
    from experiments.patch_sweep.winners import pick_fx_profile, pick_soundfont_id

    basic_id = _basic_soundfont_id()
    locked_by_category = {
        cat: (cfg.get("soundfont_id") and str(cfg["soundfont_id"])) or None
        for cat, cfg in SLAKH_CATEGORY_RENDER.items()
    }

    rec = recipe.copy()
    rec["song_id"] = rec["path"].map(_song_id_from_recipe_path)
    rec["track"] = rec["track"].astype(int)
    rec = rec.drop_duplicates(subset=["song_id", "track"], keep="last")

    stem_keys = stems[["song_id", "track"]].copy()
    stem_keys["track"] = stem_keys["track"].astype(int)
    keep_cols = [
        c
        for c in (
            "song_id",
            "track",
            "category",
            "ablation",
            "method",
            "backend",
            "fallback",
            "soundfont_id",
            "fx_profile",
            "path",
        )
        if c in rec.columns
    ]
    out = stem_keys.merge(rec[keep_cols], on=["song_id", "track"], how="left")

    for col in ("category", "ablation", "method", "backend", "fallback", "path"):
        if col not in out.columns:
            out[col] = pd.NA
    if "soundfont_id" not in out.columns:
        out["soundfont_id"] = pd.NA
    if "fx_profile" not in out.columns:
        out["fx_profile"] = pd.NA

    recorded = out["soundfont_id"].fillna("").astype(str).str.strip()
    has_recorded = recorded != ""
    backend = out["backend"].fillna("").astype(str)
    method = out["method"].fillna("").astype(str)
    ablation = out["ablation"].fillna("").astype(str)
    fallback = out["fallback"].fillna("").astype(str)
    missing_recipe = out["backend"].isna()

    is_neural = backend.isin(["midi_ddsp", "ddsp_piano"])
    is_basic = (method == "basic") | (ablation == "basic") | (
        (method == "midi-ddsp") & (fallback == "basic")
    )
    is_slakh = (
        (method == "slakh")
        | ablation.isin(["slakh", "ddsp_slakh"])
        | ((method == "midi-ddsp") & (fallback == "slakh"))
    )

    provenance = pd.Series("basic_default", index=out.index, dtype=object)
    soundfont_id = pd.Series(basic_id, index=out.index, dtype=object)
    fx_profile = out["fx_profile"]
    soundfont_id_locked = out["category"].map(locked_by_category)

    provenance = provenance.mask(is_basic, "basic_default")
    soundfont_id = soundfont_id.mask(is_basic, basic_id)
    soundfont_id_locked = soundfont_id_locked.mask(is_basic, basic_id)

    provenance = provenance.mask(is_slakh, "slakh_unlogged")
    soundfont_id = soundfont_id.mask(is_slakh, pd.NA)
    # keep category locked id for slakh

    provenance = provenance.mask(is_neural, "neural")
    soundfont_id = soundfont_id.mask(is_neural, pd.NA)
    fx_profile = fx_profile.mask(is_neural, pd.NA)
    soundfont_id_locked = soundfont_id_locked.mask(is_neural, pd.NA)

    provenance = provenance.mask(missing_recipe, "missing_recipe")
    soundfont_id = soundfont_id.mask(missing_recipe, pd.NA)
    soundfont_id_locked = soundfont_id_locked.mask(missing_recipe, pd.NA)

    provenance = provenance.mask(has_recorded, "recorded")
    soundfont_id = soundfont_id.mask(has_recorded, recorded)

    if reconstruct_slakh:
        slakh_idx = out.index[is_slakh & ~has_recorded]
        for i in slakh_idx:
            category = out.at[i, "category"]
            if pd.isna(category):
                continue
            category = str(category)
            cfg = SLAKH_CATEGORY_RENDER.get(category) or {}
            ids = list(cfg.get("soundfont_ids") or [])
            locked = cfg.get("soundfont_id")
            if not ids and locked:
                ids = [str(locked)]
            song_path = out.at[i, "path"]
            if not ids or pd.isna(song_path) or not song_path:
                continue
            soundfont_id.at[i] = pick_soundfont_id(
                ids,
                category=category,
                song_path=str(song_path),
                sample_seed=sample_seed,
            )
            fx_profiles = list(cfg.get("fx_profiles") or [])
            if not fx_profiles and cfg.get("fx_profile"):
                fx_profiles = [cfg["fx_profile"]]
            if fx_profiles:
                fx_profile.at[i] = pick_fx_profile(
                    fx_profiles,
                    category=category,
                    song_path=str(song_path),
                    sample_seed=sample_seed,
                )
            provenance.at[i] = "slakh_reconstructed"

    result = pd.DataFrame({
        "song_id": out["song_id"],
        "track": out["track"].astype(int),
        "category": out["category"],
        "ablation": out["ablation"],
        "method": out["method"],
        "backend": out["backend"],
        "soundfont_id": soundfont_id,
        "soundfont_id_locked": soundfont_id_locked,
        "fx_profile": fx_profile,
        "provenance": provenance,
    })
    return result[STEM_SOUNDFONTS_COLUMNS]


def export_release_indexes(
    *,
    spdmx_root: Path,
    out_dir: Path,
    recipe_path: Path | None = None,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    reconstruct_slakh: bool = False,
    skip_soundfonts: bool = False,
) -> dict[str, Path]:
    """Write ``stems.csv.gz``, ``songs.csv.gz``, and optionally ``stem_soundfonts.csv.gz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name in ("stems.csv", "songs.csv"):
        src = spdmx_root / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dest = out_dir / f"{name}.gz"
        _gzip_copy(src, dest)
        written[name] = dest

    if skip_soundfonts:
        return written

    if recipe_path is None:
        # Default lab layout: {OUTPUT_DIR}/dev/final/stem_recipe.csv
        recipe_path = spdmx_root.parent / "dev" / "final" / "stem_recipe.csv"
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"stem_recipe.csv not found at {recipe_path}; pass --recipe"
        )

    stems = pd.read_csv(spdmx_root / "stems.csv", usecols=["song_id", "track"])
    recipe = pd.read_csv(recipe_path)
    table = build_stem_soundfonts(
        stems,
        recipe,
        sample_seed=sample_seed,
        reconstruct_slakh=reconstruct_slakh,
    )
    dest = out_dir / "stem_soundfonts.csv.gz"
    with gzip.open(dest, "wt", compresslevel=9, encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False)
    written["stem_soundfonts.csv"] = dest
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spdmx-root",
        type=Path,
        required=True,
        help="Packaged SPDMX tree containing stems.csv / songs.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("indexes"),
        help="Destination directory (default: ./indexes)",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="Production stem_recipe.csv (default: <spdmx-root>/../dev/final/stem_recipe.csv)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=ABLATION_SAMPLE_SEED,
        help="Seed used for slakh reconstruction (default: %(default)s)",
    )
    parser.add_argument(
        "--reconstruct-slakh",
        action="store_true",
        help=(
            "Fill slakh soundfont_id via stable sha256 replay (not bit-exact to the "
            "historical spawn/hash() picks)"
        ),
    )
    parser.add_argument(
        "--skip-soundfonts",
        action="store_true",
        help="Only gzip stems.csv / songs.csv",
    )
    args = parser.parse_args(argv)
    written = export_release_indexes(
        spdmx_root=args.spdmx_root,
        out_dir=args.out_dir,
        recipe_path=args.recipe,
        sample_seed=args.sample_seed,
        reconstruct_slakh=args.reconstruct_slakh,
        skip_soundfonts=args.skip_soundfonts,
    )
    for label, path in written.items():
        print(f"{label} -> {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
