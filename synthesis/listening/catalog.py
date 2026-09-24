"""Build a JSON-serializable catalog from ablation CSV metadata and audio files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SEED,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    LISTENING_PREFER_SUMMABLE,
    OUTPUT_DIR,
    STEMS_FILE_NAME,
)
from shared.repo_symlinks import REPO_ABLATIONS_SYMLINK
from synthesis.audio import (
    encode_audio_bytes,
    list_stem_paths,
    mixture_filename,
    stem_filename,
    stem_path,
    sum_stems_in_song_dir,
    track_index_from_stem_name,
)
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES, resolve_probe_category
from synthesis.paths import ablations_root
from synthesis.paths import resolve_output_song_dir

# Stable order for sidebar filters (probe / slakh listening categories).
LISTENING_CATEGORY_ORDER: tuple[str, ...] = tuple(LISTENING_CATEGORY_GM_CLASSES.keys())

CONDITION_ORDER = (
    "basic",
    "basic_realify",
    "slakh",
    "slakh_realify",
    "ddsp_basic",
    "ddsp_basic_realify",
    "ddsp_slakh",
    "ddsp_slakh_realify",
)

CONDITION_LABELS: dict[str, str] = {
    "basic": "A1",
    "basic_realify": "A2",
    "slakh": "B1",
    "slakh_realify": "B2",
    "ddsp_basic": "CA1",
    "ddsp_basic_realify": "CA2",
    "ddsp_slakh": "CB1",
    "ddsp_slakh_realify": "CB2",
}

REALIFY_CONDITIONS = frozenset({
    "basic_realify",
    "slakh_realify",
    "ddsp_basic_realify",
    "ddsp_slakh_realify",
})


@dataclass(frozen=True)
class ConditionInfo:
    id: str
    label: str
    name: str
    available: bool


def default_ablations_dir() -> Path:
    """Prefer ``ablations/clips/`` when present (10s listening set)."""
    if REPO_ABLATIONS_SYMLINK.is_dir():
        root = REPO_ABLATIONS_SYMLINK.resolve()
    else:
        root = Path(ablations_root(OUTPUT_DIR))
    clips = root / "clips"
    if clips.is_dir() and any(
        (_condition_dir(clips, condition) / f"{DATA_DIR_NAME}.csv").is_file()
        for condition in CONDITION_ORDER
    ):
        return clips
    return root


def song_id_from_path(song_path: str | Path) -> str:
    """Relative path under ``data/`` (e.g. ``7/19/QmPfj...``)."""
    parts = Path(song_path).parts
    if DATA_DIR_NAME not in parts:
        raise ValueError(f"Song path missing {DATA_DIR_NAME}/ segment: {song_path}")
    idx = parts.index(DATA_DIR_NAME)
    return str(Path(*parts[idx + 1:]))


def detect_audio_format(song_dir: Path) -> str | None:
    from shared.config import FLAC_AUDIO_FORMAT

    for fmt in (DEFAULT_AUDIO_FORMAT, FLAC_AUDIO_FORMAT):
        # Prefer stems (mixtures are written separately via synthesis.mix).
        if any(list_stem_paths(song_dir, fmt, require_valid=False)):
            return fmt
        if (song_dir / mixture_filename(fmt)).exists():
            return fmt
    return None


def summable_condition_name(condition: str) -> str:
    return f"{condition}_summable"


def _condition_dir(
    ablations_dir: Path,
    condition: str,
    *,
    prefer_summable: bool | None = None,
) -> Path:
    """Filesystem dir for a logical condition id (``basic`` → ``basic_summable`` when preferred)."""
    prefer = LISTENING_PREFER_SUMMABLE if prefer_summable is None else prefer_summable
    if prefer:
        return ablations_dir / summable_condition_name(condition)
    return ablations_dir / condition


def require_summable_condition_trees(ablations_dir: Path) -> None:
    """Raise if any listening condition is missing its ``*_summable`` tree with data.csv."""
    if not LISTENING_PREFER_SUMMABLE:
        return
    missing = []
    for condition in CONDITION_ORDER:
        cond_dir = ablations_dir / summable_condition_name(condition)
        if not (cond_dir / f"{DATA_DIR_NAME}.csv").is_file():
            missing.append(summable_condition_name(condition))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"LISTENING_PREFER_SUMMABLE=True but missing summable ablation(s) under "
            f"{ablations_dir}: {joined}\n"
            "Run: bash synthesis/mix_ablations.sh\n"
            "Or set LISTENING_PREFER_SUMMABLE=False in shared/config.py to use raw stems."
        )


def _song_dir_for_condition(
    ablations_dir: Path,
    reference_condition: str,
    song_path: str | Path,
    target_condition: str,
) -> Path:
    source_dir = _condition_dir(ablations_dir, reference_condition)
    target_dir = _condition_dir(ablations_dir, target_condition)
    return resolve_output_song_dir(Path(song_path), source_dir, target_dir)


def _audio_cell(
    ablations_dir: Path,
    reference_condition: str,
    song_path: str | Path,
    target_condition: str,
    track: int,
    audio_format: str,
) -> dict:
    song_dir = _song_dir_for_condition(
        ablations_dir, reference_condition, song_path, target_condition
    )
    audio_path = stem_path(song_dir, track, audio_format)
    filename = audio_path.name if audio_path.is_file() else stem_filename(track, audio_format)
    available = audio_path.is_file()
    song_id = song_id_from_path(song_path)
    return {
        "available": available,
        "url": f"/audio/{target_condition}/{song_id}/{filename}" if available else None,
    }


def _mixture_cell(
    ablations_dir: Path,
    reference_condition: str,
    song_path: str | Path,
    target_condition: str,
    audio_format: str,
) -> dict:
    """Mixture URL: on-disk ``mixture.*``, or summable-from-stems when mix file is absent."""
    song_dir = _song_dir_for_condition(
        ablations_dir, reference_condition, song_path, target_condition
    )
    filename = mixture_filename(audio_format)
    has_file = (song_dir / filename).is_file()
    has_stems = bool(list_stem_paths(song_dir, audio_format, require_valid=False))
    available = has_file or has_stems
    song_id = song_id_from_path(song_path)
    return {
        "available": available,
        "url": f"/audio/{target_condition}/{song_id}/{filename}" if available else None,
        "from_stems": available and not has_file,
    }


def stem_listening_category(stem_row: pd.Series) -> str:
    """Map a stems.csv row to a probe listening category (piano/guitar/drums/…)."""
    program = stem_row.get("program", 0)
    program_int = int(program) if pd.notna(program) else 0
    return resolve_probe_category(
        program=program_int,
        is_drum=bool(stem_row.get("is_drum", False)),
        track_name=_na_to_none(stem_row.get("name")),
    )


def _row_original_track(stem_row: pd.Series) -> int:
    track = int(stem_row["track"])
    if "original_track" not in stem_row.index:
        return track
    value = stem_row.get("original_track")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return track
    try:
        if pd.isna(value):
            return track
    except TypeError:
        pass
    return int(value)


def _stem_file_track_for_condition(
    *,
    condition_stems: pd.DataFrame | None,
    song_path: str,
    original_track: int,
    fallback_track: int,
) -> int:
    """On-disk stem index in ``condition`` for a logical ``original_track``."""
    if condition_stems is None or condition_stems.empty:
        return fallback_track
    # Realify CSVs remap ``path`` into their own tree; match by song id.
    song_id = song_id_from_path(song_path)
    rows = condition_stems[
        condition_stems["path"].astype(str).map(song_id_from_path) == song_id
    ]
    if rows.empty:
        return fallback_track
    if "original_track" in rows.columns:
        matched = rows[rows["original_track"].fillna(rows["track"]).astype(int) == original_track]
        if not matched.empty:
            return int(matched.iloc[0]["track"])
    matched = rows[rows["track"].astype(int) == original_track]
    if not matched.empty:
        return int(matched.iloc[0]["track"])
    matched = rows[rows["track"].astype(int) == fallback_track]
    if not matched.empty:
        return int(matched.iloc[0]["track"])
    return fallback_track


class AblationCatalog:
    def __init__(
        self,
        ablations_dir: Path,
        caption_seed: int = ABLATION_SAMPLE_SEED,
        *,
        include_mixtures: bool | None = None,
    ):
        self.ablations_dir = ablations_dir.resolve()
        self.caption_seed = caption_seed
        require_summable_condition_trees(self.ablations_dir)
        self.reference_condition = self._pick_reference_condition()
        # Clips trees are stem-only; hide mixtures by default there.
        if include_mixtures is None:
            include_mixtures = self.ablations_dir.name != "clips"
        self.include_mixtures = include_mixtures
        self._songs_df = self._load_songs()
        self._stems_df = self._load_stems()
        self._stems_by_condition: dict[str, pd.DataFrame] = {
            self.reference_condition: self._stems_df,
        }
        self._captions_df = self._build_captions()
        self._categories_by_path = self._index_song_categories()

    def _pick_reference_condition(self) -> str:
        for condition in CONDITION_ORDER:
            if (_condition_dir(self.ablations_dir, condition) / f"{DATA_DIR_NAME}.csv").is_file():
                return condition
        raise FileNotFoundError(
            f"No ablation condition with {DATA_DIR_NAME}.csv under {self.ablations_dir}"
        )

    def _load_songs(self) -> pd.DataFrame:
        path = _condition_dir(self.ablations_dir, self.reference_condition) / f"{DATA_DIR_NAME}.csv"
        return pd.read_csv(path)

    def _load_stems(self) -> pd.DataFrame:
        path = (
            _condition_dir(self.ablations_dir, self.reference_condition)
            / f"{STEMS_FILE_NAME}.csv"
        )
        if not path.is_file():
            return pd.DataFrame(
                columns=[
                    "path", "track", "original_track", "program", "is_drum", "name", "has_lyrics",
                ]
            )
        return pd.read_csv(path)

    def _condition_stems(self, condition: str) -> pd.DataFrame | None:
        if condition in self._stems_by_condition:
            return self._stems_by_condition[condition]
        path = _condition_dir(self.ablations_dir, condition) / f"{STEMS_FILE_NAME}.csv"
        if not path.is_file():
            self._stems_by_condition[condition] = pd.DataFrame()
            return self._stems_by_condition[condition]
        df = pd.read_csv(path)
        self._stems_by_condition[condition] = df
        return df

    def _build_captions(self) -> pd.DataFrame:
        if self._stems_df.empty or self._songs_df.empty:
            return pd.DataFrame(columns=["path", "track", "prompt"])
        # SA3 caption generation removed; listening UI works without prompts.
        out = self._stems_df[["path", "track"]].copy()
        out["prompt"] = ""
        return out

    def _index_song_categories(self) -> dict[str, list[str]]:
        """Map song path -> unique listening categories present in its stems."""
        if self._stems_df.empty:
            return {}
        by_path: dict[str, set[str]] = {}
        for _, stem_row in self._stems_df.iterrows():
            path = str(stem_row["path"])
            by_path.setdefault(path, set()).add(stem_listening_category(stem_row))
        order = {name: i for i, name in enumerate(LISTENING_CATEGORY_ORDER)}
        return {
            path: sorted(cats, key=lambda c: order.get(c, len(order)))
            for path, cats in by_path.items()
        }

    def conditions(self) -> list[dict]:
        result = []
        for condition in CONDITION_ORDER:
            cond_dir = _condition_dir(self.ablations_dir, condition)
            result.append(
                ConditionInfo(
                    id=condition,
                    label=CONDITION_LABELS[condition],
                    name=condition,
                    available=(cond_dir / f"{DATA_DIR_NAME}.csv").is_file(),
                ).__dict__
            )
        return result

    def categories(self) -> list[dict]:
        """Listening categories available for sidebar filtering."""
        present: set[str] = set()
        for cats in self._categories_by_path.values():
            present.update(cats)
        return [
            {"id": name, "label": name}
            for name in LISTENING_CATEGORY_ORDER
            if name in present
        ]

    def list_songs(self) -> list[dict]:
        songs = []
        for _, row in self._songs_df.iterrows():
            song_path = row["path"]
            song_id = song_id_from_path(song_path)
            duration = row.get("song_length.seconds")
            songs.append({
                "id": song_id,
                "title": _na_to_none(row.get("title")),
                "song_name": _na_to_none(row.get("song_name")),
                "artist_name": _na_to_none(row.get("artist_name")),
                "n_tracks": int(row.get("n_tracks", 0)),
                "genres": _na_to_none(row.get("genres")),
                "duration_seconds": _safe_float(duration),
                "categories": list(self._categories_by_path.get(str(song_path), [])),
            })
        return songs

    def get_song(self, song_id: str) -> dict | None:
        for _, row in self._songs_df.iterrows():
            if song_id_from_path(row["path"]) == song_id:
                return self._build_song_detail(row)
        return None

    def _build_song_detail(self, row: pd.Series) -> dict:
        song_path = row["path"]
        song_id = song_id_from_path(song_path)
        ref_song_dir = _song_dir_for_condition(
            self.ablations_dir,
            self.reference_condition,
            song_path,
            self.reference_condition,
        )
        audio_format = detect_audio_format(ref_song_dir) or DEFAULT_AUDIO_FORMAT

        stems_rows = self._stems_df[self._stems_df["path"] == song_path].sort_values("track")
        stems = []
        for _, stem_row in stems_rows.iterrows():
            track = int(stem_row["track"])
            original_track = _row_original_track(stem_row)
            conditions = {}
            for condition in CONDITION_ORDER:
                file_track = _stem_file_track_for_condition(
                    condition_stems=self._condition_stems(condition),
                    song_path=str(song_path),
                    original_track=original_track,
                    fallback_track=track,
                )
                conditions[condition] = _audio_cell(
                    self.ablations_dir,
                    self.reference_condition,
                    song_path,
                    condition,
                    file_track,
                    audio_format,
                )
            caption = None
            if not self._captions_df.empty:
                cap_rows = self._captions_df[
                    (self._captions_df["path"] == song_path)
                    & (self._captions_df["track"] == track)
                ]
                if not cap_rows.empty:
                    caption = _na_to_none(cap_rows.iloc[0].get("prompt"))

            stems.append({
                "track": track,
                "original_track": original_track,
                "name": _na_to_none(stem_row.get("name")) or f"Track {track}",
                "program": int(stem_row["program"]) if pd.notna(stem_row.get("program")) else None,
                "is_drum": bool(stem_row.get("is_drum", False)),
                "category": stem_listening_category(stem_row),
                "caption": caption,
                "conditions": conditions,
            })

        mixture_filename_str = mixture_filename(audio_format)
        mixture = {
            condition: _mixture_cell(
                self.ablations_dir,
                self.reference_condition,
                song_path,
                condition,
                audio_format,
            )
            for condition in CONDITION_ORDER
        } if self.include_mixtures else {
            condition: {"available": False, "url": None, "from_stems": False}
            for condition in CONDITION_ORDER
        }

        return {
            "id": song_id,
            "path": str(song_path),
            "song_dirs": {
                condition: str(
                    _song_dir_for_condition(
                        self.ablations_dir,
                        self.reference_condition,
                        song_path,
                        condition,
                    )
                )
                for condition in CONDITION_ORDER
            },
            "title": _na_to_none(row.get("title")),
            "song_name": _na_to_none(row.get("song_name")),
            "subtitle": _na_to_none(row.get("subtitle")),
            "artist_name": _na_to_none(row.get("artist_name")),
            "composer_name": _na_to_none(row.get("composer_name")),
            "genres": _na_to_none(row.get("genres")),
            "n_tracks": int(row.get("n_tracks", 0)),
            "duration_seconds": _safe_float(row.get("song_length.seconds")),
            "audio_format": audio_format,
            "mixture": mixture,
            "include_mixtures": self.include_mixtures,
            "stems": stems,
        }

    def resolve_audio_path(self, condition: str, song_id: str, filename: str) -> Path | None:
        if condition not in CONDITION_ORDER:
            return None
        if ".." in Path(song_id).parts or ".." in Path(filename).parts:
            return None
        if "/" in filename or "\\" in filename:
            return None

        for _, row in self._songs_df.iterrows():
            if song_id_from_path(row["path"]) == song_id:
                song_dir = _song_dir_for_condition(
                    self.ablations_dir,
                    self.reference_condition,
                    row["path"],
                    condition,
                )
                audio_path = (song_dir / filename).resolve()
                if not str(audio_path).startswith(str(self.ablations_dir)):
                    return None
                if audio_path.is_file():
                    return audio_path
                fmt = Path(filename).suffix.lstrip(".")
                track = track_index_from_stem_name(filename, fmt)
                if track is None:
                    return None
                alt = stem_path(song_dir, track, fmt).resolve()
                if not str(alt).startswith(str(self.ablations_dir)):
                    return None
                return alt if alt.is_file() else None
        return None

    def resolve_song_dir(self, condition: str, song_id: str) -> Path | None:
        if condition not in CONDITION_ORDER:
            return None
        if ".." in Path(song_id).parts:
            return None
        for _, row in self._songs_df.iterrows():
            if song_id_from_path(row["path"]) == song_id:
                song_dir = _song_dir_for_condition(
                    self.ablations_dir,
                    self.reference_condition,
                    row["path"],
                    condition,
                ).resolve()
                if not str(song_dir).startswith(str(self.ablations_dir)):
                    return None
                return song_dir
        return None

    def mixture_audio_bytes(self, condition: str, song_id: str, audio_format: str) -> bytes | None:
        """Return on-disk mixture bytes, or sum stems when ``mixture.*`` is missing."""
        song_dir = self.resolve_song_dir(condition, song_id)
        if song_dir is None:
            return None
        mix_path = song_dir / mixture_filename(audio_format)
        if mix_path.is_file():
            return mix_path.read_bytes()
        try:
            mixture = sum_stems_in_song_dir(song_dir, audio_format)
        except FileNotFoundError:
            return None
        return encode_audio_bytes(mixture, audio_format)


def _na_to_none(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text == "NA":
        return None
    return text


def _safe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
