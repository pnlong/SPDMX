"""Write dense corrected MIDI copies (register applied, empty tracks dropped)."""

from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path

import mido
import pandas as pd

from shared.csv_tables import sanitize_track_name
from shared.config import (
    DEV_DIR_NAME,
    MID_CORRECTED_DIR_NAME,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_FILE_NAME,
    SPDMX_MID_DIR_NAME,
)
from synthesis.patches import PatchAssignment, apply_patch_to_midi_track

TRACK_MAP_FILE_NAME = f"{SPDMX_FILE_NAME}.csv"
LEGACY_TRACK_MAP_FILE_NAME = "track_map.csv"
# ``song_id`` is ``<shard>/<shard>/<hash>`` (PDMX ``./data/{song_id}.json``).
# Row identity in this table is (song_id, track).
TRACK_MAP_COLUMNS = [
    "song_id",
    "path",
    "mid",
    "track",
    "original_track",
    "program",
    "is_drum",
    "name",
]

_TRACK_MAPS_CACHE: dict[str, dict[str, dict[int, dict]]] = {}
_TRACK_MAPS_LOCK = threading.Lock()

_CONDUCTOR_META_TYPES = frozenset({
    "set_tempo",
    "time_signature",
    "key_signature",
    "smpte_offset",
})


def note_on_count(track) -> int:
    return sum(
        1
        for message in track
        if getattr(message, "type", None) == "note_on"
        and getattr(message, "velocity", 0) > 0
    )


def dest_rel_from_mid_col(mid_rel: str) -> str:
    """``./mid/a/b/Qm.mid`` or an absolute ``…/mid/a/b/Qm.mid`` → ``a/b/Qm.mid``."""
    text = str(mid_rel).replace("\\", "/")
    marker = f"/{SPDMX_MID_DIR_NAME}/"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx + len(marker) :]
    text = text.lstrip("./")
    if text.startswith(f"{SPDMX_MID_DIR_NAME}/"):
        text = text[len(SPDMX_MID_DIR_NAME) + 1 :]
    return text


def song_id_from_mid(mid_rel: str) -> str:
    """PDMX/sPDMX song id: ``./mid/a/b/Qm.mid`` → ``a/b/Qm``."""
    rel = dest_rel_from_mid_col(mid_rel)
    if rel.endswith(".mid"):
        rel = rel[: -len(".mid")]
    return rel


def song_id_from_pdmx_path(path_rel: str) -> str:
    """``./data/a/b/Qm.json`` → ``a/b/Qm``."""
    text = str(path_rel).replace("\\", "/").lstrip("./")
    if text.startswith("data/"):
        text = text[len("data/") :]
    if text.endswith(".json"):
        text = text[: -len(".json")]
    return text


def pdmx_path_from_mid(mid_rel: str) -> str:
    """PDMX metadata path: ``./mid/a/b/Qm.mid`` → ``./data/a/b/Qm.json``."""
    return f"./data/{song_id_from_mid(mid_rel)}.json"


def spdmx_audio_rel(song_id: str) -> str:
    """Dataset-relative audio directory: ``./audio/a/b/Qm``."""
    return f"./audio/{song_id}"


def spdmx_mid_rel(song_id: str) -> str:
    """Dataset-relative dense MIDI: ``./mid/a/b/Qm.mid``."""
    return f"./mid/{song_id}.mid"


def track_map_csv_path(corrected_midi_dir: str | Path) -> Path:
    """Preferred global track map: ``{SPDMX}/SPDMX.csv`` when mid is ``{SPDMX}/mid/``."""
    root = Path(corrected_midi_dir)
    if root.name == SPDMX_MID_DIR_NAME:
        return root.parent / TRACK_MAP_FILE_NAME
    return root / TRACK_MAP_FILE_NAME


def track_map_csv_candidates(corrected_midi_dir: str | Path) -> list[Path]:
    root = Path(corrected_midi_dir)
    primary = track_map_csv_path(root)
    locations = [primary]
    nested = root / TRACK_MAP_FILE_NAME
    if nested.resolve() != primary.resolve():
        locations.append(nested)
    if (
        root.name == SPDMX_MID_DIR_NAME
        and root.parent.name == SPDMX_DATASET_DIR_NAME
    ):
        locations.append(
            root.parent.parent
            / DEV_DIR_NAME
            / MID_CORRECTED_DIR_NAME
            / TRACK_MAP_FILE_NAME
        )
    candidates: list[Path] = []
    for path in locations:
        candidates.append(path)
        legacy = path.with_name(LEGACY_TRACK_MAP_FILE_NAME)
        if legacy.resolve() != path.resolve():
            candidates.append(legacy)
    return candidates


def resolve_track_map_csv(corrected_midi_dir: str | Path) -> Path:
    """First existing candidate, else the preferred path (for error messages)."""
    root = Path(corrected_midi_dir)
    for path in track_map_csv_candidates(root):
        if path.is_file():
            return path
    return track_map_csv_path(root)


def dest_rel_for_midi(midi_path: str | Path, corrected_midi_dir: str | Path) -> str:
    """Song id for a dense MIDI file (``a/b/Qm``)."""
    midi_path = Path(midi_path).resolve()
    corrected_midi_dir = Path(corrected_midi_dir).resolve()
    try:
        rel = str(midi_path.relative_to(corrected_midi_dir))
    except ValueError:
        rel = None
        parts = midi_path.parts
        for marker in (SPDMX_MID_DIR_NAME, MID_CORRECTED_DIR_NAME):
            if marker in parts:
                idx = parts.index(marker)
                rel = str(Path(*parts[idx + 1 :]))
                break
        if rel is None:
            rel = midi_path.name
    return song_id_from_mid(rel)


def clear_track_map_cache() -> None:
    with _TRACK_MAPS_LOCK:
        _TRACK_MAPS_CACHE.clear()


def load_track_maps(track_map_csv: str | Path) -> dict[str, dict[int, dict]]:
    """Load the global track map. Keys are ``song_id`` (``a/b/Qm``)."""
    path = Path(track_map_csv).resolve()
    cache_key = str(path)
    with _TRACK_MAPS_LOCK:
        cached = _TRACK_MAPS_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(
                f"Dense MIDI track map not found: {path}\n"
                "Re-run: uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
            )
        df = pd.read_csv(path)
        maps: dict[str, dict[int, dict]] = {}
        if "song_id" in df.columns:
            keys = df["song_id"].astype(str).to_numpy()
        elif "path" in df.columns:
            keys = df["path"].map(song_id_from_pdmx_path).astype(str).to_numpy()
        else:
            keys = df["mid"].map(song_id_from_mid).astype(str).to_numpy()
        tracks = df["track"].astype(int).to_numpy()
        originals = df["original_track"].astype(int).to_numpy()
        programs = (
            df["program"].fillna(0).astype(int).to_numpy()
            if "program" in df.columns
            else [0] * len(df)
        )
        drums = (
            df["is_drum"].fillna(False).astype(bool).to_numpy()
            if "is_drum" in df.columns
            else [False] * len(df)
        )
        if "name" in df.columns:
            names = df["name"].tolist()
        else:
            names = [None] * len(df)
        for i in range(len(df)):
            name = names[i]
            maps.setdefault(keys[i], {})[int(tracks[i])] = {
                "original_track": int(originals[i]),
                "program": int(programs[i]),
                "is_drum": bool(drums[i]),
                "name": None if name is None or (isinstance(name, float) and name != name) else str(name),
            }
        _TRACK_MAPS_CACHE[cache_key] = maps
        return maps


def load_track_map(
    midi_path: str | Path,
    *,
    corrected_midi_dir: str | Path,
    track_maps: dict[str, dict[int, dict]] | None = None,
) -> dict[int, dict]:
    """Map dense ``track`` → row dict from the global ``SPDMX.csv``."""
    csv_path = resolve_track_map_csv(corrected_midi_dir)
    key = dest_rel_for_midi(midi_path, corrected_midi_dir)
    maps = track_maps if track_maps is not None else load_track_maps(csv_path)
    if key not in maps:
        raise FileNotFoundError(
            f"No track map rows for {midi_path} (key {key!r}) in {csv_path}"
        )
    return maps[key]


def _conductor_meta_prefix(midi: mido.MidiFile) -> list:
    """Tempo / meter meta to prepend onto each note-bearing track.

    Prefer messages from track 0 (standard MIDI conductor). Falls back to the
    first matching meta found anywhere so timing is preserved when track 0 is
    missing those events.
    """
    collected: dict[str, object] = {}
    sources: list = list(midi.tracks[:1]) + list(midi.tracks[1:])
    for track in sources:
        abs_time = 0
        for message in track:
            abs_time += message.time
            if not getattr(message, "is_meta", False):
                continue
            if message.type not in _CONDUCTOR_META_TYPES:
                continue
            # Keep the earliest event of each type.
            if message.type not in collected:
                collected[message.type] = message.copy(time=0)
        if len(collected) == len(_CONDUCTOR_META_TYPES):
            break
    # Stable order for readers.
    ordered = []
    for key in ("set_tempo", "time_signature", "key_signature", "smpte_offset"):
        if key in collected:
            ordered.append(collected[key])
    return ordered


def _track_program_and_drum(track) -> tuple[int, bool]:
    program = 0
    is_drum = False
    for message in track:
        if message.type == "program_change":
            program = int(message.program)
        # Any sounding note on GM channel 9 → drum (not the first channel msg).
        if (
            message.type == "note_on"
            and int(getattr(message, "velocity", 0) or 0) > 0
            and getattr(message, "channel", None) == 9
        ):
            is_drum = True
    return program, is_drum


def _track_name(track) -> str | None:
    for message in track:
        if message.type == "track_name":
            return sanitize_track_name(message.name)
    return None


def write_corrected_midi(
    src_mid: str | Path,
    dest_mid: str | Path,
    *,
    program_by_original_track: dict[int, int] | None = None,
    mid_rel: str | None = None,
) -> list[dict]:
    """Write a dense MIDI: note-bearing tracks only, register programs applied.

    Timing meta (tempo / time signature / key) is folded onto each kept track so
    single-track stem exports still have correct wall-clock duration. Empty
    grand-staff / conductor stubs are omitted.

    Songs with no sounding notes are skipped: no dest file is written (and any
    leftover dest is removed), and the returned row list is empty.

    Returns track-map rows (``TRACK_MAP_COLUMNS``). The batch writer stores them
    in a single ``SPDMX.csv``; this function does not write a per-file map.
    """
    src_mid = Path(src_mid)
    dest_mid = Path(dest_mid)
    program_by_original_track = program_by_original_track or {}
    rel = mid_rel if mid_rel is not None else str(src_mid)
    song_id = song_id_from_mid(rel)

    midi = mido.MidiFile(filename=str(src_mid), charset="utf8")
    meta_prefix = _conductor_meta_prefix(midi)
    out = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat, charset="utf8")
    map_rows: list[dict] = []

    dense_idx = 0
    for original_idx, track in enumerate(midi.tracks):
        if note_on_count(track) == 0:
            continue
        program_orig, is_drum = _track_program_and_drum(track)
        program = int(program_by_original_track.get(original_idx, program_orig))
        name = _track_name(track)

        new_track = mido.MidiTrack()
        for meta in meta_prefix:
            new_track.append(meta.copy(time=0))
        for message in track:
            new_track.append(message.copy())
        apply_patch_to_midi_track(
            new_track,
            PatchAssignment(program=program, is_drum=is_drum),
        )
        out.tracks.append(new_track)
        map_rows.append({
            "song_id": song_id,
            "path": spdmx_audio_rel(song_id),
            "mid": spdmx_mid_rel(song_id),
            "track": dense_idx,
            "original_track": original_idx,
            "program": program,
            "is_drum": bool(is_drum),
            "name": name,
        })
        dense_idx += 1

    if not map_rows:
        if dest_mid.is_file():
            dest_mid.unlink()
        return []

    dest_mid.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dest_mid))
    return map_rows


def _corrected_midi_dest_rel(mid_rel: str) -> str:
    rel = str(mid_rel).lstrip("./")
    if rel.startswith("mid/"):
        return rel[len("mid/") :]
    return rel


def _write_one_corrected_midi(
    item: tuple[str, str, dict[int, int], str],
) -> tuple[list[dict], bool]:
    """Worker: ``(mid_rel, dest, programs, pdmx_root)`` → ``(rows, failed)``."""
    from analysis.track_names import mid_path_for_row

    mid_rel, dest, programs, pdmx_root = item
    src = mid_path_for_row(mid_rel, pdmx_root)
    if not src.is_file():
        return [], True
    try:
        rows = write_corrected_midi(
            src,
            dest,
            program_by_original_track=programs,
            mid_rel=mid_rel,
        )
    except Exception:
        return [], True
    return rows, False


def write_corrected_midis_from_register(
    register: pd.DataFrame,
    *,
    pdmx_root: str | Path,
    corrected_midi_dir: str | Path,
    jobs: int = 1,
) -> tuple[int, int]:
    """Write all corrected midis referenced by ``register``. Returns (ok, failed)."""
    pdmx_root = Path(pdmx_root)
    corrected_midi_dir = Path(corrected_midi_dir)
    corrected_midi_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[str, str, dict[int, int], str]] = []
    pdmx_root_s = str(pdmx_root)
    for mid_rel, group in register.groupby("mid", sort=False):
        dest = str(corrected_midi_dir / _corrected_midi_dest_rel(str(mid_rel)))
        programs = {
            int(track): int(program)
            for track, program in zip(
                group["track"].to_numpy(),
                group["program_corrected"].to_numpy(),
            )
        }
        work.append((str(mid_rel), dest, programs, pdmx_root_s))

    ok = 0
    failed = 0
    skipped = 0
    global_rows: list[dict] = []
    from tqdm import tqdm

    n_jobs = max(1, int(jobs))
    desc = f"Writing corrected MIDI (-j {n_jobs})"
    if n_jobs <= 1:
        results = (
            _write_one_corrected_midi(item)
            for item in work
        )
        iterator = tqdm(results, total=len(work), desc=desc, unit="song")
    else:
        chunksize = max(8, min(64, len(work) // (n_jobs * 4) or 8))
        pool = multiprocessing.Pool(processes=n_jobs)
        iterator = tqdm(
            pool.imap_unordered(_write_one_corrected_midi, work, chunksize=chunksize),
            total=len(work),
            desc=desc,
            unit="song",
        )

    try:
        for rows, song_failed in iterator:
            if song_failed:
                failed += 1
                continue
            if not rows:
                skipped += 1
                continue
            global_rows.extend(rows)
            ok += 1
    finally:
        if n_jobs > 1:
            pool.close()
            pool.join()

    if global_rows:
        out_csv = track_map_csv_path(corrected_midi_dir)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        legacy = out_csv.with_name(LEGACY_TRACK_MAP_FILE_NAME)
        if legacy.is_file() and legacy.resolve() != out_csv.resolve():
            legacy.unlink()
        pd.DataFrame(global_rows, columns=TRACK_MAP_COLUMNS).to_csv(out_csv, index=False)
        clear_track_map_cache()
        from synthesis.spdmx_release import maybe_write_spdmx_release_docs

        maybe_write_spdmx_release_docs(out_csv.parent)
    if skipped:
        print(
            f"Skipped {skipped} songs with no sounding notes (no MIDI / SPDMX.csv rows)",
            flush=True,
        )
    return ok, failed


def resolve_corrected_midi_path(
    pdmx_mid: str | Path,
    *,
    pdmx_root: str | Path,
    corrected_midi_dir: str | Path,
) -> Path:
    """Map a PDMX absolute/relative mid path to its corrected copy."""
    pdmx_mid = Path(pdmx_mid)
    pdmx_root = Path(pdmx_root)
    corrected_midi_dir = Path(corrected_midi_dir)
    try:
        rel = pdmx_mid.resolve().relative_to(pdmx_root.resolve())
    except ValueError:
        # Fall back: use filename layout mid/a/b/x.mid → a/b/x.mid
        parts = pdmx_mid.parts
        if "mid" in parts:
            idx = parts.index("mid")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = Path(pdmx_mid.name)
    rel_s = str(rel).lstrip("./")
    if rel_s.startswith("mid/"):
        rel_s = rel_s[len("mid/"):]
    primary = corrected_midi_dir / rel_s
    if primary.is_file():
        return primary
    if (
        corrected_midi_dir.name == SPDMX_MID_DIR_NAME
        and corrected_midi_dir.parent.name == SPDMX_DATASET_DIR_NAME
    ):
        legacy = (
            corrected_midi_dir.parent.parent
            / DEV_DIR_NAME
            / MID_CORRECTED_DIR_NAME
            / rel_s
        )
        if legacy.is_file():
            return legacy
    return primary
