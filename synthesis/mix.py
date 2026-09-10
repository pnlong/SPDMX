"""Peak-normalize stems so they remain linearly summable.

Synthesis / realify write raw stems. Run this afterward to:

1. Loudness-normalize stems (−23 LUFS)
2. Apply MIDI velocity dynamics (track_max / song_max)
3. Sum in memory and compute the Slakh-style anti-clip peak gain
4. Write stems with that same peak gain (overwrites by default; use ``--no-overwrite``)
5. Optionally write ``mixture.*`` with ``--write-mixture`` (otherwise mix = sum(stems))

See ``synthesis/MIXING.md`` for the full pipeline description.

Example::

    uv run python -m synthesis.mix --render-mode basic -j 8
    uv run python -m synthesis.mix --stems-dir /path/to/ablation --no-overwrite --write-mixture -j 8
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shared.config import (
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    PDMX_FILEPATH,
    RENDER_MODE_BASIC,
    RENDER_MODES,
    STEMS_FILE_NAME,
)
from synthesis.paths import remap_path_prefix
from synthesis.audio import (
    mixture_path,
    normalize_stems_in_song_dir,
    stem_path,
    synthesis_audio_format,
)
from synthesis.cli_common import add_audio_format_arg
from synthesis.dense_midi import default_corrected_midi_dir
from synthesis.paths import (
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
    resolve_output_song_dir,
)
from synthesis.velocity import (
    pdmx_mid_from_song_dir,
    velocity_scales_for_midi,
)


def normalize_song_task(task: dict) -> str | None:
    scales = task.get("velocity_scales")
    peak_gain = normalize_stems_in_song_dir(
        Path(task["song_dir"]),
        task["tracks"],
        task["audio_format"],
        dest_song_dir=Path(task["out_song_dir"]),
        write_mixture=bool(task.get("write_mixture", False)),
        velocity_scales=scales,
    )
    return task["out_song_dir"] if peak_gain is not None else None


# Back-compat alias
write_mixture_task = normalize_song_task


def mix_output_ready(
    dest_song_dir: Path,
    tracks: list[int],
    audio_format: str,
    *,
    write_mixture: bool = False,
) -> bool:
    """True when dest already has non-empty stems for every track (resume skip).

    Uses size>0 rather than ``stem_is_valid`` so scanning hundreds of thousands of
    songs on NFS stays a cheap ``stat``, not an ``sf.info`` per file.
    """
    if not dest_song_dir.is_dir():
        return False
    for track in tracks:
        path = stem_path(dest_song_dir, track, audio_format)
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    if write_mixture:
        mix = mixture_path(dest_song_dir, audio_format)
        try:
            if not mix.is_file() or mix.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _scales_from_stems_group(group: pd.DataFrame) -> dict[int, float] | None:
    """Prefer persisted ``velocity_scale`` column when present and complete."""
    if "velocity_scale" not in group.columns:
        return None
    scales: dict[int, float] = {}
    for _, row in group.iterrows():
        track = int(row["track"])
        value = row["velocity_scale"]
        if pd.isna(value):
            return None
        scales[track] = float(value)
    return scales if scales else None


def resolve_song_midi(
    song_dir: Path,
    *,
    pdmx_root: Path,
    output_dir: str = OUTPUT_DIR,
) -> Path:
    """Resolve the dense corrected MIDI whose track indices match rendered stems."""
    from analysis.corrected_midi import resolve_corrected_midi_path

    pdmx_mid = pdmx_mid_from_song_dir(song_dir, pdmx_root)
    corrected = resolve_corrected_midi_path(
        pdmx_mid,
        pdmx_root=pdmx_root,
        corrected_midi_dir=default_corrected_midi_dir(output_dir),
    )
    if not corrected.is_file():
        raise FileNotFoundError(
            f"Corrected MIDI missing for {song_dir}: {corrected}\n"
            "Run: uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
        )
    return corrected


def velocity_scales_for_song(
    song_dir: Path,
    group: pd.DataFrame,
    *,
    pdmx_root: Path,
    output_dir: str = OUTPUT_DIR,
    use_velocity_dynamics: bool = True,
) -> dict[int, float] | None:
    """Return per-track velocity scales, or None when dynamics are disabled."""
    if not use_velocity_dynamics:
        return None
    persisted = _scales_from_stems_group(group)
    if persisted is not None:
        return persisted
    midi_path = resolve_song_midi(song_dir, pdmx_root=pdmx_root, output_dir=output_dir)
    return velocity_scales_for_midi(midi_path)


def build_mixture_tasks(
    stems: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    audio_format: str,
    *,
    write_mixture: bool = False,
    pdmx_root: Path | None = None,
    spdmx_output_dir: str = OUTPUT_DIR,
    use_velocity_dynamics: bool = True,
    dest_song_dir_fn=None,
    reset: bool = False,
) -> tuple[list[dict], int]:
    """Build per-song mix tasks.

    Returns ``(tasks, n_skipped)``. When ``reset`` is False and the destination
    tree differs from the source, songs whose dest stems are already present are
    skipped (resume). In-place overwrite (dest == source) never skips.
    """
    tasks = []
    skipped = 0
    root = Path(pdmx_root) if pdmx_root is not None else Path(PDMX_FILEPATH).parent
    for song_path, group in stems.groupby("path"):
        src_song_dir = Path(song_path)
        if dest_song_dir_fn is not None:
            out_song_dir = Path(dest_song_dir_fn(str(src_song_dir)))
        else:
            out_song_dir = resolve_output_song_dir(src_song_dir, source_dir, output_dir)
        tracks = sorted(int(t) for t in group["track"])
        if (
            not reset
            and out_song_dir.resolve() != src_song_dir.resolve()
            and mix_output_ready(
                out_song_dir,
                tracks,
                audio_format,
                write_mixture=write_mixture,
            )
        ):
            skipped += 1
            continue
        try:
            scales = velocity_scales_for_song(
                src_song_dir,
                group,
                pdmx_root=root,
                output_dir=spdmx_output_dir,
                use_velocity_dynamics=use_velocity_dynamics,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Cannot resolve MIDI velocity dynamics for {src_song_dir}: {exc}\n"
                "Pass --dataset to the PDMX.csv path, or --no-velocity-dynamics to skip."
            ) from exc
        tasks.append({
            "song_dir": str(src_song_dir),
            "out_song_dir": str(out_song_dir),
            "tracks": tracks,
            "audio_format": audio_format,
            "write_mixture": write_mixture,
            "velocity_scales": scales,
        })
    return tasks, skipped


def _shutdown_pool(pool) -> None:
    pool.close()
    pool.join()


def copy_metadata_tables(source_dir: Path, output_dir: Path) -> None:
    """Copy data/stems CSVs into ``output_dir``, remapping absolute song paths."""
    if source_dir.resolve() == output_dir.resolve():
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in (f"{STEMS_FILE_NAME}.csv", f"{DATA_DIR_NAME}.csv", "ddsp_routing.csv"):
        src = source_dir / name
        if not src.exists():
            continue
        table = pd.read_csv(src)
        if "path" in table.columns:
            table["path"] = table["path"].map(
                lambda p: remap_path_prefix(str(p), source_dir, output_dir)
            )
        table.to_csv(output_dir / name, index=False)


def default_dest_dir(stems_dir: Path) -> Path:
    """Sibling directory for non-overwrite writes: ``{name}_summable``."""
    return stems_dir.parent / f"{stems_dir.name}_summable"


def confirm_overwrite(stems_dir: Path) -> bool:
    prompt = (
        f"This will OVERWRITE existing stems in:\n  {stems_dir}\n"
        "Are you sure you want to overwrite? [y/N] "
    )
    try:
        reply = input(prompt).strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def normalize_stems_for_dataset(
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    jobs: int = 1,
    *,
    write_mixture: bool = False,
    pdmx_root: Path | None = None,
    spdmx_output_dir: str = OUTPUT_DIR,
    use_velocity_dynamics: bool = True,
    dest_song_dir_fn=None,
    reset: bool = False,
):
    """Peak-normalize stems so they remain linearly summable.

    Loads stems from ``source_dir`` (via stems.csv paths) and writes to the
    mirrored tree under ``output_dir``. When ``source_dir == output_dir``, stems
    are overwritten in place unless ``dest_song_dir_fn`` remaps each song path
    (hybrid ``raw`` → ``audio``).

    When the destination differs from the source, songs whose dest stems are
    already present are skipped unless ``reset`` is True.
    """
    stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        stems_csv = output_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        return

    if (
        dest_song_dir_fn is None
        and source_dir.resolve() != output_dir.resolve()
    ):
        copy_metadata_tables(source_dir, output_dir)

    stems = pd.read_csv(stems_csv)
    tasks, n_skipped = build_mixture_tasks(
        stems,
        source_dir,
        output_dir,
        audio_format,
        write_mixture=write_mixture,
        pdmx_root=pdmx_root,
        spdmx_output_dir=spdmx_output_dir,
        use_velocity_dynamics=use_velocity_dynamics,
        dest_song_dir_fn=dest_song_dir_fn,
        reset=reset,
    )
    n_total = n_skipped + len(tasks)
    if n_skipped:
        print(
            f"Resume: skipping {n_skipped}/{n_total} song(s) with mix output already on disk; "
            f"{len(tasks)} remaining.",
            flush=True,
        )
    if not tasks:
        if n_skipped:
            print("Nothing left to mix.", flush=True)
        return

    n_workers = min(max(jobs, 1), len(tasks))
    parts = ["Normalizing stems"]
    if use_velocity_dynamics:
        parts.append("+ velocity dynamics")
    if write_mixture:
        parts.append("+ writing mixtures")
    action = " ".join(parts)
    desc = action if n_workers == 1 else f"{action} ({n_workers} workers)"
    if n_workers == 1:
        for task in tqdm(tasks, desc=desc, unit="song"):
            normalize_song_task(task)
        return

    pool = multiprocessing.Pool(processes=n_workers)
    try:
        for _ in tqdm(
            pool.imap(normalize_song_task, tasks, chunksize=1),
            total=len(tasks),
            desc=desc,
            unit="song",
        ):
            pass
    finally:
        _shutdown_pool(pool)


# Back-compat alias
write_mixtures_for_dataset = normalize_stems_for_dataset


def _mixed_stem_path_ok(path_str: str) -> bool:
    """Picklable worker: full FLAC decode of one mixed stem path."""
    from synthesis.audio import flac_fully_decodes

    return flac_fully_decodes(Path(path_str))


def _mixed_stem_path_ok_indexed(item: tuple[int, str]) -> tuple[int, bool]:
    idx, path_str = item
    return idx, _mixed_stem_path_ok(path_str)


def _song_id_from_stem_path(path: str) -> str | None:
    """Extract ``song_id`` from a raw/audio song directory path."""
    text = str(path).replace("\\", "/")
    for marker in ("/audio/", "/raw/"):
        if marker in text:
            return text.split(marker, 1)[1].rstrip("/")
    for marker in ("./audio/", "./raw/"):
        if text.startswith(marker):
            return text[len(marker) :].rstrip("/")
    return None


def mixed_stem_paths_from_tables(
    tables_dir: str | Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    *,
    media_dir: str | Path | None = None,
) -> list[str]:
    """Absolute mixed-stem paths under ``audio/`` implied by ``stems.csv``.

    When ``media_dir`` is set (production ``SPDMX_dev/``), paths are resolved
    under that tree. Bookkeeping ``dev/final/stems.csv`` often still has absolute
    ``…/SPDMX/raw/…`` paths from an older layout; those must not be trusted as
    on-disk locations.
    """
    from shared.config import SPDMX_AUDIO_DIR_NAME, SPDMX_FILE_NAME
    from synthesis.paths import raw_path_to_audio

    media = Path(media_dir) if media_dir is not None else None

    # Prefer the flat production track map when present (has song_id).
    if media is not None:
        media_csv = media / f"{SPDMX_FILE_NAME}.csv"
        if media_csv.is_file() and media_csv.stat().st_size > 0:
            table = pd.read_csv(media_csv, low_memory=False)
            if {"song_id", "track"}.issubset(table.columns):
                return [
                    str(
                        media
                        / SPDMX_AUDIO_DIR_NAME
                        / str(song_id)
                        / f"{int(track)}.{audio_format}"
                    )
                    for song_id, track in zip(
                        table["song_id"].tolist(),
                        table["track"].tolist(),
                        strict=False,
                    )
                ]

    root = Path(tables_dir)
    stems_csv = root / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file() or stems_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Missing stems table for mix verify: {stems_csv}")
    stems = pd.read_csv(stems_csv, usecols=["path", "track"], low_memory=False)
    if stems.empty:
        return []
    out: list[str] = []
    for path, track in zip(stems["path"].tolist(), stems["track"].tolist(), strict=False):
        song = str(path)
        if media is not None:
            song_id = _song_id_from_stem_path(song)
            if song_id is None:
                raise ValueError(
                    f"Cannot map stems path to song_id under media_dir={media}: {song}"
                )
            song = str(media / SPDMX_AUDIO_DIR_NAME / song_id)
        else:
            try:
                if "/raw/" in song.replace("\\", "/") or song.replace("\\", "/").startswith(
                    "./raw/"
                ):
                    song = raw_path_to_audio(song)
            except ValueError:
                pass
        out.append(str(stem_path(Path(song), int(track), audio_format)))
    return out


def song_mix_paths_from_media(
    media_dir: str | Path,
    tables_dir: str | Path | None = None,
) -> list[str]:
    """Absolute ``mix/<song_id>.flac`` paths for songs in the track map / stems table."""
    from shared.config import SPDMX_FILE_NAME, SPDMX_MIX_DIR_NAME

    media = Path(media_dir)
    mix_root = media / SPDMX_MIX_DIR_NAME
    song_ids: list[str] = []

    media_csv = media / f"{SPDMX_FILE_NAME}.csv"
    if media_csv.is_file():
        table = pd.read_csv(media_csv, low_memory=False)
        if "song_id" in table.columns:
            song_ids = sorted(table["song_id"].astype(str).unique())
    if not song_ids and tables_dir is not None:
        stems_csv = Path(tables_dir) / f"{STEMS_FILE_NAME}.csv"
        if stems_csv.is_file():
            stems = pd.read_csv(stems_csv, usecols=["path"], low_memory=False)
            ids: set[str] = set()
            for path in stems["path"].astype(str):
                song_id = _song_id_from_stem_path(path)
                if song_id is not None:
                    ids.add(song_id)
            song_ids = sorted(ids)
    return [str(mix_root / f"{song_id}.flac") for song_id in song_ids]


def verify_mixed_stems_on_disk(
    tables_dir: str | Path,
    *,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    jobs: int = 1,
    limit: int = 25,
    media_dir: str | Path | None = None,
) -> None:
    """Require mixed ``audio/`` stems (and optional ``mix/`` song mixes) to FLAC-decode.

    Raises ``RuntimeError`` listing up to ``limit`` failures. Intended as
    ``--only-pass verify_mix`` after ``mix`` and ``song_mix``.
    """
    paths = mixed_stem_paths_from_tables(
        tables_dir, audio_format=audio_format, media_dir=media_dir,
    )
    if media_dir is not None:
        paths.extend(song_mix_paths_from_media(media_dir, tables_dir=tables_dir))
    if not paths:
        raise RuntimeError(f"No stems/mixes to verify under {tables_dir}")

    n_jobs = max(1, int(jobs))
    label = f"verify_mix decode (-j {n_jobs})" if n_jobs > 1 else "verify_mix decode"
    bad: list[str] = []
    if n_jobs <= 1 or len(paths) <= 1:
        for path_str in tqdm(paths, total=len(paths), desc=label, unit="file"):
            if not _mixed_stem_path_ok(path_str):
                bad.append(path_str)
                if len(bad) >= limit:
                    break
    else:
        chunksize = max(4, min(32, len(paths) // (n_jobs * 8) or 4))
        pbar = tqdm(total=len(paths), desc=label, unit="file", miniters=1, smoothing=0.05)
        try:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for idx, ok in pool.imap_unordered(
                    _mixed_stem_path_ok_indexed,
                    enumerate(paths),
                    chunksize=chunksize,
                ):
                    pbar.update(1)
                    if not ok:
                        bad.append(paths[idx])
                        if len(bad) >= limit:
                            pool.terminate()
                            break
        finally:
            pbar.close()
        bad.sort()

    if bad:
        n_bad = len(bad)
        if n_bad >= limit and len(paths) > limit:
            extra = f" (showing first {limit}; scan stopped early)"
        else:
            extra = ""
        lines = "\n".join(f"  {p}" for p in bad[:limit])
        raise RuntimeError(
            f"Mixed stems/mixes failed FLAC decode or are missing ({n_bad}{extra}):\n{lines}\n"
            "Re-run: uv run python -m synthesis.final --only-pass mix -j 8\n"
            "Then:    uv run python -m synthesis.final --only-pass song_mix -j 8\n"
            "Then:    uv run python -m synthesis.final --only-pass verify_mix -j 8"
        )
    print(f"verify_mix ok: {len(paths)} file(s) fully decoded.", flush=True)


def resolve_stems_dir(
    *,
    stems_dir: str | None = None,
    output_dir: str = OUTPUT_DIR,
    render_mode: str = RENDER_MODE_BASIC,
    realify: bool = False,
    full: bool = False,
) -> Path:
    """Resolve the stem tree to normalize, mirroring synthesize output layout."""
    if stems_dir is not None:
        return Path(stems_dir)
    if full:
        return Path(
            full_stems_realify_dir(output_dir) if realify else full_stems_dir(output_dir)
        )
    return Path(
        ablation_realify_dir(output_dir, render_mode)
        if realify
        else ablation_raw_dir(output_dir, render_mode)
    )


def mix_command(
    stems_dir: str | Path,
    *,
    jobs: int | None = None,
    flac: bool = False,
) -> str:
    """CLI string to print after a stems-only synthesis/realify run."""
    n_jobs = jobs if jobs is not None else max(1, int(multiprocessing.cpu_count() / 4))
    cmd = f"uv run python -m synthesis.mix --stems-dir {stems_dir} -j {n_jobs}"
    if flac:
        cmd += " --flac"
    return cmd


def print_mix_hint(
    stems_dir: str | Path,
    *,
    jobs: int | None = None,
    flac: bool = False,
) -> None:
    print(
        "\nStems written raw (no LUFS). "
        "To apply LUFS + velocity dynamics + peak-normalize so they remain linearly "
        "summable (mix = sum of stems by default), run:",
        flush=True,
    )
    print(f"  {mix_command(stems_dir, jobs=jobs, flac=flac)}", flush=True)
    print(
        "  # Preview without overwriting + write mixture files:\n"
        f"  {mix_command(stems_dir, jobs=jobs, flac=flac)} --no-overwrite --write-mixture",
        flush=True,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Normalize stems for summability: LUFS + MIDI velocity dynamics + "
            "uniform anti-clip peak gain. See synthesis/MIXING.md."
        ),
    )
    parser.add_argument(
        "--stems-dir",
        default=None,
        type=str,
        help="Stem tree to normalize (overrides --render-mode / --realify / --full).",
    )
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument(
        "-df",
        "--dataset",
        "--dataset_filepath",
        dest="dataset_filepath",
        default=PDMX_FILEPATH,
        type=str,
        help="PDMX.csv path; parent dir is used to resolve mid/… MIDI files.",
    )
    parser.add_argument(
        "--render-mode",
        default=RENDER_MODE_BASIC,
        choices=list(RENDER_MODES),
        help="Ablation/full stem tree to resolve when --stems-dir is omitted.",
    )
    parser.add_argument("--realify", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Normalize full PDMX stems tree instead of an ablation.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help=(
            "Do not overwrite source stems; write normalized stems to --dest-dir "
            "(default: sibling {name}_summable)."
        ),
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        type=str,
        help="Destination tree when using --no-overwrite (default: {stems-dir}_summable).",
    )
    parser.add_argument(
        "--write-mixture",
        action="store_true",
        help="Also write mixture.* beside the (normalized) stems.",
    )
    parser.add_argument(
        "--no-velocity-dynamics",
        action="store_true",
        help="Skip MIDI velocity scaling (LUFS + peak only).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the overwrite confirmation prompt.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Re-mix even when destination stems already exist "
            "(default resumes and skips complete dest songs)."
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU workers for stem normalization.",
    )
    add_audio_format_arg(parser)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    stems_dir = resolve_stems_dir(
        stems_dir=opts.stems_dir,
        output_dir=opts.output_dir,
        render_mode=opts.render_mode,
        realify=opts.realify,
        full=opts.full,
    )
    if not stems_dir.is_dir():
        raise SystemExit(f"Stem directory not found: {stems_dir}")

    overwrite = not bool(opts.no_overwrite)
    if overwrite:
        dest_dir = stems_dir
        if not opts.yes and not confirm_overwrite(stems_dir):
            raise SystemExit("Aborted (stems not overwritten).")
    else:
        dest_dir = Path(opts.dest_dir) if opts.dest_dir else default_dest_dir(stems_dir)
        if dest_dir.resolve() == stems_dir.resolve():
            raise SystemExit(
                "--no-overwrite requires a different --dest-dir than the source stems tree."
            )

    audio_format = synthesis_audio_format(opts.flac)
    use_velocity = not bool(opts.no_velocity_dynamics)
    mode = "overwrite in place" if overwrite else f"write to {dest_dir}"
    notes = []
    if use_velocity:
        notes.append("velocity dynamics")
    if opts.write_mixture:
        notes.append("writing mixtures")
    note_str = f", {', '.join(notes)}" if notes else ""
    print(
        f"Normalizing stems for summability ({mode}{note_str}; "
        f"{audio_format}, -j {opts.jobs})",
        flush=True,
    )
    normalize_stems_for_dataset(
        stems_dir,
        dest_dir,
        audio_format=audio_format,
        jobs=opts.jobs,
        write_mixture=bool(opts.write_mixture),
        pdmx_root=Path(opts.dataset_filepath).parent,
        spdmx_output_dir=opts.output_dir,
        use_velocity_dynamics=use_velocity,
        reset=bool(opts.reset),
    )
    print(f"Done. Output: {dest_dir}", flush=True)


if __name__ == "__main__":
    main()
