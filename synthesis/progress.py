"""Report pre-realify hybrid render progress from CSVs + on-disk stems.

Example::

    uv run python -m synthesis.progress
    uv run python -m synthesis.progress -j 16
    uv run python -m synthesis.progress --no-check-disk
    uv run python -m synthesis.progress -y
    uv run python -m synthesis.progress -o /deepfreeze/share/SPDMX --count-flac
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from tqdm import tqdm

from shared.config import FLAC_AUDIO_FORMAT, NA_STRING, OUTPUT_DIR
from synthesis.audio import stem_is_valid, stem_path
from synthesis.final import hybrid_dirs
from synthesis.pass_tables import (
    _song_id_from_audio_dir,
    pass_recipe_csv,
    pass_routing_csv,
    pass_stems_csv,
)
from synthesis.paths import MIDI_INDEX_FILE_NAME, PASS_TRACK_COLUMNS, spdmx_raw_dir
from synthesis.recipe import DEFAULT_RECIPE_PATH, load_recipe
from synthesis.synthesize import _render_passes_for_recipe

MISSING_STEMS_REPORT_NAME = "missing_claimed_stems.txt"


@dataclass(frozen=True)
class MissingClaimedStem:
    pass_name: str
    song_path: str
    track: int
    flac_path: str

    @property
    def song_id(self) -> str:
        return _song_id_from_audio_dir(self.song_path)


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return 100.0 * float(done) / float(total)


def _recipe_done_by_song_id(recipe_csv: Path) -> pd.Series:
    """song_id → row count from a pass ``stem_recipe`` CSV."""
    if not recipe_csv.is_file() or recipe_csv.stat().st_size == 0:
        return pd.Series(dtype=int)
    df = pd.read_csv(recipe_csv, usecols=["path"])
    if df.empty:
        return pd.Series(dtype=int)
    sids = df["path"].astype(str).map(_song_id_from_audio_dir)
    return sids.value_counts()


def count_pass_progress_fast(tables_dir: Path, recipe) -> list[dict]:
    """Vectorized assigned / recipe-done / remaining per render pass."""
    index_path = tables_dir / MIDI_INDEX_FILE_NAME
    index = pd.read_csv(index_path)
    if index.empty or "song_id" not in index.columns:
        return []

    rows: list[dict] = []
    for pass_name in _render_passes_for_recipe(recipe):
        col = PASS_TRACK_COLUMNS.get(pass_name)
        if col is None or col not in index.columns:
            continue
        assigned_col = index[col].fillna(0).astype(int)
        mask = assigned_col > 0
        assigned = int(assigned_col.sum())
        if assigned == 0:
            rows.append({
                "pass": pass_name,
                "assigned": 0,
                "recipe_done": 0,
                "remaining": 0,
                "songs_left": 0,
            })
            continue
        done_counts = _recipe_done_by_song_id(pass_recipe_csv(tables_dir, pass_name))
        songs = index.loc[mask, ["song_id"]].copy()
        songs["assigned"] = assigned_col.loc[mask].to_numpy()
        songs["song_id"] = songs["song_id"].astype(str)
        songs["done"] = songs["song_id"].map(done_counts).fillna(0).astype(int)
        songs["remaining"] = (songs["assigned"] - songs["done"]).clip(lower=0)
        remaining = int(songs["remaining"].sum())
        songs_left = int((songs["remaining"] > 0).sum())
        rows.append({
            "pass": pass_name,
            "assigned": assigned,
            "recipe_done": assigned - remaining,
            "remaining": remaining,
            "songs_left": songs_left,
        })
    return rows


def _claimed_stem_rows(tables_dir: Path, pass_name: str) -> pd.DataFrame:
    """Prefer ``stems.<pass>.csv``; fall back to ``stem_recipe.<pass>.csv``."""
    for path in (pass_stems_csv(tables_dir, pass_name), pass_recipe_csv(tables_dir, pass_name)):
        if path.is_file() and path.stat().st_size > 0:
            df = pd.read_csv(path, usecols=lambda c: c in ("path", "track"))
            if not df.empty and "path" in df.columns and "track" in df.columns:
                return df.drop_duplicates(["path", "track"])
    return pd.DataFrame(columns=["path", "track"])


def _stem_exists(path_str: str) -> bool:
    """True when the path is a non-empty file (no FLAC decode)."""
    try:
        return os.path.isfile(path_str) and os.path.getsize(path_str) > 0
    except OSError:
        return False


def _stem_strict_ok(path_str: str) -> bool:
    """True when ``stem_is_valid`` passes (opens FLAC header via soundfile)."""
    return stem_is_valid(Path(path_str))


def check_claimed_stems_on_disk(
    tables_dir: Path,
    pass_name: str,
    *,
    audio_format: str = FLAC_AUDIO_FORMAT,
    strict: bool = True,
    jobs: int = 1,
    collect_missing: bool = False,
) -> tuple[int, int, list[MissingClaimedStem]]:
    """Return ``(on_disk, claimed, missing)`` for one pass's stem table rows.

    ``strict=False``: exists + size &gt; 0.
    ``strict=True``: ``stem_is_valid`` (opens each FLAC).
    """
    frames = _claimed_stem_rows(tables_dir, pass_name)
    claimed = len(frames)
    if claimed == 0:
        return 0, 0, []
    song_paths = frames["path"].tolist()
    tracks = frames["track"].tolist()
    path_strs = [
        str(stem_path(Path(str(p)), int(t), audio_format))
        for p, t in zip(song_paths, tracks, strict=True)
    ]
    worker = _stem_strict_ok if strict else _stem_exists
    label = f"{pass_name} {'validate' if strict else 'exists'}"
    n_jobs = max(1, int(jobs))
    chunksize = 1 if strict else max(4, min(32, claimed // (n_jobs * 8) or 4))
    pbar = tqdm(
        total=claimed,
        desc=f"{label} (-j {n_jobs})" if n_jobs > 1 else label,
        unit="stem",
        miniters=1,
        smoothing=0.05,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    on_disk = 0
    missing: list[MissingClaimedStem] = []
    try:
        if n_jobs <= 1 or claimed <= 1:
            for song_path, track, path_str in zip(
                song_paths, tracks, path_strs, strict=True,
            ):
                if worker(path_str):
                    on_disk += 1
                elif collect_missing:
                    missing.append(MissingClaimedStem(
                        pass_name=pass_name,
                        song_path=str(song_path),
                        track=int(track),
                        flac_path=path_str,
                    ))
                pbar.update(1)
        else:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                for idx, ok in pool.imap_unordered(
                    _check_indexed_stem,
                    [(i, path_strs[i], strict) for i in range(len(path_strs))],
                    chunksize=chunksize,
                ):
                    if ok:
                        on_disk += 1
                    elif collect_missing:
                        missing.append(MissingClaimedStem(
                            pass_name=pass_name,
                            song_path=str(song_paths[idx]),
                            track=int(tracks[idx]),
                            flac_path=path_strs[idx],
                        ))
                    pbar.update(1)
    finally:
        pbar.close()
    missing.sort(key=lambda m: (m.pass_name, m.song_path, m.track))
    return on_disk, claimed, missing


def _check_indexed_stem(args: tuple[int, str, bool]) -> tuple[int, bool]:
    _idx, path_str, strict = args
    ok = _stem_strict_ok(path_str) if strict else _stem_exists(path_str)
    return _idx, ok


def count_claimed_on_disk(
    tables_dir: Path,
    pass_name: str,
    *,
    audio_format: str = FLAC_AUDIO_FORMAT,
    strict: bool = True,
    jobs: int = 1,
) -> tuple[int, int]:
    """Return ``(on_disk, claimed)`` for one pass's stem table rows."""
    on_disk, claimed, _missing = check_claimed_stems_on_disk(
        tables_dir,
        pass_name,
        audio_format=audio_format,
        strict=strict,
        jobs=jobs,
        collect_missing=False,
    )
    return on_disk, claimed


def write_missing_stems_report(
    tables_dir: Path,
    missing: list[MissingClaimedStem],
) -> Path:
    """Write tab-separated missing stems under ``tables_dir``."""
    out = Path(tables_dir) / MISSING_STEMS_REPORT_NAME
    lines = ["pass\tsong_id\tsong_path\ttrack\tflac_path"]
    for stem in missing:
        lines.append(
            f"{stem.pass_name}\t{stem.song_id}\t{stem.song_path}\t"
            f"{stem.track}\t{stem.flac_path}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _sed_delete_pattern(song_path: str, track: int) -> str:
    return re.escape(f"{song_path},{track},")


def format_remove_missing_commands(
    tables_dir: Path,
    missing: list[MissingClaimedStem],
) -> str:
    """Shell commands to drop invalid claimed stems for manual re-render."""
    if not missing:
        return ""
    tables = Path(tables_dir)
    lines = [
        "# Remove invalid claimed stems so synthesis can re-render them.",
        f"TABLES={tables}",
        "",
    ]
    flac_paths = sorted({m.flac_path for m in missing})
    lines.append("# Delete invalid or placeholder FLAC files")
    for flac_path in flac_paths:
        lines.append(f'rm -v "{flac_path}"')
    lines.append("")

    by_pass: dict[str, list[MissingClaimedStem]] = defaultdict(list)
    for stem in missing:
        by_pass[stem.pass_name].append(stem)

    lines.append("# Drop (path, track) rows from pass CSV shards")
    for pass_name in sorted(by_pass):
        sed_exprs: list[str] = []
        for stem in by_pass[pass_name]:
            sed_exprs.append(
                f"-e '/{_sed_delete_pattern(stem.song_path, stem.track)}/d'",
            )
        for csv_name in (
            f"stems.{pass_name}.csv",
            f"stem_recipe.{pass_name}.csv",
            f"ddsp_routing.{pass_name}.csv",
        ):
            csv_path = tables / csv_name
            if not csv_path.is_file():
                continue
            lines.append(
                f"sed -i {' '.join(sed_exprs)} \"$TABLES/{csv_name}\"",
            )
    return "\n".join(lines) + "\n"


def remove_missing_claimed_stems(
    tables_dir: Path,
    missing: list[MissingClaimedStem],
) -> tuple[int, int]:
    """Remove invalid FLACs and CSV rows. Returns ``(flacs_removed, csv_rows_removed)``."""
    if not missing:
        return 0, 0
    root = Path(tables_dir)
    flacs_removed = 0
    for flac_path in {m.flac_path for m in missing}:
        path = Path(flac_path)
        if path.is_file():
            path.unlink()
            flacs_removed += 1

    keys_by_pass: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for stem in missing:
        keys_by_pass[stem.pass_name].add((stem.song_path, stem.track))

    csv_rows_removed = 0
    for pass_name, keys in keys_by_pass.items():
        for csv_fn in (pass_stems_csv, pass_recipe_csv, pass_routing_csv):
            csv_path = csv_fn(root, pass_name)
            if not csv_path.is_file() or csv_path.stat().st_size == 0:
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "path" not in df.columns or "track" not in df.columns:
                continue
            before = len(df)
            keep = ~df.apply(
                lambda row, keys=keys: (str(row["path"]), int(row["track"])) in keys,
                axis=1,
            )
            df = df.loc[keep]
            csv_rows_removed += before - len(df)
            df.to_csv(csv_path, index=False, na_rep=NA_STRING)
    return flacs_removed, csv_rows_removed


def _prompt_remove_missing(missing: list[MissingClaimedStem], *, auto_yes: bool) -> bool:
    if not missing:
        return False
    if auto_yes:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(
            f"Remove {len(missing)} missing claimed stem(s) from CSVs and disk? [y/N] ",
        ).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def handle_missing_claimed_stems(
    tables_dir: Path,
    missing: list[MissingClaimedStem],
    *,
    auto_yes: bool = False,
) -> None:
    """Write report, optionally remove stems, otherwise print manual commands."""
    if not missing:
        return
    report_path = write_missing_stems_report(tables_dir, missing)
    print(
        f"\nWrote {len(missing)} missing claimed stem(s) to {report_path}",
        flush=True,
    )
    by_pass: dict[str, int] = defaultdict(int)
    for stem in missing:
        by_pass[stem.pass_name] += 1
    for pass_name in sorted(by_pass):
        print(f"  {pass_name}: {by_pass[pass_name]}", flush=True)
    for stem in missing[:10]:
        print(
            f"    {stem.pass_name} track={stem.track} {stem.song_id} → {stem.flac_path}",
            flush=True,
        )
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more (see {report_path})", flush=True)

    if _prompt_remove_missing(missing, auto_yes=auto_yes):
        flacs, rows = remove_missing_claimed_stems(tables_dir, missing)
        print(
            f"Removed {flacs} invalid FLAC file(s) and {rows} CSV row(s).",
            flush=True,
        )
        return

    print("\nManual cleanup commands:", flush=True)
    print(format_remove_missing_commands(tables_dir, missing), end="", flush=True)


def count_raw_flac_files(raw_root: Path) -> int:
    """Count ``*.flac`` files under ``SPDMX/raw`` (can be slow on NFS)."""
    if not raw_root.is_dir():
        return 0
    n = 0
    for dirpath, _dirnames, filenames in os.walk(raw_root):
        for name in filenames:
            if name.endswith(".flac"):
                n += 1
    return n


def report_pre_realify_progress(
    *,
    output_dir: str,
    recipe_path: str | Path = DEFAULT_RECIPE_PATH,
    check_disk: bool = True,
    strict_disk: bool = True,
    count_flac: bool = False,
    audio_format: str = FLAC_AUDIO_FORMAT,
    jobs: int = 1,
    auto_remove_missing: bool = False,
) -> list[dict]:
    """Print and return per-pass progress for fluidsynth / DDSP (pre-realify)."""
    recipe = load_recipe(recipe_path)
    tables_dir, media_dir = hybrid_dirs(
        SimpleNamespace(full=True, output_dir=output_dir),
    )
    tables = Path(tables_dir)
    raw_root = Path(media_dir) / Path(spdmx_raw_dir(output_dir)).name

    print(f"Output:  {output_dir}", flush=True)
    print(f"Tables:  {tables}", flush=True)
    print(f"Raw:     {raw_root}", flush=True)
    print(f"Recipe:  {recipe_path}", flush=True)

    index_path = tables / MIDI_INDEX_FILE_NAME
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path} (run --only-pass layout first).")

    print("Counting from midi_index + stem_recipe CSVs …", flush=True)
    remaining_rows = count_pass_progress_fast(tables, recipe)
    if not remaining_rows:
        raise SystemExit(f"No pass counts from {index_path}.")

    if check_disk:
        mode = (
            "strict FLAC validate (opens each file)"
            if strict_disk
            else "exists + size>0"
        )
        print(
            f"Checking claimed stems on disk ({mode}; -j {max(1, int(jobs))}) …",
            flush=True,
        )

    disk_by_pass: dict[str, tuple[int, int]] = {}
    missing_all: list[MissingClaimedStem] = []
    if check_disk:
        for row in remaining_rows:
            pass_name = row["pass"]
            on_disk, claimed, missing = check_claimed_stems_on_disk(
                tables,
                pass_name,
                audio_format=audio_format,
                strict=strict_disk,
                jobs=jobs,
                collect_missing=True,
            )
            disk_by_pass[pass_name] = (on_disk, claimed)
            missing_all.extend(missing)

    lines: list[dict] = []
    assigned_total = 0
    recipe_done_total = 0
    disk_of_claimed_total = 0
    claimed_total = 0

    header = (
        f"{'pass':<12} {'assigned':>10} {'recipe':>10} {'left':>10} "
        f"{'songs_left':>10} {'recipe%':>8}"
    )
    if check_disk:
        header += f" {'on_disk':>10} {'claimed':>10} {'disk%':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for row in remaining_rows:
        pass_name = row["pass"]
        assigned = int(row["assigned"])
        done = int(row["recipe_done"])
        remaining = int(row["remaining"])
        songs_left = int(row["songs_left"])
        assigned_total += assigned
        recipe_done_total += done
        line = {
            "pass": pass_name,
            "assigned": assigned,
            "recipe_done": done,
            "remaining": remaining,
            "songs_left": songs_left,
            "recipe_pct": _pct(done, assigned),
        }
        msg = (
            f"{pass_name:<12} {assigned:>10} {done:>10} {remaining:>10} "
            f"{songs_left:>10} {line['recipe_pct']:>7.1f}%"
        )
        if check_disk:
            on_disk, claimed = disk_by_pass.get(pass_name, (0, 0))
            disk_of_claimed_total += on_disk
            claimed_total += claimed
            line["on_disk"] = on_disk
            line["claimed"] = claimed
            line["disk_pct"] = _pct(on_disk, claimed) if claimed else 100.0
            msg += f" {on_disk:>10} {claimed:>10} {line['disk_pct']:>7.1f}%"
        print(msg, flush=True)
        lines.append(line)

    print("-" * len(header), flush=True)
    overall_pct = _pct(recipe_done_total, assigned_total)
    summary = (
        f"{'TOTAL':<12} {assigned_total:>10} {recipe_done_total:>10} "
        f"{assigned_total - recipe_done_total:>10} {'':>10} {overall_pct:>7.1f}%"
    )
    if check_disk:
        disk_pct = _pct(disk_of_claimed_total, claimed_total) if claimed_total else 100.0
        summary += (
            f" {disk_of_claimed_total:>10} {claimed_total:>10} {disk_pct:>7.1f}%"
        )
    print(summary, flush=True)

    print(
        f"\nPre-realify recipe progress: {overall_pct:.1f}% "
        f"({recipe_done_total:,} / {assigned_total:,} stems recorded in stem_recipe).",
        flush=True,
    )
    if check_disk and claimed_total:
        print(
            f"Claimed stems present on disk: "
            f"{_pct(disk_of_claimed_total, claimed_total):.1f}% "
            f"({disk_of_claimed_total:,} / {claimed_total:,}).",
            flush=True,
        )
    if check_disk and missing_all:
        handle_missing_claimed_stems(
            tables,
            missing_all,
            auto_yes=auto_remove_missing,
        )
    if count_flac:
        print(f"Counting *.flac under {raw_root} (NFS walk) …", flush=True)
        n_flac = count_raw_flac_files(raw_root)
        print(
            f"FLAC files under raw/: {n_flac:,} "
            f"(assigned tracks across passes: {assigned_total:,})",
            flush=True,
        )
    return lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.progress",
        description=(
            "Show pre-realify hybrid render progress from midi_index + "
            "stem_recipe CSVs (strict on-disk validation of claimed stems by default)."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=OUTPUT_DIR,
        help=f"SPDMX output root (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--recipe",
        default=str(DEFAULT_RECIPE_PATH),
        help=f"Category recipe YAML (default: {DEFAULT_RECIPE_PATH})",
    )
    parser.add_argument(
        "--no-check-disk",
        action="store_true",
        help="Skip on-disk validation of claimed stems.",
    )
    parser.add_argument(
        "--no-strict-disk",
        action="store_true",
        help="Check exists + size>0 only (skip FLAC header validation).",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Remove missing claimed stems without prompting.",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4))),
        help="Worker processes for disk validation (default: min(16, CPUs)).",
    )
    parser.add_argument(
        "--count-flac",
        action="store_true",
        help="Also walk SPDMX/raw and count *.flac files (can be slow on NFS).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report_pre_realify_progress(
        output_dir=args.output_dir,
        recipe_path=args.recipe,
        check_disk=not args.no_check_disk,
        strict_disk=not args.no_strict_disk,
        count_flac=bool(args.count_flac),
        jobs=max(1, int(args.jobs)),
        auto_remove_missing=bool(args.yes),
    )


if __name__ == "__main__":
    main()
