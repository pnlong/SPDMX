"""Analyze General MIDI program usage from PDMX MIDI files (incl. drums)."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.gm_programs import (
    build_gm_report,
    extract_gm_stems_from_mid,
    print_gm_report,
    stems_dataframe,
    stems_from_register,
)
from analysis.pdmx_subset import subset_output_dir
from analysis.plots import plot_gm_program_bar, plot_gm_program_compare
from analysis.track_names import load_pdmx_mid_paths, mid_path_for_row
from shared.config import CHUNK_SIZE, OUTPUT_DIR, PDMX_FILEPATH
from shared.repo_symlinks import link_analysis_in_repo
from synthesis.cli_common import default_gm_register_path
from synthesis.paths import instruments_dir

CORRECTED_SOURCE = (
    "GM register program_corrected (track-name corrections; "
    "excludes skipped_unnamed empty-name rows)"
)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize General MIDI program ids from PDMX MIDI files "
            "(melodic program_change + channel-10 drum kits), or from a "
            "corrected GM register CSV."
        ),
    )
    parser.add_argument("-df", "--dataset_filepath", default=PDMX_FILEPATH, type=str)
    parser.add_argument("-o", "--output_dir", default=instruments_dir(OUTPUT_DIR), type=str)
    parser.add_argument(
        "--subset",
        choices=("all_valid", "rated_deduplicated", "all"),
        default="all_valid",
        help="PDMX rows to include (default: all_valid, same as synthesize --full).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=max(1, multiprocessing.cpu_count() // 2),
        type=int,
        help="Worker processes for MIDI parsing.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        default=None,
        type=int,
        help="Optional cap on number of songs (for smoke tests).",
    )
    parser.add_argument(
        "--top-n",
        default=40,
        type=int,
        help="Number of GM ids to show in the bar chart (rest grouped as Other).",
    )
    parser.add_argument(
        "--from-register",
        nargs="?",
        const="",
        default=None,
        type=str,
        help=(
            "Build inventory from register program_corrected instead of raw MIDI. "
            "Optional path (default: instruments/.../register.csv for --subset). "
            "Writes gm_program_*_corrected.* outputs."
        ),
    )
    return parser.parse_args(args=args, namespace=namespace)


_WORKER_PRESETS: dict | None = None


def _init_worker() -> None:
    global _WORKER_PRESETS
    _WORKER_PRESETS = None


def _worker(args: tuple[str, str]) -> tuple[list[dict], bool]:
    mid_rel, pdmx_root = args
    mid_path = mid_path_for_row(mid_rel, pdmx_root)
    if not mid_path.is_file():
        return [], True
    rows = extract_gm_stems_from_mid(mid_path, presets=_WORKER_PRESETS)
    if rows is None:
        return [], True
    return rows, False


def analyze_gm_programs(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
    jobs: int = 1,
    limit: int | None = None,
) -> tuple[dict, object]:
    mid_paths, pdmx_root = load_pdmx_mid_paths(dataset_filepath, subset=subset)
    if limit is not None:
        mid_paths = mid_paths.head(limit)

    n_songs = len(mid_paths)
    worker_args = [(mid_rel, str(pdmx_root)) for mid_rel in mid_paths]

    stem_records: list[dict] = []
    n_songs_failed = 0

    if jobs <= 1:
        _init_worker()
        iterator = worker_args
        if n_songs > 1:
            iterator = tqdm(worker_args, total=n_songs, desc="Parsing MIDI", unit="song")
        for mid_rel, root in iterator:
            rows, failed = _worker((mid_rel, root))
            stem_records.extend(rows)
            n_songs_failed += int(failed)
    else:
        with multiprocessing.Pool(processes=jobs, initializer=_init_worker) as pool:
            for rows, failed in tqdm(
                pool.imap(_worker, worker_args, chunksize=CHUNK_SIZE),
                total=n_songs,
                desc="Parsing MIDI",
                unit="song",
            ):
                stem_records.extend(rows)
                n_songs_failed += int(failed)

    stems = stems_dataframe(stem_records)
    report = build_gm_report(
        stems,
        subset=subset,
        n_songs=n_songs,
        n_songs_failed=n_songs_failed,
    )
    return report, stems


def analyze_gm_programs_from_register(
    register: pd.DataFrame,
    *,
    subset: str = "all_valid",
) -> tuple[dict, pd.DataFrame]:
    """Inventory from register ``program_corrected`` (excludes skipped_unnamed)."""
    from analysis.gm_register import STATUS_SKIPPED_UNNAMED

    filtered = register
    if "status" in register.columns:
        filtered = register[register["status"] != STATUS_SKIPPED_UNNAMED]
    stems = stems_from_register(filtered)
    n_songs = int(register["mid"].nunique()) if len(register) and "mid" in register.columns else 0
    report = build_gm_report(
        stems,
        subset=subset,
        n_songs=n_songs,
        n_songs_failed=0,
        source=CORRECTED_SOURCE,
    )
    return report, stems


def write_gm_program_outputs(
    report: dict,
    stems: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 40,
    corrected: bool = False,
    stems_original: pd.DataFrame | None = None,
) -> list[Path]:
    """Write report JSON, stems CSV, and bar chart(s). Returns written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_corrected" if corrected else ""
    chart_path = output_dir / f"gm_program_counts{suffix}.png"
    report_path = output_dir / f"gm_program_report{suffix}.json"
    stems_path = output_dir / f"gm_program_stems{suffix}.csv"

    title = (
        "sPDMX corrected GM program usage (from register)"
        if corrected
        else "PDMX General MIDI program usage (drums = channel 10)"
    )
    plot_gm_program_bar(stems, chart_path, top_n=top_n, title=title)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    stems.to_csv(stems_path, index=False)

    written = [report_path, chart_path, stems_path]

    if corrected and stems_original is not None and len(stems_original):
        compare_path = output_dir / "gm_program_counts_compare.png"
        plot_gm_program_compare(
            stems_original,
            stems,
            compare_path,
            top_n=top_n,
        )
        written.append(compare_path)

    return written


def _load_original_stems(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "gm_program_stems.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def main():
    args = parse_args()
    output_dir = subset_output_dir(args.output_dir, args.subset)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_register is not None:
        register_path = args.from_register or str(output_dir / "register.csv")
        if args.from_register == "" and not Path(register_path).is_file():
            register_path = default_gm_register_path(OUTPUT_DIR)
        register_path = Path(register_path)
        if not register_path.is_file():
            raise SystemExit(
                f"GM register not found: {register_path}\n"
                "Run: uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
            )
        register = pd.read_csv(register_path)
        report, stems = analyze_gm_programs_from_register(register, subset=args.subset)
        written = write_gm_program_outputs(
            report,
            stems,
            output_dir,
            top_n=args.top_n,
            corrected=True,
            stems_original=_load_original_stems(output_dir),
        )
    else:
        report, stems = analyze_gm_programs(
            args.dataset_filepath,
            subset=args.subset,
            jobs=args.jobs,
            limit=args.limit,
        )
        written = write_gm_program_outputs(
            report,
            stems,
            output_dir,
            top_n=args.top_n,
            corrected=False,
        )

    link, target = link_analysis_in_repo(OUTPUT_DIR)

    print_gm_report(report)
    print()
    for path in written:
        print(f"Wrote {path}")
    print(f"Symlinked {link} -> {target}")


if __name__ == "__main__":
    main()
