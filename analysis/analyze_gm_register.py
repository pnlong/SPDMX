"""Build the GM program register from PDMX MIDI track names."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.gm_register import (
    STATUS_CORRECTED,
    build_register_report,
    format_register_report,
    extract_register_rows_from_mid,
    load_alias_config,
    print_register_report,
    register_dataframe,
    top_corrections_dataframe,
)
from analysis.pdmx_subset import subset_output_dir
from analysis.track_names import load_pdmx_mid_paths, mid_path_for_row
from shared.config import CHUNK_SIZE, OUTPUT_DIR, PDMX_FILEPATH
from shared.repo_symlinks import link_analysis_in_repo
from synthesis.paths import instruments_dir, mid_corrected_dir

_WORKER_CONFIG = None


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description=(
            "Step-0 synthesis setup: correct GM program ids from MIDI track names, "
            "write register.csv, and (by default) write dense corrected MIDI copies "
            f"under {OUTPUT_DIR}/SPDMX_dev/mid/ (empty tracks dropped). "
            "Prefer: python -m analysis.prepare_synthesis"
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
        "--aliases",
        default=None,
        type=str,
        help="Path to gm_register_aliases.yaml (default: analysis/gm_register_aliases.yaml).",
    )
    parser.add_argument(
        "--from-csv",
        default=None,
        type=str,
        help="Recompute/print stats from an existing register.csv (skip MIDI parse).",
    )
    parser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="How many top corrections / keys to include in the report.",
    )
    midi_group = parser.add_mutually_exclusive_group()
    midi_group.add_argument(
        "--write-corrected-midi",
        dest="write_corrected_midi",
        action="store_true",
        default=True,
        help=(
            "Write dense corrected MIDI copies under --corrected-midi-dir "
            "(default: on). Empty tracks dropped; register programs applied."
        ),
    )
    midi_group.add_argument(
        "--no-write-corrected-midi",
        dest="write_corrected_midi",
        action="store_false",
        help="Skip dense corrected MIDI copies (register CSV / reports only).",
    )
    parser.add_argument(
        "--corrected-midi-dir",
        default=None,
        type=str,
        help=f"Output root for corrected MIDIs (default: {OUTPUT_DIR}/SPDMX_dev/mid/).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def _init_worker(aliases_path: str | None) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = load_alias_config(aliases_path)


def _worker(args: tuple[str, str]) -> tuple[list[dict], bool]:
    mid_rel, pdmx_root = args
    mid_path = mid_path_for_row(mid_rel, pdmx_root)
    if not mid_path.is_file():
        return [], True
    rows = extract_register_rows_from_mid(
        mid_path,
        mid_rel=mid_rel,
        config=_WORKER_CONFIG,
    )
    if rows is None:
        return [], True
    return rows, False


def analyze_gm_register(
    dataset_filepath: str | Path,
    *,
    subset: str = "all_valid",
    jobs: int = 1,
    limit: int | None = None,
    aliases_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict, int]:
    """Parse PDMX MIDI files and return (register_df, report_meta_stub, n_songs_failed)."""
    mid_paths, pdmx_root = load_pdmx_mid_paths(dataset_filepath, subset=subset)
    if limit is not None:
        mid_paths = mid_paths.head(limit)

    n_songs = len(mid_paths)
    worker_args = [(mid_rel, str(pdmx_root)) for mid_rel in mid_paths]
    aliases = str(aliases_path) if aliases_path is not None else None

    register_rows: list[dict] = []
    n_songs_failed = 0

    if jobs <= 1:
        _init_worker(aliases)
        for item in tqdm(worker_args, total=n_songs, desc="Building register", unit="song"):
            rows, failed = _worker(item)
            register_rows.extend(rows)
            n_songs_failed += int(failed)
    else:
        with multiprocessing.Pool(
            processes=jobs,
            initializer=_init_worker,
            initargs=(aliases,),
        ) as pool:
            for rows, failed in tqdm(
                pool.imap(_worker, worker_args, chunksize=CHUNK_SIZE),
                total=n_songs,
                desc="Building register",
                unit="song",
            ):
                register_rows.extend(rows)
                n_songs_failed += int(failed)

    register = register_dataframe(register_rows)
    return register, {"subset": subset, "n_songs": n_songs}, n_songs_failed


def write_register_outputs(
    register: pd.DataFrame,
    output_dir: Path,
    *,
    subset: str | None = None,
    n_songs: int | None = None,
    n_songs_failed: int = 0,
    top_n: int = 20,
    write_tables: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_register_report(
        register,
        subset=subset,
        n_songs=n_songs,
        n_songs_failed=n_songs_failed,
        top_n=top_n,
    )

    written: list[Path] = []
    if write_tables:
        register_path = output_dir / "register.csv"
        register.to_csv(register_path, index=False)
        written.append(register_path)

        corrections = (
            register[register["status"] == STATUS_CORRECTED] if len(register) else register
        )
        corrections_path = output_dir / "register_corrections.csv"
        corrections.to_csv(corrections_path, index=False)
        written.append(corrections_path)

    summary_path = output_dir / "register_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    written.append(summary_path)

    report_txt_path = output_dir / "register_report.txt"
    report_txt_path.write_text(format_register_report(report))
    written.append(report_txt_path)

    top_path = output_dir / "register_top_corrections.csv"
    top_corrections_dataframe(report).to_csv(top_path, index=False)
    written.append(top_path)

    print_register_report(report)
    print()
    for path in written:
        print(f"Wrote {path}")
    return report


def main():
    args = parse_args()

    if args.from_csv:
        register = pd.read_csv(args.from_csv)
        out = Path(args.from_csv).resolve().parent
        if args.output_dir != instruments_dir(OUTPUT_DIR):
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
        write_register_outputs(
            register,
            out,
            subset=args.subset,
            n_songs=None,
            n_songs_failed=0,
            top_n=args.top_n,
            write_tables=False,
        )
        if args.write_corrected_midi:
            from analysis.corrected_midi import write_corrected_midis_from_register
            from analysis.track_names import load_pdmx_mid_paths

            _, pdmx_root = load_pdmx_mid_paths(args.dataset_filepath, subset=args.subset)
            corrected_root = Path(args.corrected_midi_dir or mid_corrected_dir(OUTPUT_DIR))
            ok, failed = write_corrected_midis_from_register(
                register,
                pdmx_root=pdmx_root,
                corrected_midi_dir=corrected_root,
                jobs=args.jobs,
            )
            print(f"Corrected MIDI: wrote {ok}, failed {failed} → {corrected_root}")
        return

    output_dir = subset_output_dir(args.output_dir, args.subset)
    register, meta, n_songs_failed = analyze_gm_register(
        args.dataset_filepath,
        subset=args.subset,
        jobs=args.jobs,
        limit=args.limit,
        aliases_path=args.aliases,
    )
    write_register_outputs(
        register,
        output_dir,
        subset=meta["subset"],
        n_songs=meta["n_songs"],
        n_songs_failed=n_songs_failed,
        top_n=args.top_n,
    )

    # Also write corrected GM inventory plot/report next to the register.
    from analysis.analyze_gm_programs import (
        _load_original_stems,
        analyze_gm_programs_from_register,
        write_gm_program_outputs,
    )
    from analysis.gm_programs import print_gm_report

    gm_report, gm_stems = analyze_gm_programs_from_register(
        register, subset=meta["subset"]
    )
    gm_paths = write_gm_program_outputs(
        gm_report,
        gm_stems,
        output_dir,
        top_n=40,
        corrected=True,
        stems_original=_load_original_stems(output_dir),
    )
    print("\nCorrected GM program inventory:")
    print_gm_report(gm_report)
    for path in gm_paths:
        print(f"Wrote {path}")

    if args.write_corrected_midi:
        from analysis.corrected_midi import write_corrected_midis_from_register
        from analysis.track_names import load_pdmx_mid_paths

        _, pdmx_root = load_pdmx_mid_paths(args.dataset_filepath, subset=args.subset)
        corrected_root = Path(args.corrected_midi_dir or mid_corrected_dir(OUTPUT_DIR))
        ok, failed = write_corrected_midis_from_register(
            register,
            pdmx_root=pdmx_root,
            corrected_midi_dir=corrected_root,
            jobs=args.jobs,
        )
        print(f"Corrected MIDI: wrote {ok}, failed {failed} → {corrected_root}")

    link, target = link_analysis_in_repo(OUTPUT_DIR)
    print(f"Symlinked {link} -> {target}")


if __name__ == "__main__":
    main()
