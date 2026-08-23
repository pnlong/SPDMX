"""Hybrid production synthesis from a per-category recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared.config import (
    FLAC_AUDIO_FORMAT,
    OUTPUT_DIR,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_FILE_NAME,
)
from synthesis.cli_common import add_synthesis_args
from synthesis.paths import (
    ablation_raw_dir,
    production_tables_dir,
    spdmx_dataset_dir,
)
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.pass_tables import merge_pass_tables
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    load_recipe,
    require_recipe_conflicts_ok,
    scan_recipe_conflicts,
)
from synthesis.synthesize import (
    require_raw_synthesis,
    run_layout_pass,
    run_realify_pass,
    run_synthesis,
)

FINAL_CONDITION = "final"
ONLY_PASSES = (
    "layout", "fluidsynth", "ddsp_piano", "midi_ddsp", "merge", "realify", "mix",
)
DDSP_PASSES = ("ddsp_piano", "midi_ddsp")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.final",
        description=(
            "Synthesize the sPDMX dataset using a per-category recipe. "
            f"Writes FLAC stems under {OUTPUT_DIR}/{SPDMX_DATASET_DIR_NAME}/audio/ "
            "and sanitized MIDI under mid/. Join SPDMX.csv to PDMX.csv on song_id. "
            "Audio format is always FLAC. "
            "Run one pass at a time with --only-pass "
            "(layout → fluidsynth → ddsp_piano → midi_ddsp → mix). "
            "Fluidsynth, ddsp_piano, and midi_ddsp may run in parallel. "
            "Realify and mix merge per-pass CSVs first."
        ),
    )
    add_synthesis_args(
        parser,
        include_render_mode=False,
        include_realify=False,
        full_default=True,
        flac_default=True,
        include_audio_format=False,
    )
    parser.add_argument(
        "--recipe",
        default=str(DEFAULT_RECIPE_PATH),
        type=str,
        help="YAML mapping listening category → ablation id (or method/realify/fallback).",
    )
    parser.add_argument(
        "--only-pass",
        choices=list(ONLY_PASSES),
        required=True,
        help=(
            "Required. One method pass: layout, fluidsynth, ddsp_piano, midi_ddsp, "
            "merge, realify, or mix. Fluidsynth, ddsp_piano, and midi_ddsp may "
            "run in parallel. Mix/realify merge per-pass tables first."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Regenerate stems that no longer match the recipe without prompting.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Echo per-song job-log INFO lines (song start/done, heartbeat) to "
            "stdout. The log file is always written; WARN/ERROR always print."
        ),
    )
    ns = parser.parse_args(args=args, namespace=namespace)
    ns.flac = True
    return ns


def hybrid_dirs(args) -> tuple[str, str]:
    """Return (tables_dir, media_dir).

    Production ``--full``: CSVs under ``dev/final/``, audio/MIDI under ``SPDMX/``.
    ``--ablation-sample``: both under ``dev/ablations/final/``.
    """
    if args.full:
        return production_tables_dir(args.output_dir), spdmx_dataset_dir(args.output_dir)
    dest = ablation_raw_dir(args.output_dir, FINAL_CONDITION)
    return dest, dest


def pass_sequence(recipe) -> tuple[str, ...]:
    """Ordered production passes for this recipe (always starts with layout)."""
    steps = ["layout", "fluidsynth"]
    if recipe.uses_ddsp_piano():
        steps.append("ddsp_piano")
    if recipe.uses_ddsp():
        steps.append("midi_ddsp")
    if recipe.uses_realify():
        steps.append("realify")
    steps.append("mix")
    return tuple(steps)


def raw_upstream_command(recipe) -> str:
    """CLI that must finish before realify or mix."""
    parts = ["uv run python -m synthesis.final --only-pass fluidsynth"]
    if recipe.uses_ddsp_piano():
        parts.append("uv run python -m synthesis.final --only-pass ddsp_piano")
    if recipe.uses_ddsp():
        parts.append("uv run python -m synthesis.final --only-pass midi_ddsp")
    return " && ".join(parts)


def expected_song_count(args, media_dir: str) -> int | None:
    """Unique songs in SPDMX.csv, or None if that table is missing."""
    candidates = [
        Path(media_dir) / f"{SPDMX_FILE_NAME}.csv",
        Path(spdmx_dataset_dir(args.output_dir)) / f"{SPDMX_FILE_NAME}.csv",
    ]
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else ""
        if not path.is_file() or resolved in seen:
            continue
        seen.add(resolved)
        songs = pd.read_csv(path, usecols=["song_id"])
        return int(songs["song_id"].nunique())
    return None


def log_recipe_plan(recipe, *, tables_dir: str, media_dir: str, only: str) -> None:
    grouped = recipe.pass_categories()
    plan = pass_sequence(recipe)
    print(f"Recipe: {recipe.path or '(in-memory)'}")
    print(f"  Fluidsynth categories: {', '.join(grouped['fluidsynth']) or '(none)'}")
    print(f"  MIDI-DDSP categories: {', '.join(grouped['ddsp']) or '(none)'}")
    print(f"  Realify categories: {', '.join(grouped['realify']) or '(none)'}")
    print(f"  Tables: {tables_dir}")
    print(f"  Media: {media_dir}")
    print("  Audio: flac")
    print(f"  Passes: {' → '.join(plan)}")
    print(f"  This job: {only}")


def log_next_pass(recipe, only: str) -> None:
    plan = pass_sequence(recipe)

    def _extra(nxt: str) -> str:
        return " -j 8" if nxt in ("fluidsynth", "mix") else ""

    if only == "merge":
        nxt = "realify" if recipe.uses_realify() else "mix"
        print(
            f"Next: uv run python -m synthesis.final --only-pass {nxt}{_extra(nxt)}",
            flush=True,
        )
        return
    if only not in plan:
        return
    idx = plan.index(only)
    if idx + 1 >= len(plan):
        print("All passes complete.", flush=True)
        return
    nxt = plan[idx + 1]
    extra = _extra(nxt)
    if only == "fluidsynth" and any(p in plan for p in DDSP_PASSES):
        print(
            "Note: --only-pass ddsp_piano and --only-pass midi_ddsp can run in "
            "other jobs at the same time as Fluidsynth (and as each other).",
            flush=True,
        )
        if "realify" in plan:
            print(
                "Realify waits until Fluidsynth, DDSP-Piano, and MIDI-DDSP have all exited.",
                flush=True,
            )
    if only == "ddsp_piano" and "midi_ddsp" in plan:
        print(
            "Note: --only-pass midi_ddsp can run in another job at the same time.",
            flush=True,
        )
    if only in DDSP_PASSES and "realify" in plan:
        print(
            "Start realify only after Fluidsynth and both DDSP jobs have exited.",
            flush=True,
        )
    print(f"Next: uv run python -m synthesis.final --only-pass {nxt}{extra}", flush=True)


def run_summable_mix(args, stems_dir: str) -> None:
    from synthesis.mix import normalize_stems_for_dataset

    print(
        "Normalizing stems in place "
        f"(LUFS + velocity + peak; {FLAC_AUDIO_FORMAT}; mix = sum of stems, no mixture file).",
        flush=True,
    )
    normalize_stems_for_dataset(
        Path(stems_dir),
        Path(stems_dir),
        audio_format=FLAC_AUDIO_FORMAT,
        jobs=args.jobs,
        write_mixture=False,
        pdmx_root=Path(args.dataset_filepath).parent,
        spdmx_output_dir=args.output_dir,
    )


def main(argv=None):
    args = parse_args(argv)
    recipe = load_recipe(args.recipe)
    args.recipe = recipe
    args.render_mode = "basic"
    args.realify = False
    only = args.only_pass
    args.only_pass = only
    tables_dir, media_dir = hybrid_dirs(args)
    log_recipe_plan(recipe, tables_dir=tables_dir, media_dir=media_dir, only=only)
    audio_format = FLAC_AUDIO_FORMAT

    if only != "layout":
        args.skip_output_reset = True

    if only == "layout":
        run_layout_pass(args, tables_dir, media_dir=media_dir)
    elif only in ("fluidsynth", "ddsp_piano", "midi_ddsp"):
        run_synthesis(args, tables_dir, media_dir=media_dir)
    elif only == "merge":
        merge_pass_tables(tables_dir)
    elif only == "realify":
        if not recipe.uses_realify():
            print("Realify pass skipped (no category recipe sets realify).")
        else:
            merge_pass_tables(tables_dir)
            require_raw_synthesis(
                tables_dir,
                run_command=raw_upstream_command(recipe),
                audio_format=audio_format,
                expected_n_songs=expected_song_count(args, media_dir),
            )
            if not args.reset:
                require_recipe_conflicts_ok(
                    scan_recipe_conflicts(
                        tables_dir, recipe, audio_format=audio_format, stage="realify",
                    ),
                    yes=bool(args.yes),
                )
            run_realify_pass(args, tables_dir, tables_dir)
    elif only == "mix":
        merge_pass_tables(tables_dir)
        require_raw_synthesis(
            tables_dir,
            run_command=raw_upstream_command(recipe),
            audio_format=audio_format,
            expected_n_songs=expected_song_count(args, media_dir),
        )
        run_summable_mix(args, tables_dir)

    log_next_pass(recipe, only)
    link_ablations_in_repo(args.output_dir)


if __name__ == "__main__":
    main()
