"""Hybrid production synthesis from a per-category recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared.config import (
    FLAC_AUDIO_FORMAT,
    OUTPUT_DIR,
    SPDMX_DEV_DIR_NAME,
    SPDMX_FILE_NAME,
)
from synthesis.cli_common import add_synthesis_args
from synthesis.paths import (
    MIDI_INDEX_FILE_NAME,
    ablation_raw_dir,
    production_tables_dir,
    spdmx_dev_dir,
)
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.pass_tables import merge_pass_tables
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    load_recipe,
    require_recipe_conflicts_ok,
    scan_recipe_conflicts,
)
from synthesis.shard import format_shard_summary, shard_song_ids, validate_shard_args
from synthesis.synthesize import (
    require_raw_synthesis,
    run_layout_pass,
    run_realify_pass,
    run_synthesis,
    verify_claimed_stems_on_disk,
)
GLOBAL_ONLY_PASSES = ("layout", "merge", "mix", "verify")


def _reject_sharded_global_pass(args) -> None:
    shard_count = int(getattr(args, "shard_count", 1) or 1)
    if shard_count > 1:
        raise SystemExit(
            f"--only-pass {args.only_pass} must run unsharded (--shard-count 1). "
            "Use --shard-count / --shard-index only on fluidsynth, ddsp_piano, "
            "midi_ddsp, or realify."
        )


def _realify_allowed_song_ids(args, tables_dir: str) -> set[str] | None:
    shard_count = int(getattr(args, "shard_count", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    try:
        validate_shard_args(shard_count, shard_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if shard_count == 1:
        return None
    index_path = Path(tables_dir) / MIDI_INDEX_FILE_NAME
    if not index_path.is_file():
        raise SystemExit(
            f"Cannot shard realify: missing {index_path}. "
            "Run --only-pass layout first."
        )
    song_ids = pd.read_csv(index_path, usecols=["song_id"])["song_id"].astype(str).tolist()
    allowed = shard_song_ids(
        song_ids, shard_count=shard_count, shard_index=shard_index,
    )
    print(
        format_shard_summary(shard_count, shard_index, len(allowed), len(allowed)),
        flush=True,
    )
    return allowed

FINAL_CONDITION = "final"
ONLY_PASSES = (
    "layout", "fluidsynth", "ddsp_piano", "midi_ddsp", "merge", "realify",
    "mix", "verify",
)
DDSP_PASSES = ("ddsp_piano", "midi_ddsp")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.final",
        description=(
            "Synthesize the sPDMX dataset using a per-category recipe. "
            f"Writes raw FLAC stems under {OUTPUT_DIR}/{SPDMX_DEV_DIR_NAME}/raw/ "
            f"(mix writes summable stems to audio/ and mix/<song_id>.flac) "
            "and sanitized MIDI under mid/. Join stems.csv to PDMX.csv on song_id. "
            "Audio format is always FLAC. "
            "Run one pass at a time with --only-pass "
            "(layout → fluidsynth → ddsp_piano → midi_ddsp → mix → verify). "
            "Fluidsynth, ddsp_piano, and midi_ddsp may run in parallel. "
            "Realify, mix, and verify merge per-pass CSVs first."
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
            "merge, realify, mix, or verify. Fluidsynth, ddsp_piano, and midi_ddsp "
            "may run in parallel. Mix normalizes raw→audio and writes "
            "mix/<song_id>.flac (dirty-aware resume). Verify checks raw completeness "
            "and fully FLAC-decodes audio/ stems and song mixes, then checks "
            "sample-wise mix == sum(stems). Use --delete-bad-mix-sums with verify "
            "to remove failing mixes (then re-run mix), or --repair-mix-sums with "
            "mix to delete+remake in one shot. "
            "Mix/realify/verify merge per-pass tables first."
        ),
    )
    parser.add_argument(
        "--delete-bad-mix-sums",
        action="store_true",
        help=(
            "With --only-pass verify: delete only mixes that fail a *content* "
            "sum check (max|mix-sum| / length / sample-rate). Never deletes on "
            "missing stems or read errors. Refuses if more than "
            "max(100, 5% of catalog) mixes would be deleted unless also "
            "pass -y/--yes. Then re-run mix to remake them."
        ),
    )
    parser.add_argument(
        "--repair-mix-sums",
        action="store_true",
        help=(
            "With --only-pass mix: delete mixes that fail sum(stems) and remake "
            "them only (skips stem normalize)."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Regenerate stems that no longer match the recipe without prompting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report recipe conflicts and remaining work for --only-pass without "
            "writing audio, dropping tables, or merging. Exits after the report."
        ),
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
        return production_tables_dir(args.output_dir), spdmx_dev_dir(args.output_dir)
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
    steps.extend(["mix", "verify"])
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
    """Unique songs in stems.csv, or None if that table is missing."""
    candidates = [
        Path(media_dir) / f"{SPDMX_FILE_NAME}.csv",
        Path(spdmx_dev_dir(args.output_dir)) / f"{SPDMX_FILE_NAME}.csv",
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
        return " -j 8" if nxt in ("fluidsynth", "verify", "mix") else ""

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
        if only == "verify":
            print(
                "After verify, run: uv run python -m synthesis.build_spdmx "
                "to publish flattened chunk_N/<song_id>/ trees.",
                flush=True,
            )
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
    if only == "mix":
        print(
            "Mix writes summable audio/ stems and mix/<song_id>.flac "
            "(dirty-aware: only songs with newer raw/audio inputs).",
            flush=True,
        )
    if only == "verify":
        print(
            "Verify checks raw completeness and fully FLAC-decodes audio/ stems "
            "and mix/<song_id>.flac via -j/--jobs.",
            flush=True,
        )
    print(f"Next: uv run python -m synthesis.final --only-pass {nxt}{extra}", flush=True)


def run_summable_mix(args, stems_dir: str, *, media_dir: str) -> None:
    from shared.config import (
        SPDMX_AUDIO_DIR_NAME,
        SPDMX_FILE_NAME,
        SPDMX_RAW_DIR_NAME,
    )
    from synthesis.mix import normalize_stems_for_dataset
    from synthesis.paths import raw_path_to_audio
    from synthesis.render_mixes import render_dataset_mixes, repair_mix_sum_mismatches

    media = Path(media_dir)
    raw_root = media / SPDMX_RAW_DIR_NAME
    audio_root = media / SPDMX_AUDIO_DIR_NAME
    reset = bool(getattr(args, "reset", False))
    repair_sums = bool(getattr(args, "repair_mix_sums", False))

    if repair_sums and not reset:
        # Delete mismatched mixes and remake them; leave audio/ alone.
        counts = repair_mix_sum_mismatches(
            media_dir,
            tables_dir=stems_dir,
            jobs=args.jobs,
            force_delete=bool(getattr(args, "yes", False)),
        )
        if counts.get("error", 0):
            raise SystemExit(1)
        return

    print(
        f"Writing mixable stems to {audio_root}/ "
        f"(raw {raw_root}/ untouched; LUFS + velocity + peak; "
        f"{FLAC_AUDIO_FORMAT}; then mix/<song_id>.flac; "
        f"{'reset' if reset else 'dirty-aware resume'}).",
        flush=True,
    )
    dirty_ids = normalize_stems_for_dataset(
        Path(stems_dir),
        Path(stems_dir),
        audio_format=FLAC_AUDIO_FORMAT,
        jobs=args.jobs,
        write_mixture=False,
        pdmx_root=Path(args.dataset_filepath).parent,
        spdmx_output_dir=args.output_dir,
        dest_song_dir_fn=raw_path_to_audio,
        reset=reset,
    )
    counts = render_dataset_mixes(
        media_dir,
        jobs=args.jobs,
        force=reset,
        force_ids=dirty_ids,
    )
    print(
        f"mix files: wrote={counts.get('wrote', 0)} "
        f"skip_up_to_date={counts.get('skip_exists', 0)} "
        f"skip_no_stems={counts.get('skip_no_stems', 0)} "
        f"error={counts.get('error', 0)}",
        flush=True,
    )
    if counts.get("error", 0):
        raise SystemExit(1)
    spdmx_csv = media / f"{SPDMX_FILE_NAME}.csv"
    if spdmx_csv.is_file():
        table = pd.read_csv(spdmx_csv)
        if "path" in table.columns:
            def _to_audio(p: str) -> str:
                text = str(p).replace("\\", "/")
                if (
                    f"/{SPDMX_RAW_DIR_NAME}/" in text
                    or text.startswith(f"./{SPDMX_RAW_DIR_NAME}/")
                ):
                    return raw_path_to_audio(str(p))
                return p

            table["path"] = table["path"].map(_to_audio)
            table.to_csv(spdmx_csv, index=False)
            print(
                f"Updated {spdmx_csv} paths → ./{SPDMX_AUDIO_DIR_NAME}/…",
                flush=True,
            )


def run_dry_run(
    args,
    recipe,
    tables_dir: str,
    media_dir: str,
    *,
    only: str,
    audio_format: str,
) -> None:
    """Print what ``--only-pass`` would do without writing anything."""
    from collections import Counter

    from synthesis.recipe import scan_recipe_conflicts
    from synthesis.synthesize import count_pass_remaining

    print(f"DRY RUN: --only-pass {only} (no writes)", flush=True)

    if only in ("fluidsynth", "ddsp_piano", "midi_ddsp", "realify", "mix", "verify"):
        stage = "realify" if only == "realify" else "raw"
        categories = None
        if only == "fluidsynth":
            categories = frozenset(recipe.pass_categories()["fluidsynth"])
        elif only == "midi_ddsp":
            categories = frozenset(recipe.pass_categories()["ddsp"])
        elif only == "ddsp_piano":
            categories = frozenset({"piano"}) if recipe.uses_ddsp_piano() else frozenset()
        elif only == "realify":
            categories = recipe.realify_categories() or None
        print("Scanning recipe conflicts …", flush=True)
        conflicts = scan_recipe_conflicts(
            tables_dir,
            recipe,
            audio_format=audio_format,
            stage=stage,
            categories=categories,
        )
        by_cat = Counter(c.category or "?" for c in conflicts)
        print(f"Recipe conflicts ({stage}): {len(conflicts)}", flush=True)
        for cat, n in sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cat}: {n}", flush=True)
        preview = conflicts[:20]
        for item in preview:
            loc = f"{item.path} track={item.track}"
            cat = item.category or "?"
            print(f"  [{cat}] {loc}", flush=True)
            print(f"    recorded: {item.recorded}", flush=True)
            print(f"    desired:  {item.desired}", flush=True)
        if len(conflicts) > len(preview):
            print(f"  … and {len(conflicts) - len(preview)} more", flush=True)

    if only in ("fluidsynth", "ddsp_piano", "midi_ddsp"):
        rows = count_pass_remaining(tables_dir, recipe=recipe)
        matched = [r for r in rows if r["pass"] == only]
        if not matched:
            print(
                f"Pass remaining ({only}): (no midi_index / pass counts available)",
                flush=True,
            )
        for row in matched:
            print(
                f"Pass remaining ({only}): {row['remaining']} tracks "
                f"({row['songs_left']} songs; "
                f"{row['done']}/{row['assigned']} recorded in stem_recipe)",
                flush=True,
            )
            for ex in row.get("examples") or []:
                print(
                    f"  e.g. song_id={ex.get('song_id')} "
                    f"remaining={ex.get('remaining')}",
                    flush=True,
                )

    if only == "mix":
        from synthesis.mix import build_mixture_tasks
        from synthesis.paths import raw_path_to_audio
        from synthesis.render_mixes import render_dataset_mixes
        from shared.config import STEMS_FILE_NAME

        stems_csv = Path(tables_dir) / f"{STEMS_FILE_NAME}.csv"
        if not stems_csv.is_file():
            print(
                "Mix: missing stems.csv (run --only-pass merge first to preview).",
                flush=True,
            )
        else:
            stems = pd.read_csv(stems_csv)
            tasks = build_mixture_tasks(
                stems,
                Path(tables_dir),
                Path(tables_dir),
                audio_format,
                write_mixture=False,
                pdmx_root=Path(args.dataset_filepath).parent,
                spdmx_output_dir=args.output_dir,
                dest_song_dir_fn=raw_path_to_audio,
                reset=bool(getattr(args, "reset", False)),
            )
            print(
                f"Mix normalize: {len(tasks)} song(s) "
                f"(clean audio/ skipped inline during normalize)",
                flush=True,
            )
            if Path(media_dir).joinpath("audio").is_dir():
                counts = render_dataset_mixes(
                    media_dir,
                    jobs=int(getattr(args, "jobs", 1) or 1),
                    force=bool(getattr(args, "reset", False)),
                    dry_run=True,
                    force_ids=None,
                    update_csv=False,
                )
                print(
                    f"Mix files: would write={counts.get('wrote', 0)} "
                    f"skip_up_to_date={counts.get('skip_exists', 0)} "
                    f"skip_no_stems={counts.get('skip_no_stems', 0)}",
                    flush=True,
                )
            else:
                print(f"Mix files: no {media_dir}/audio/ yet", flush=True)

    if only == "merge":
        print(
            "Merge: would rebuild stems.csv / stem_recipe.csv / data.csv "
            "from per-pass shards (no audio writes).",
            flush=True,
        )
    elif only == "layout":
        print("Layout: would create song dirs + empty tables / midi_index.", flush=True)
    elif only == "verify":
        delete_note = ""
        if getattr(args, "delete_bad_mix_sums", False):
            delete_note = (
                " Would delete mix files that fail sum(stems) "
                "(then re-run mix to remake)."
            )
        print(
            "Verify: would check raw completeness, FLAC-decode audio/ + mix/, "
            f"and sample-wise mix == sum(stems).{delete_note}",
            flush=True,
        )
    elif only == "realify":
        if not recipe.uses_realify():
            print("Realify: skipped (no category sets realify).", flush=True)
        else:
            print("Realify: would overwrite stems marked *_realify in the recipe.", flush=True)

    print("DRY RUN complete (nothing written).", flush=True)


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

    if getattr(args, "dry_run", False):
        if only in GLOBAL_ONLY_PASSES:
            _reject_sharded_global_pass(args)
        run_dry_run(
            args,
            recipe,
            tables_dir,
            media_dir,
            only=only,
            audio_format=audio_format,
        )
        return

    if only != "layout":
        args.skip_output_reset = True

    if only in GLOBAL_ONLY_PASSES:
        _reject_sharded_global_pass(args)

    if only == "layout":
        run_layout_pass(args, tables_dir, media_dir=media_dir)
    elif only in ("fluidsynth", "ddsp_piano", "midi_ddsp"):
        run_synthesis(args, tables_dir, media_dir=media_dir)
    elif only == "merge":
        merge_pass_tables(tables_dir, media_dir=media_dir)
    elif only == "realify":
        if not recipe.uses_realify():
            print("Realify pass skipped (no category recipe sets realify).")
        else:
            merge_pass_tables(tables_dir, media_dir=media_dir)
            require_raw_synthesis(
                tables_dir,
                run_command=raw_upstream_command(recipe),
                audio_format=audio_format,
                expected_n_songs=expected_song_count(args, media_dir),
                jobs=args.jobs,
            )
            if not args.reset:
                require_recipe_conflicts_ok(
                    scan_recipe_conflicts(
                        tables_dir, recipe, audio_format=audio_format, stage="realify",
                    ),
                    yes=bool(args.yes),
                )
            allowed = _realify_allowed_song_ids(args, tables_dir)
            run_realify_pass(args, tables_dir, tables_dir, allowed_song_ids=allowed)
    elif only == "verify":
        merge_pass_tables(tables_dir, media_dir=media_dir)
        # Report remaining-per-pass and missing FLACs first (actionable), then
        # the stricter data.csv completeness check, then decode audio/ + mix/.
        verify_claimed_stems_on_disk(
            tables_dir, audio_format, recipe=recipe, jobs=args.jobs,
        )
        require_raw_synthesis(
            tables_dir,
            run_command=raw_upstream_command(recipe),
            audio_format=audio_format,
            expected_n_songs=expected_song_count(args, media_dir),
            jobs=args.jobs,
        )
        from synthesis.mix import verify_mixed_stems_on_disk

        verify_mixed_stems_on_disk(
            tables_dir,
            audio_format=audio_format,
            jobs=args.jobs,
            media_dir=media_dir,
            delete_bad_mix_sums=bool(getattr(args, "delete_bad_mix_sums", False)),
            force_delete_bad_mixes=bool(getattr(args, "yes", False)),
        )
    elif only == "mix":
        merge_pass_tables(tables_dir, media_dir=media_dir)
        require_raw_synthesis(
            tables_dir,
            run_command=raw_upstream_command(recipe),
            audio_format=audio_format,
            expected_n_songs=expected_song_count(args, media_dir),
            jobs=args.jobs,
        )
        run_summable_mix(args, tables_dir, media_dir=media_dir)

    log_next_pass(recipe, only)
    link_ablations_in_repo(args.output_dir)


if __name__ == "__main__":
    main()
