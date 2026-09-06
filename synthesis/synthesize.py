"""Synthesize PDMX MIDI stems; optionally realify with SA3."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from os import makedirs, remove
from os.path import dirname, exists, expanduser
from pathlib import Path

import mido
import pandas as pd
from tqdm import tqdm

from shared.config import (
    CHUNK_SIZE,
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    MAX_N_NOTES_IN_STEM,
    NA_STRING,
    REALIFY_BATCH_SIZE,
    REALIFY_CONTENT_FIDELITY_ENFORCE,
    REALIFY_SILENCE_ENFORCE,
    SONGS_TABLE_COLUMNS,
    SOUNDFONT_DIR,
    SPDMX_FILE_NAME,
    SPDMX_MID_DIR_NAME,
    SPDMX_RAW_DIR_NAME,
    STEMS_FILE_NAME,
    STEMS_TABLE_COLUMNS,
)
from synthesis.audio import (
    get_waveform_tensor,
    save_stem,
    song_is_complete,
    stem_is_valid,
    stem_path,
    synthesis_audio_format,
)
from synthesis.cli_common import add_synthesis_args, default_gm_register_path
from synthesis.shard import (
    filter_work_indices_by_shard,
    format_shard_summary,
    validate_shard_args,
)
from synthesis.dataset import listening_sample_path, prepare_ablation_dataset, prepare_full_dataset
from shared.csv_tables import append_rows, append_rows_deduped, sanitize_track_name
from synthesis.paths import (
    MIDI_INDEX_FILE_NAME,
    PASS_TRACK_COLUMNS,
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
)
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.ddsp.config import DDSP_ROUTING_COLUMNS, DDSP_ROUTING_FILE_NAME
from synthesis.patches import PatchAssignment, apply_patch_to_midi_track
from synthesis.reuse import (
    copy_stem,
    donor_raw_stem_path,
    fallback_donor_mode,
    reused_source_label,
    song_rel_under_data,
    uses_ddsp,
    uses_slakh_recipes,
)


def _hybrid_recipe(args):
    return getattr(args, "recipe", None)


def _needs_ddsp_routing(args) -> bool:
    recipe = _hybrid_recipe(args)
    if recipe is not None:
        return recipe.uses_ddsp()
    return uses_ddsp(getattr(args, "render_mode", "") or "")


def _pool_should_spawn(args) -> bool:
    if uses_ddsp(getattr(args, "render_mode", "") or "") and _hybrid_recipe(args) is None:
        return True
    return False


def _resume_check_disk(args) -> bool:
    """True when resume requires both stem_recipe CSV and a valid on-disk FLAC.

    Default is on; ``--no-resume-check-disk`` allows CSV-only skip (e.g. rsync).
    """
    return bool(getattr(args, "resume_check_disk", True))


def _hybrid_raw_current(args, song_path: str, track: int, out_stem, plan, backend: str) -> bool:
    """True when this stem can be skipped for the current raw recipe.

    Default: matching ``stem_recipe`` row and a valid on-disk FLAC. With
    ``--no-resume-check-disk``, the CSV row alone is enough.
    """
    from synthesis.recipe import raw_fingerprint, recorded_raw_fingerprint

    if args.reset:
        return False
    rec = (getattr(args, "stem_recipe_index", None) or {}).get(
        (str(song_path), int(track))
    )
    if recorded_raw_fingerprint(rec) != raw_fingerprint(
        plan.method, plan.fallback, backend,
    ):
        return False
    if _resume_check_disk(args) and not stem_is_valid(out_stem):
        return False
    return True


def _hybrid_song_raw_current(
    args, path_output: str, n_tracks: int, song_dir, audio_format: str, recipe,
) -> bool:
    from synthesis.recipe import desired_raw_fingerprint, recorded_raw_fingerprint

    index = getattr(args, "stem_recipe_index", None) or {}
    check_disk = _resume_check_disk(args)
    for j in range(n_tracks):
        if check_disk and not stem_is_valid(stem_path(song_dir, j, audio_format)):
            return False
        rec = index.get((str(path_output), j))
        if rec is None:
            return False
        category = rec.get("category")
        if category not in recipe.specs:
            return False
        spec = recipe.spec_for_category(str(category))
        backend = str(rec.get("backend") or "fluidsynth")
        if recorded_raw_fingerprint(rec) != desired_raw_fingerprint(spec, backend):
            return False
    return True


def _song_table_row(dataset: pd.DataFrame, i: int, path_output: str, n_tracks: int) -> dict:
    song_info = dataset.loc[i].to_dict()
    song_info["path"] = path_output
    song_info.pop("path_output", None)
    song_info.pop("mid", None)
    song_info.pop("mid_pdmx", None)
    song_info["n_tracks"] = n_tracks
    return song_info


def _hybrid_routing_rows(
    path_output: str, rendered_stem_rows: list[dict], track_render_meta: list,
) -> list[dict]:
    rows = []
    for stem in rendered_stem_rows:
        j = int(stem["track"])
        meta = track_render_meta[j] if j < len(track_render_meta) else {}
        rows.append({
            "path": path_output,
            "track": j,
            "original_track": meta.get("original_track", j),
            "program": stem["program"],
            "is_drum": stem["is_drum"],
            "name": stem["name"],
            "backend": meta.get("ddsp_backend") or "soundfont",
            "instrument_key": meta.get("ddsp_instrument_key"),
            "reason": meta.get("ddsp_reason"),
            "n_notes": meta.get("n_notes"),
            "source": "rendered",
            "original_path": None,
        })
    return rows


def _hybrid_pass_result(
    *,
    path_output: str,
    args,
    rendered_stem_rows: list[dict],
    recipe_rows: list[dict],
    track_render_meta: list,
) -> tuple[dict | None, list[dict], list[dict], list[dict], int]:
    """Return this pass's new table rows. Canonical data.csv is built at merge."""
    routing = (
        _hybrid_routing_rows(path_output, rendered_stem_rows, track_render_meta)
        if _needs_ddsp_routing(args)
        else []
    )
    return None, rendered_stem_rows, routing, recipe_rows, len(rendered_stem_rows)


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="Synthesize",
        description="Synthesize PDMX stems; pass --full for all valid songs, --realify for SA3.",
    )
    add_synthesis_args(parser)
    return parser.parse_args(args=args, namespace=namespace)


def song_output_dir(
    output_dir: str,
    original_dataset_dir: str,
    json_path: str,
    *,
    tree_dir_name: str = DATA_DIR_NAME,
) -> str:
    """Map a PDMX ``data/…/Qm.json`` path to a stem directory under ``output_dir``.

    Hybrid production uses ``tree_dir_name="raw"`` so pre-mix stems are
    ``{SPDMX}/raw/…`` (mix writes the released tree under ``audio/``).
    """
    rel = json_path[len(original_dataset_dir):]
    rel_no_ext = ".".join(rel.split(".")[:-1])
    if tree_dir_name != DATA_DIR_NAME:
        old = f"/{DATA_DIR_NAME}/"
        new = f"/{tree_dir_name}/"
        if old in rel_no_ext:
            rel_no_ext = rel_no_ext.replace(old, new, 1)
    return f"{output_dir}{rel_no_ext}"


def render_tree_dir_name(args) -> str:
    return SPDMX_RAW_DIR_NAME if _hybrid_recipe(args) is not None else DATA_DIR_NAME


def songs_missing_routing(songs: pd.DataFrame, routing: pd.DataFrame) -> set[str]:
    """Return song paths lacking routing coverage for tracks ``0..n_tracks-1``.

    Uses subset checks so a larger routing track set than ``n_tracks`` (e.g. dense
    vs legacy metadata mismatch) still counts as complete.
    """
    if songs.empty:
        return set()
    if routing is None or routing.empty or "path" not in routing.columns:
        return set(songs["path"].astype(str))
    by_path: dict[str, set[int]] = {}
    for path, group in routing.groupby(routing["path"].astype(str), sort=False):
        by_path[str(path)] = {int(t) for t in group["track"]}
    missing: set[str] = set()
    for _, row in songs.iterrows():
        path = str(row["path"])
        n_tracks = int(row["n_tracks"])
        if not set(range(n_tracks)).issubset(by_path.get(path, set())):
            missing.add(path)
    return missing


def load_completed_song_paths(
    data_csv: str | Path,
    *,
    routing_csv: str | Path | None = None,
) -> set[str]:
    """Paths listed in ``data.csv``, excluding DDSP songs with incomplete routing coverage."""
    data_csv = Path(data_csv)
    if not data_csv.is_file():
        return set()
    songs = pd.read_csv(data_csv, sep=",", header=0, index_col=False)
    if songs.empty or "path" not in songs.columns:
        return set()
    completed = set(songs["path"].astype(str))
    if routing_csv is None:
        return completed
    routing_path = Path(routing_csv)
    if not routing_path.is_file():
        # DDSP mode expects routing; treat all as incomplete until the file exists.
        return set()
    routing = pd.read_csv(routing_path, sep=",", header=0, index_col=False)
    return completed - songs_missing_routing(songs, routing)


def synthesize_song_at_index(
    i: int,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    args,
) -> tuple[dict | None, list[dict], list[dict]]:
    """Synthesize one song. Returns (song_row, stem_rows, ddsp_routing_rows).

    For DDSP render modes, ``args.ddsp_pass`` selects a global phase:
    ``ddsp_piano`` / ``midi_ddsp`` only render that neural backend; ``finalize``
    fills donor/soundfont stems and CSV rows (ablation ``--render-mode ddsp_*``).
    Hybrid ``synthesis.final`` uses ``fluidsynth`` and the two neural passes in
    parallel. Mix merges per-pass CSVs into canonical tables.
    """
    from synthesis.dense_midi import resolve_synthesis_midi

    path_output = dataset.at[i, "path_output"]
    song_dir = Path(path_output)
    audio_format = synthesis_audio_format(args.flac)
    ddsp_pass = getattr(args, "ddsp_pass", None)
    recipe = _hybrid_recipe(args)
    hybrid = recipe is not None

    pdmx_mid = dataset.at[i, "mid_pdmx"] if "mid_pdmx" in dataset.columns else dataset.at[i, "mid"]
    pdmx_root = dirname(args.dataset_filepath)
    midi_path, track_map = resolve_synthesis_midi(
        pdmx_mid, args=args, pdmx_root=pdmx_root,
    )
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    n_tracks = len(track_map)

    if hybrid and not args.reset and _hybrid_song_raw_current(
        args, path_output, n_tracks, song_dir, audio_format, recipe,
    ):
        # CSV-only resume: recipe rows cover every track (disk optional).
        if not _resume_check_disk(args) or song_is_complete(
            song_dir, n_tracks, audio_format, require_mixture=False,
        ):
            del midi
            return None, [], [], [], 0
    if (
        not hybrid
        and path_output in completed_paths
        and song_is_complete(song_dir, n_tracks, audio_format, require_mixture=False)
        and not args.reset
    ):
        del midi
        return None, [], [], [], 0
    stems_complete = all(
        stem_is_valid(stem_path(song_dir, j, audio_format)) for j in range(n_tracks)
    )
    # Phased hybrid / DDSP passes enter the render block even when stems exist so
    # they can skip valid files and still emit CSV rows when the song is complete.
    need_to_synthesize = args.reset or not stems_complete
    if (
        uses_ddsp(getattr(args, "render_mode", "") or "")
        or hybrid
    ) and ddsp_pass in (
        "fluidsynth", "ddsp_piano", "midi_ddsp", "finalize",
    ):
        need_to_synthesize = True
    stem_rows: list[dict] = []
    routing_rows: list[dict] = []
    recipe_rows: list[dict] = []

    if need_to_synthesize:
        temp_dir = tempfile.TemporaryDirectory()
        track_paths = [f"{temp_dir.name}/{j}.mid" for j in range(n_tracks)]
        track_render_meta: list[dict] = []

    tracks_to_render = [
        (j, midi.tracks[j])
        for j in sorted(track_map.keys())
        if j < len(midi.tracks)
    ]

    for j, track in tracks_to_render:
        if need_to_synthesize:
            track_midi = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat, charset="utf8")
            track_midi_track = mido.MidiTrack()

        program = 0
        is_drum = False
        track_name = None
        has_lyrics = False
        n_notes = 0
        max_velocity = 0
        original_track = int(track_map[j]["original_track"])

        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                n_notes += 1
                max_velocity = max(max_velocity, int(message.velocity))
                if getattr(message, "channel", None) == 9:
                    is_drum = True
            elif message.type == "program_change":
                program = message.program
            elif message.type == "track_name":
                track_name = sanitize_track_name(
                    " ".join(message.name.replace(",", " ").split())
                )
            elif message.type == "lyrics":
                has_lyrics = True
            if need_to_synthesize and n_notes <= MAX_N_NOTES_IN_STEM:
                track_midi_track.append(message)

        # Dense corrected midis already bake register programs / drum flags.
        program = int(track_map[j].get("program", program))
        if bool(track_map[j].get("is_drum", False)):
            is_drum = True
        from synthesis.ddsp.routing import has_positive_duration_notes

        zero_duration_notes = bool(n_notes > 0 and not has_positive_duration_notes(track))

        if need_to_synthesize:
            track_midi.tracks.append(track_midi_track)
            slakh_cfg: dict = {}
            plan = None
            if recipe is not None:
                plan = recipe.plan_for_track(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                )
            use_slakh = (
                plan.use_slakh if plan is not None
                else uses_slakh_recipes(args.render_mode)
            )
            if use_slakh:
                from synthesis.patches import (
                    select_patch,
                    slakh_render_for_track,
                )

                slakh_cfg = slakh_render_for_track(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                )
                apply_patch_to_midi_track(
                    track_midi_track,
                    select_patch(
                        program=program,
                        is_drum=is_drum,
                        pool_id=None,
                        category=slakh_cfg.get("category"),
                    ),
                )
            track_midi.save(track_paths[j])
            route_meta: dict = {}
            should_route = uses_ddsp(getattr(args, "render_mode", "") or "") or (
                plan is not None and plan.neural_ok
            )
            if should_route:
                from synthesis.ddsp.routing import (
                    BACKEND_DDSP_PIANO,
                    BACKEND_SOUNDFONT,
                    StemRoute,
                    route_stem,
                )

                route = route_stem(
                    program=program,
                    is_drum=is_drum,
                    track_name=track_name,
                    track=track_midi_track,
                    ticks_per_beat=midi.ticks_per_beat,
                    check_monophony=True,
                )
                if (
                    hybrid
                    and route.backend == BACKEND_DDSP_PIANO
                    and not recipe.uses_ddsp_piano()
                ):
                    route = StemRoute(
                        BACKEND_SOUNDFONT, None, "piano_recipe_not_midi_ddsp",
                    )
                route_meta = {
                    "ddsp_backend": route.backend,
                    "ddsp_instrument_key": route.instrument_key,
                    "ddsp_reason": route.reason,
                    "n_notes": n_notes,
                    "original_track": original_track,
                }
            track_render_meta.append({
                "soundfont_filepath": args.soundfont_filepath,
                "fx_profile": None,
                "original_track": original_track,
                "use_slakh": use_slakh,
                "neural_ok": bool(plan.neural_ok) if plan is not None else False,
                "listening_category": plan.category if plan is not None else None,
                "plan": plan,
                **slakh_cfg,
                **route_meta,
            })

        stem_rows.append({
            "path": path_output,
            "track": j,
            "original_track": original_track,
            "program": program,
            "is_drum": is_drum,
            "name": track_name if track_name and len(track_name) > 0 else None,
            "has_lyrics": has_lyrics,
            "max_velocity": max_velocity,
            "velocity_scale": None,  # filled after all tracks
            "zero_duration_notes": zero_duration_notes,
        })

    from synthesis.velocity import velocity_scales_from_track_maxima

    track_maxima = {int(row["track"]): int(row["max_velocity"]) for row in stem_rows}
    scales = velocity_scales_from_track_maxima(track_maxima)
    for row in stem_rows:
        row["velocity_scale"] = scales.get(int(row["track"]), 1.0)

    del midi

    if need_to_synthesize:
        donor_mode = fallback_donor_mode(getattr(args, "render_mode", None))
        song_rel = None
        if (
            not hybrid
            and uses_ddsp(getattr(args, "render_mode", "") or "")
            and donor_mode is not None
        ):
            song_rel = song_rel_under_data(
                song_dir,
                ablation_raw_dir(args.output_dir, args.render_mode),
            )

        if hybrid and ddsp_pass == "fluidsynth":
            from synthesis.ddsp.routing import (
                midi_path_has_positive_duration_notes,
                midi_path_uses_drum_channel,
            )
            from synthesis.recipe import (
                BACKEND_FLUIDSYNTH,
                BACKEND_PENDING_MIDI_DDSP,
            )

            rendered_stem_rows: list[dict] = []
            recipe_index = getattr(args, "stem_recipe_index", None) or {}
            for j, track_path in enumerate(track_paths):
                meta = track_render_meta[j]
                plan = meta.get("plan")
                out_stem = stem_path(song_dir, j, audio_format)
                backend = meta.get("ddsp_backend")
                leave_for_neural = (
                    meta.get("neural_ok")
                    and backend in ("midi_ddsp", "ddsp_piano")
                    and not midi_path_uses_drum_channel(track_path)
                    # Zero-duration notes → Fluidsynth (never pending for MIDI-DDSP).
                    and midi_path_has_positive_duration_notes(track_path)
                )
                claimed = recipe_index.get((str(path_output), j))
                # Already claimed this pass (native SF, neural→SF, or deferred)?
                if claimed is not None and not args.reset:
                    claimed_backend = str(claimed.get("backend") or "")
                    # Reclaim pending rows when routing now prefers soundfont
                    # (e.g. zero-duration notes that MIDI-DDSP cannot render).
                    if (
                        claimed_backend == BACKEND_PENDING_MIDI_DDSP
                        and not leave_for_neural
                    ):
                        pass  # fall through to SF-render
                    else:
                        continue
                if leave_for_neural:
                    # Bookkeeping only — MIDI-DDSP / DDSP-Piano owns the audio.
                    if plan is not None:
                        recipe_rows.append(plan.sidecar_row(
                            path=path_output,
                            track=j,
                            backend=BACKEND_PENDING_MIDI_DDSP,
                            reason=meta.get("ddsp_reason") or "midi_ddsp_eligible",
                        ))
                    continue
                if (
                    plan is not None
                    and _hybrid_raw_current(
                        args, path_output, j, out_stem, plan, BACKEND_FLUIDSYNTH,
                    )
                ):
                    continue
                if stem_is_valid(out_stem) and not args.reset and plan is None:
                    continue
                waveform = _render_soundfont_stem(
                    track_path, meta, args, path_output,
                )
                save_stem(waveform, song_dir, j, audio_format)
                rendered_stem_rows.append(stem_rows[j])
                if plan is not None:
                    recipe_rows.append(plan.sidecar_row(
                        path=path_output, track=j, backend=BACKEND_FLUIDSYNTH,
                        reason=meta.get("ddsp_reason"),
                    ))
            for path in track_paths:
                if exists(path):
                    remove(path)
            temp_dir.cleanup()
            return _hybrid_pass_result(
                path_output=path_output,
                args=args,
                rendered_stem_rows=rendered_stem_rows,
                recipe_rows=recipe_rows,
                track_render_meta=track_render_meta,
            )

        ddsp_like = uses_ddsp(getattr(args, "render_mode", "") or "") or (
            hybrid and ddsp_pass in ("ddsp_piano", "midi_ddsp", "finalize")
        )
        if ddsp_like:
            from synthesis.ddsp.pool import ddsp_oneshot_enabled, get_ddsp_pool
            from synthesis.ddsp.routing import StemRoute
            from synthesis.ddsp.synthesize import synthesize_stem_neural

            # Global two-pass: neural phases only render one backend; finalize does the rest.
            # Fluidsynth owns all soundfont fallbacks — never SF-render here.
            if ddsp_pass in ("ddsp_piano", "midi_ddsp"):
                neural_jobs: list[tuple[int, str, StemRoute]] = []
                rendered_stem_rows: list[dict] = []
                pending_keys = getattr(args, "_pending_midi_ddsp_keys", None) or frozenset()
                path_key = os.path.normpath(str(path_output))
                for j, track_path in enumerate(track_paths):
                    meta = track_render_meta[j]
                    is_pending = (
                        (path_key, j) in pending_keys
                        or (str(path_output), j) in pending_keys
                    )
                    # Fluidsynth deferred these as pending_midi_ddsp — always try neural.
                    if hybrid and not meta.get("neural_ok") and not is_pending:
                        continue
                    if (
                        meta.get("ddsp_backend") != ddsp_pass
                        and not is_pending
                    ):
                        continue
                    out_stem = stem_path(song_dir, j, audio_format)
                    plan = meta.get("plan")
                    if (
                        plan is not None
                        and not is_pending
                        and _hybrid_raw_current(
                            args, path_output, j, out_stem, plan, ddsp_pass,
                        )
                    ):
                        continue
                    if (
                        stem_is_valid(out_stem)
                        and not args.reset
                        and plan is None
                        and not is_pending
                    ):
                        continue
                    # Defense: never send channel-9 / PrettyMIDI-drum stems to neural.
                    # Same for zero-duration-only MIDI (PrettyMIDI end_time=0).
                    # Leave unclaimed so Fluidsynth (re)claims them as SF fallbacks.
                    from synthesis.ddsp.routing import (
                        midi_path_has_positive_duration_notes,
                        midi_path_uses_drum_channel,
                    )

                    if midi_path_uses_drum_channel(track_path):
                        continue
                    if not midi_path_has_positive_duration_notes(track_path):
                        continue
                    backend = meta.get("ddsp_backend") or ddsp_pass
                    if is_pending and backend != ddsp_pass:
                        backend = ddsp_pass
                    neural_jobs.append((
                        j,
                        track_path,
                        StemRoute(
                            backend=backend,
                            instrument_key=meta.get("ddsp_instrument_key"),
                            reason=meta.get("ddsp_reason") or (
                                "midi_ddsp_eligible" if is_pending else ""
                            ),
                        ),
                    ))

                if neural_jobs:
                    def _neural_one(job: tuple[int, str, StemRoute]):
                        idx, mid_path, route = job
                        try:
                            return idx, synthesize_stem_neural(mid_path, route), None
                        except Exception as exc:
                            if "Cannot synthesize drum" not in str(exc):
                                raise
                            # Leave for Fluidsynth — do not SF-render in this pass.
                            return idx, None, exc

                    inner = int(getattr(args, "neural_inner_workers", 0) or 0)
                    if ddsp_oneshot_enabled():
                        max_workers = 1
                    elif inner > 0:
                        max_workers = inner
                    else:
                        max_workers = max(1, get_ddsp_pool().size)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(_neural_one, job) for job in neural_jobs
                        ]
                        for fut in as_completed(futures):
                            idx, waveform, drum_exc = fut.result()
                            if waveform is None:
                                from synthesis.job_log import get_job_log

                                log = get_job_log()
                                if log is not None:
                                    log.warn(
                                        f"{ddsp_pass} drum reject; leaving for Fluidsynth "
                                        f"track={idx} error={drum_exc}"
                                    )
                                continue
                            meta = track_render_meta[idx]
                            plan = meta.get("plan")
                            save_stem(waveform, song_dir, idx, audio_format)
                            rendered_stem_rows.append(stem_rows[idx])
                            if plan is not None:
                                recipe_rows.append(plan.sidecar_row(
                                    path=path_output,
                                    track=idx,
                                    backend=ddsp_pass,
                                    reason=meta.get("ddsp_reason") or "midi_ddsp_eligible",
                                ))

                for path in track_paths:
                    if exists(path):
                        remove(path)
                temp_dir.cleanup()
                return _hybrid_pass_result(
                    path_output=path_output,
                    args=args,
                    rendered_stem_rows=rendered_stem_rows,
                    recipe_rows=recipe_rows,
                    track_render_meta=track_render_meta,
                )

            # Finalize (default when ddsp_pass is None or "finalize"): non-neural stems.
            for j, track_path in enumerate(track_paths):
                meta = track_render_meta[j]
                backend = meta.get("ddsp_backend")
                out_stem = stem_path(song_dir, j, audio_format)
                source = "rendered"
                original_path = None
                neural_track = backend in ("midi_ddsp", "ddsp_piano") and (
                    not hybrid or meta.get("neural_ok")
                )

                if neural_track:
                    if not stem_is_valid(out_stem):
                        for path in track_paths:
                            if exists(path):
                                remove(path)
                        temp_dir.cleanup()
                        raise RuntimeError(
                            f"Missing neural DDSP stem after neural passes: {out_stem}\n"
                            f"backend={backend} song={path_output} track={j}"
                        )
                elif hybrid:
                    if not stem_is_valid(out_stem):
                        for path in track_paths:
                            if exists(path):
                                remove(path)
                        temp_dir.cleanup()
                        raise RuntimeError(
                            f"Missing Fluidsynth stem after hybrid passes: {out_stem}\n"
                            f"song={path_output} track={j}"
                        )
                elif stem_is_valid(out_stem) and not args.reset:
                    pass
                elif donor_mode is not None and song_rel is not None:
                    donor_stem = donor_raw_stem_path(
                        args.output_dir,
                        donor_mode,
                        song_rel,
                        j,
                        audio_format,
                    )
                    orig_track = int(meta.get("original_track", j))
                    if not stem_is_valid(donor_stem) and orig_track != j:
                        alt = donor_raw_stem_path(
                            args.output_dir,
                            donor_mode,
                            song_rel,
                            orig_track,
                            audio_format,
                        )
                        if stem_is_valid(alt):
                            donor_stem = alt
                    if stem_is_valid(donor_stem):
                        copy_stem(donor_stem, out_stem)
                        source = reused_source_label(donor_mode)
                        original_path = str(donor_stem.resolve())
                    elif getattr(args, "allow_fallback_render", False):
                        waveform = _render_soundfont_stem(
                            track_path, meta, args, path_output,
                        )
                        save_stem(waveform, song_dir, j, audio_format)
                    else:
                        for path in track_paths:
                            if exists(path):
                                remove(path)
                        temp_dir.cleanup()
                        raise RuntimeError(
                            f"Missing donor stem for DDSP fallback: {donor_stem}\n"
                            f"Generate the donor ablation first:\n"
                            f"  uv run python -m synthesis.synthesize --render-mode {donor_mode}\n"
                            "Or pass --allow-fallback-render to Fluidsynth-render missing donors."
                        )
                else:
                    waveform = _render_soundfont_stem(
                        track_path, meta, args, path_output,
                    )
                    save_stem(waveform, song_dir, j, audio_format)

                if hybrid and not _needs_ddsp_routing(args):
                    plan = meta.get("plan")
                    if plan is not None:
                        from synthesis.recipe import resolve_track_backend
                        from synthesis.ddsp.routing import StemRoute

                        backend_name = resolve_track_backend(
                            plan,
                            StemRoute(
                                backend=meta.get("ddsp_backend") or "soundfont",
                                instrument_key=meta.get("ddsp_instrument_key"),
                                reason=meta.get("ddsp_reason") or "",
                            ),
                        )
                        recipe_rows.append(plan.sidecar_row(
                            path=path_output, track=j, backend=backend_name,
                            reason=meta.get("ddsp_reason"),
                        ))
                    remove(track_path)
                    continue
                routing_rows.append({
                    "path": path_output,
                    "track": j,
                    "original_track": meta.get("original_track", j),
                    "program": stem_rows[j]["program"],
                    "is_drum": stem_rows[j]["is_drum"],
                    "name": stem_rows[j]["name"],
                    "backend": meta.get("ddsp_backend") or "soundfont",
                    "instrument_key": meta.get("ddsp_instrument_key"),
                    "reason": meta.get("ddsp_reason"),
                    "n_notes": meta.get("n_notes"),
                    "source": source,
                    "original_path": original_path,
                })
                plan = meta.get("plan")
                if plan is not None:
                    from synthesis.recipe import resolve_track_backend
                    from synthesis.ddsp.routing import StemRoute

                    backend_name = resolve_track_backend(
                        plan,
                        StemRoute(
                            backend=meta.get("ddsp_backend") or "soundfont",
                            instrument_key=meta.get("ddsp_instrument_key"),
                            reason=meta.get("ddsp_reason") or "",
                        ),
                    )
                    recipe_rows.append(plan.sidecar_row(
                        path=path_output, track=j, backend=backend_name,
                        reason=meta.get("ddsp_reason"),
                    ))
                remove(track_path)
            temp_dir.cleanup()
        else:
            waveforms = []
            for j, track_path in enumerate(track_paths):
                meta = track_render_meta[j]
                waveforms.append(
                    _render_soundfont_stem(track_path, meta, args, path_output)
                )
                remove(track_path)
            temp_dir.cleanup()
            for j, waveform in enumerate(waveforms):
                save_stem(waveform, song_dir, j, audio_format)

    song_info = _song_table_row(dataset, i, path_output, n_tracks)
    return song_info, stem_rows, routing_rows, recipe_rows, 1


def _render_soundfont_stem(track_path: str, meta: dict, args, path_output: str):
    soundfont_filepath = meta.get("soundfont_filepath") or args.soundfont_filepath
    fx_profile = meta.get("fx_profile")
    if meta.get("use_slakh", uses_slakh_recipes(getattr(args, "render_mode", "") or "")):
        from experiments.patch_sweep.config import soundfont_file_for_id
        from experiments.patch_sweep.winners import pick_fx_profile, pick_soundfont_id

        soundfont_ids = meta.get("soundfont_ids") or []
        if not soundfont_ids and meta.get("soundfont_id"):
            soundfont_ids = [meta["soundfont_id"]]
        category = meta.get("category") or "default"
        if soundfont_ids:
            picked = pick_soundfont_id(
                list(soundfont_ids),
                category=category,
                song_path=path_output,
                sample_seed=args.sample_seed,
            )
            soundfont_filepath = str(
                Path(SOUNDFONT_DIR) / soundfont_file_for_id(picked)
            )
        elif meta.get("soundfont"):
            soundfont_filepath = str(Path(SOUNDFONT_DIR) / meta["soundfont"])

        fx_profiles = meta.get("fx_profiles") or []
        if not fx_profiles and meta.get("fx_profile"):
            fx_profiles = [meta["fx_profile"]]
        if fx_profiles:
            fx_profile = pick_fx_profile(
                list(fx_profiles),
                category=category,
                song_path=path_output,
                sample_seed=args.sample_seed,
            )
    elif meta.get("soundfont"):
        soundfont_filepath = str(Path(SOUNDFONT_DIR) / meta["soundfont"])
    return get_waveform_tensor(
        track_path,
        soundfont_filepath,
        fx_profile=fx_profile,
    )


_WORKER_CTX: dict = {}
_WORKER_SONG_COUNT = 0


def _resolved_recipe(args):
    recipe = _hybrid_recipe(args)
    if recipe is None:
        return None
    if isinstance(recipe, (str, Path)):
        from synthesis.recipe import load_recipe

        return load_recipe(recipe)
    return recipe


def _recipe_done_by_path(stem_recipe_index: dict | None, pass_name: str) -> dict[str, int]:
    """Count tracks finished in this pass's stem_recipe index.

    The index is loaded from a pass-scoped CSV (``stem_recipe.<pass>.csv``), so
    every recorded track counts — including soundfont/fluidsynth redirects written
    when a neural pass re-routes a mis-indexed drum or similar.
    """
    done: dict[str, int] = {}
    if not stem_recipe_index:
        return done
    for (path, _track), rec in stem_recipe_index.items():
        if not rec:
            continue
        key = str(path)
        done[key] = done.get(key, 0) + 1
    return done


def _fluidsynth_recipe_frame(tables_dir: str | Path) -> pd.DataFrame:
    from shared.csv_tables import read_csv_flexible
    from synthesis.pass_tables import pass_recipe_csv
    from synthesis.recipe import STEM_RECIPE_COLUMNS

    return read_csv_flexible(
        pass_recipe_csv(tables_dir, "fluidsynth"),
        columns=STEM_RECIPE_COLUMNS,
    )


# Fluidsynth soundfont renders of layout-MIDI-DDSP stems (any render-time reject).
# Excludes pending_midi_ddsp (no audio). After reason backfill, blank/NA is rare.
_MIDI_DDSP_SOUNDFONT_FALLBACK_REASONS = frozenset({
    "soundfont_polyphonic",
    "soundfont_unsupported",
    "soundfont_drum",
    "soundfont_vocal",
    "soundfont_guitar",
    "soundfont_bass_guitar",
    "soundfont_zero_duration_notes",
    # Legacy reason string from the one-shot backfill (same meaning).
    "soundfont_zero_duration",
    "piano_recipe_not_midi_ddsp",
    "drum_fluidsynth_fallback",
    "empty_track",
})


def _is_fluidsynth_midi_ddsp_fallback_row(method, backend, reason=None) -> bool:
    """True for layout-MIDI-DDSP stems Fluidsynth soundfont-rendered as fallbacks.

    Any ``method=midi-ddsp`` + ``backend=fluidsynth`` row with a soundfont-style
    reason (or blank/legacy ``NA``) counts. ``pending_midi_ddsp`` does not.
    """
    from synthesis.recipe import BACKEND_FLUIDSYNTH, BACKEND_PENDING_MIDI_DDSP, METHOD_MIDI_DDSP

    if str(method) != METHOD_MIDI_DDSP:
        return False
    backend_s = "" if backend is None else str(backend)
    if backend_s in ("", "nan", "None"):
        backend_s = BACKEND_FLUIDSYNTH
    if backend_s == BACKEND_PENDING_MIDI_DDSP:
        return False
    if backend_s != BACKEND_FLUIDSYNTH:
        return False
    # Legacy rows without reason (including CSV na_rep "NA"): keep crediting.
    reason_s = "" if reason is None else str(reason).strip()
    if reason_s in ("", "nan", "None", "NA", "<NA>"):
        return True
    return reason_s in _MIDI_DDSP_SOUNDFONT_FALLBACK_REASONS


def _fluidsynth_midi_ddsp_fallback_counts_by_path(
    tables_dir: str | Path,
) -> dict[str, int]:
    """``path`` → Fluidsynth recipe rows that are real neural→soundfont fallbacks.

    Layout assigns these to ``midi_ddsp``; Fluidsynth renders them (polyphony,
    drum, unsupported, etc.). Excludes ``pending_midi_ddsp`` bookkeeping rows.
    """
    df = _fluidsynth_recipe_frame(tables_dir)
    if df.empty or "path" not in df.columns or "method" not in df.columns:
        return {}
    methods = df["method"].astype(str)
    backends = (
        df["backend"] if "backend" in df.columns
        else pd.Series([None] * len(df))
    )
    reasons = (
        df["reason"] if "reason" in df.columns
        else pd.Series([None] * len(df))
    )
    mask = [
        _is_fluidsynth_midi_ddsp_fallback_row(m, b, r)
        for m, b, r in zip(methods, backends, reasons, strict=False)
    ]
    if not any(mask):
        return {}
    return df.loc[mask, "path"].astype(str).value_counts().to_dict()


def _fluidsynth_pass_credits(
    stem_recipe_index: dict | None,
    *,
    n_midi_ddsp_by_path: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-path ``(native_done, neural_handled)`` for Fluidsynth resume / bar.

    Neural-category stems Fluidsynth soundfont-renders are stored as
    ``method=midi-ddsp`` + ``backend=fluidsynth``. Layout may put those in
    ``n_fluidsynth`` (soundfont at layout) *or* ``n_midi_ddsp`` (mono layout,
    SF at render). Attribute ``method=midi-ddsp`` SF rows to ``n_midi_ddsp``
    first; any overflow counts toward ``n_fluidsynth`` so the bar does not stay
    stuck on already-rendered layout-Fluidsynth tracks.
    """
    from synthesis.recipe import (
        BACKEND_FLUIDSYNTH,
        BACKEND_PENDING_MIDI_DDSP,
        METHOD_MIDI_DDSP,
    )

    basic: dict[str, int] = {}
    md_sf: dict[str, int] = {}
    pending: dict[str, int] = {}
    if stem_recipe_index:
        for (path, _track), rec in stem_recipe_index.items():
            if not rec:
                continue
            key = str(path)
            method = str(rec.get("method") or "")
            backend = str(rec.get("backend") or "")
            reason = rec.get("reason")
            if method == METHOD_MIDI_DDSP:
                if backend == BACKEND_PENDING_MIDI_DDSP:
                    pending[key] = pending.get(key, 0) + 1
                elif _is_fluidsynth_midi_ddsp_fallback_row(method, backend, reason):
                    md_sf[key] = md_sf.get(key, 0) + 1
                elif backend in ("", BACKEND_FLUIDSYNTH, "nan", "None"):
                    # Neural-category layout-Fluidsynth SF (unsupported, guitar, …).
                    basic[key] = basic.get(key, 0) + 1
                else:
                    pending[key] = pending.get(key, 0) + 1
            else:
                basic[key] = basic.get(key, 0) + 1

    native_done: dict[str, int] = {}
    neural_handled: dict[str, int] = {}
    paths = set(basic) | set(md_sf) | set(pending) | set(n_midi_ddsp_by_path)
    for path in paths:
        n_md = int(n_midi_ddsp_by_path.get(path, 0))
        sf = int(md_sf.get(path, 0))
        pend = int(pending.get(path, 0))
        # SF rows fill layout-MIDI-DDSP fallback slots first.
        md_sf_credit = min(sf, n_md)
        native_done[path] = int(basic.get(path, 0)) + max(0, sf - n_md)
        neural_handled[path] = pend + md_sf_credit
    return native_done, neural_handled


def _pass_recipe_counts_by_path(
    tables_dir: str | Path, pass_name: str,
) -> dict[str, int]:
    """``path`` → row counts in ``stem_recipe.<pass>.csv``."""
    from synthesis.pass_tables import pass_recipe_csv
    from synthesis.recipe import load_stem_recipe_index

    path = pass_recipe_csv(tables_dir, pass_name)
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    return _recipe_done_by_path(
        load_stem_recipe_index(tables_dir, filename=path.name),
        pass_name,
    )


def neural_fluidsynth_fallback_reason_counts(
    tables_dir: str | Path,
) -> dict[str, int]:
    """Reason → count for Fluidsynth-rendered stems whose recipe method is midi-ddsp.

    Prefer recipe ``reason`` when present. Excludes ``pending_midi_ddsp`` (no audio).
    """
    from synthesis.pass_tables import pass_routing_csv

    recipe = _fluidsynth_recipe_frame(tables_dir)
    if recipe.empty or "method" not in recipe.columns:
        return {}
    methods = recipe["method"].astype(str)
    backends = (
        recipe["backend"] if "backend" in recipe.columns
        else pd.Series([None] * len(recipe))
    )
    reasons = (
        recipe["reason"] if "reason" in recipe.columns
        else pd.Series([None] * len(recipe))
    )
    mask = [
        _is_fluidsynth_midi_ddsp_fallback_row(m, b, r)
        for m, b, r in zip(methods, backends, reasons, strict=False)
    ]
    fb = recipe.loc[mask].copy()
    if fb.empty:
        return {}
    if "reason" in fb.columns and fb["reason"].notna().any():
        from_recipe = (
            fb["reason"].fillna("").astype(str).replace("", pd.NA).dropna()
        )
        if len(from_recipe) == len(fb):
            return from_recipe.value_counts().to_dict()
    routing_path = pass_routing_csv(tables_dir, "fluidsynth")
    if routing_path.is_file() and routing_path.stat().st_size > 0:
        routing = pd.read_csv(
            routing_path, usecols=lambda c: c in ("path", "track", "reason"),
        )
        if not routing.empty and "reason" in routing.columns:
            merged = fb.merge(routing, on=["path", "track"], how="left")
            reasons = merged["reason"].fillna("(unspecified)").astype(str)
            return reasons.value_counts().to_dict()
    return {"(unspecified)": len(fb)}


def _fluidsynth_pending_unclaimed_keys(
    tables_dir: str | Path,
    stem_recipe_index: dict | None = None,
) -> set[tuple[str, int]]:
    """``(path, track)`` Fluidsynth ``pending_midi_ddsp`` rows with no MIDI-DDSP recipe."""
    from synthesis.recipe import BACKEND_PENDING_MIDI_DDSP, METHOD_MIDI_DDSP

    df = _fluidsynth_recipe_frame(tables_dir)
    if df.empty or "path" not in df.columns or "method" not in df.columns:
        return set()
    if "backend" not in df.columns or "track" not in df.columns:
        return set()
    mask = (
        (df["method"].astype(str) == METHOD_MIDI_DDSP)
        & (df["backend"].astype(str) == BACKEND_PENDING_MIDI_DDSP)
    )
    if not mask.any():
        return set()
    neural = stem_recipe_index or {}
    keys: set[tuple[str, int]] = set()
    for row in df.loc[mask, ["path", "track"]].itertuples(index=False):
        path = os.path.normpath(str(row.path))
        track = int(row.track)
        # Match either normalized or raw recipe path against neural index keys.
        if (path, track) in neural or (str(row.path), track) in neural:
            continue
        keys.add((path, track))
    return keys


def _fluidsynth_pending_unclaimed_counts_by_path(
    tables_dir: str | Path,
    stem_recipe_index: dict | None = None,
) -> dict[str, int]:
    """``path`` → Fluidsynth ``pending_midi_ddsp`` tracks with no MIDI-DDSP recipe row.

    Song-level SF-fallback credits can mask these (no audio yet). Verify / resume
    must still treat them as MIDI-DDSP work.
    """
    counts: dict[str, int] = {}
    for path, _track in _fluidsynth_pending_unclaimed_keys(
        tables_dir, stem_recipe_index,
    ):
        counts[path] = counts.get(path, 0) + 1
    return counts


def _work_done_by_path(
    pass_name: str,
    *,
    stem_recipe_index: dict | None,
    tables_dir: str | Path | None = None,
) -> dict[str, int]:
    """Per-path finished-track counts for resume / progress of one pass."""
    done = _recipe_done_by_path(stem_recipe_index, pass_name)
    if pass_name != "midi_ddsp" or tables_dir is None:
        return done
    # Normalize keys so recipe paths match dataset path_output.
    normalized: dict[str, int] = {}
    for path, n in done.items():
        key = os.path.normpath(str(path))
        normalized[key] = normalized.get(key, 0) + int(n)
    for path, n in _fluidsynth_midi_ddsp_fallback_counts_by_path(tables_dir).items():
        key = os.path.normpath(str(path))
        normalized[key] = normalized.get(key, 0) + int(n)
    return normalized


def _midi_ddsp_remaining_for_path(
    *,
    assigned: int,
    done: int,
    pending_unclaimed: int,
) -> int:
    """Tracks still owed by MIDI-DDSP (neural), including unclaimed Fluidsynth pendings."""
    return max(max(0, int(assigned) - int(done)), int(pending_unclaimed))


def _reload_progress_from_disk(
    args,
    *,
    output_filepath: str,
    routing_output_filepath: str | None,
    recipe_output_filepath: str | None,
    completed_paths: set[str],
) -> set[str]:
    """Reload pass tables and completed-song sets from shared storage."""
    if not args.reset and Path(output_filepath).is_file():
        routing_for_completed = (
            routing_output_filepath if _needs_ddsp_routing(args) else None
        )
        completed_paths = load_completed_song_paths(
            output_filepath,
            routing_csv=routing_for_completed,
        )
    ddsp_pass = getattr(args, "ddsp_pass", None)
    if recipe_output_filepath and ddsp_pass in (
        "fluidsynth", "ddsp_piano", "midi_ddsp",
    ):
        from synthesis.recipe import load_stem_recipe_index

        recipe_path = Path(recipe_output_filepath)
        args.stem_recipe_index = load_stem_recipe_index(
            recipe_path.parent,
            filename=recipe_path.name,
        )
    return completed_paths


def _song_index_needs_work(
    i: int,
    dataset: pd.DataFrame,
    args,
    completed_paths: set[str],
    audio_format: str,
) -> bool:
    """False when another job already finished this song for the current pass."""
    if args.reset:
        return True
    path_output = str(dataset.at[i, "path_output"])
    n_tracks = int(dataset.at[i, "n_tracks"])
    recipe = _hybrid_recipe(args)
    ddsp_pass = getattr(args, "ddsp_pass", None)
    if recipe is not None and ddsp_pass in ("fluidsynth", "ddsp_piano", "midi_ddsp"):
        indices, _ = _work_for_pass(
            dataset,
            [i],
            ddsp_pass,
            stem_recipe_index=getattr(args, "stem_recipe_index", None),
            tables_dir=getattr(args, "_tables_dir", None),
        )
        return i in indices
    if path_output not in completed_paths:
        return True
    if not song_is_complete(
        Path(path_output), n_tracks, audio_format, require_mixture=False,
    ):
        return True
    if recipe is not None and not _hybrid_song_raw_current(
        args, path_output, n_tracks, Path(path_output), audio_format, recipe,
    ):
        return True
    return False


def _work_for_pass(
    dataset: pd.DataFrame,
    work_indices: list,
    pass_name: str,
    *,
    stem_recipe_index: dict | None = None,
    tables_dir: str | Path | None = None,
):
    """Songs and remaining tracks to *render* for one hybrid engine pass.

    Skips songs with no assigned tracks for this engine, and songs whose
    ``stem_recipe.<pass>.csv`` already covers those tracks (CSV resume).
    For ``midi_ddsp``, also credits Fluidsynth rows with ``method=midi-ddsp``
    and ``backend=fluidsynth`` (polyphony / unsupported → soundfont fallbacks).
    For ``fluidsynth``, also visits songs with unclaimed layout-MIDI-DDSP tracks
    so soundfont fallbacks are claimed here (not by the neural pass).
    Bar total is assigned tracks minus stems already recorded for that backend.
    """
    col = PASS_TRACK_COLUMNS.get(pass_name)
    if col is None or col not in dataset.columns:
        return list(work_indices), None

    if pass_name == "fluidsynth":
        md_done: dict[str, int] = {}
        if tables_dir is not None:
            md_done = _pass_recipe_counts_by_path(tables_dir, "midi_ddsp")
        n_md_by_path: dict[str, int] = {}
        for i in work_indices:
            path = (
                str(dataset.at[i, "path_output"])
                if "path_output" in dataset.columns
                else ""
            )
            if "n_midi_ddsp" in dataset.columns:
                n_md_by_path[path] = int(dataset.at[i, "n_midi_ddsp"])
        native_done, neural_handled = _fluidsynth_pass_credits(
            stem_recipe_index,
            n_midi_ddsp_by_path=n_md_by_path,
        )
        kept: list = []
        total = 0
        for i in work_indices:
            path = (
                str(dataset.at[i, "path_output"])
                if "path_output" in dataset.columns
                else ""
            )
            native_n = int(dataset.at[i, col])
            native_rem = max(0, native_n - int(native_done.get(path, 0)))
            md_n = int(n_md_by_path.get(path, 0))
            neural_rem = max(
                0,
                md_n
                - int(md_done.get(path, 0))
                - int(neural_handled.get(path, 0)),
            )
            if native_rem + neural_rem <= 0:
                continue
            kept.append(i)
            # Bar denominator: only tracks Fluidsynth will actually render.
            # Layout-MIDI-DDSP deferrals (pending_midi_ddsp) are excluded; the
            # few new SF fallbacks among them may tick slightly past 100%.
            total += native_rem
        return kept, total

    done = _work_done_by_path(
        pass_name,
        stem_recipe_index=stem_recipe_index,
        tables_dir=tables_dir,
    )
    pending_unclaimed: dict[str, int] = {}
    if pass_name == "midi_ddsp" and tables_dir is not None:
        pending_unclaimed = _fluidsynth_pending_unclaimed_counts_by_path(
            tables_dir,
            stem_recipe_index,
        )
    kept = []
    total = 0
    for i in work_indices:
        n = int(dataset.at[i, col])
        path = str(dataset.at[i, "path_output"]) if "path_output" in dataset.columns else ""
        path_key = os.path.normpath(path) if path else ""
        rem_pending = int(pending_unclaimed.get(path_key, 0)) or int(
            pending_unclaimed.get(path, 0)
        )
        if n <= 0 and rem_pending <= 0:
            continue
        remaining = (
            _midi_ddsp_remaining_for_path(
                assigned=n,
                done=int(done.get(path_key, 0)) or int(done.get(path, 0)),
                pending_unclaimed=rem_pending,
            )
            if pass_name == "midi_ddsp"
            else max(0, n - int(done.get(path_key, 0) or done.get(path, 0)))
        )
        if remaining <= 0:
            continue
        kept.append(i)
        total += remaining
    return kept, total


def _preload_track_maps(args) -> None:
    """Load SPDMX.csv track map once on the parent thread (avoid 4× NFS stampede)."""
    if getattr(args, "track_maps", None):
        return
    from analysis.corrected_midi import load_track_maps, resolve_track_map_csv

    csv_path = resolve_track_map_csv(_hybrid_corrected_midi_root(args))
    print(f"Loading track map from {csv_path} (one-time, can take a few minutes) ...", flush=True)
    args.track_maps = load_track_maps(csv_path)
    n_songs = len(args.track_maps)
    n_tracks = sum(len(v) for v in args.track_maps.values())
    print(f"Track map ready ({n_songs} songs, {n_tracks} tracks).", flush=True)


def _init_synthesis_worker(dataset, completed_paths, args):
    global _WORKER_CTX, _WORKER_SONG_COUNT
    _WORKER_SONG_COUNT = 0
    _WORKER_CTX = {
        "dataset": dataset,
        "completed_paths": completed_paths,
        "args": args,
    }
    if uses_ddsp(args.render_mode) or getattr(args, "ddsp_pass", None) in (
        "ddsp_piano",
        "midi_ddsp",
    ):
        from synthesis.ddsp.pool import ddsp_oneshot_enabled, ensure_ddsp_pool

        ddsp_pass = getattr(args, "ddsp_pass", None)
        if (
            ddsp_pass in ("ddsp_piano", "midi_ddsp")
            and not ddsp_oneshot_enabled()
        ):
            ensure_ddsp_pool(preload=ddsp_pass)


def _synthesis_worker(i: int):
    global _WORKER_SONG_COUNT
    ctx = _WORKER_CTX
    args = ctx["args"]
    refresh_every = int(getattr(args, "refresh_every", 0) or 0)
    if refresh_every > 0:
        _WORKER_SONG_COUNT += 1
        if _WORKER_SONG_COUNT % refresh_every == 0:
            ctx["completed_paths"] = _reload_progress_from_disk(
                args,
                output_filepath=str(getattr(args, "_progress_output_filepath", "")),
                routing_output_filepath=getattr(args, "_progress_routing_filepath", None),
                recipe_output_filepath=getattr(args, "_progress_recipe_filepath", None),
                completed_paths=ctx["completed_paths"],
            )
    return synthesize_song_at_index(
        i,
        ctx["dataset"],
        ctx["completed_paths"],
        args,
    )


def _synthesis_worker_logged(idx: int):
    """Module-level so multiprocessing.Pool can pickle it (nested defs cannot)."""
    from synthesis.job_log import get_job_log

    log = get_job_log()
    ctx = _WORKER_CTX
    dataset = ctx["dataset"]
    args = ctx["args"]
    path = str(dataset.at[idx, "path_output"])
    if log is not None:
        log.song_start(idx, path)
    try:
        result = _synthesis_worker(idx)
    except Exception as exc:
        if log is not None:
            log.song_error(idx, path, exc)
        raise
    if log is not None:
        _, _, _, _, n_progress = result
        use_tracks = bool(getattr(args, "_use_tracks_progress", False))
        log.song_done(
            idx,
            path,
            n_tracks=int(n_progress) if use_tracks else 1,
        )
    return result


def _run_song_pool(
    *,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    args,
    work_indices: list,
    jobs: int,
    desc: str,
    stems_output_filepath: str,
    routing_output_filepath: str | None,
    output_filepath: str,
    write_tables: bool,
    recipe_output_filepath: str | None = None,
    track_total: int | None = None,
    append_only: bool = False,
    use_threads: bool = False,
) -> None:
    """Run song workers once (one DDSP pass or the non-DDSP path)."""
    from synthesis.recipe import STEM_RECIPE_COLUMNS

    write_row = append_rows if append_only else append_rows_deduped
    write_kw = {} if append_only else {"key_cols": ["path", "track"]}
    # track_total is None for non-hybrid pools; otherwise it is remaining stems.
    use_tracks = track_total is not None
    pbar = tqdm(
        total=int(track_total) if use_tracks else len(work_indices),
        desc=desc,
        unit="track" if use_tracks else "song",
        # stdout: many long jobs redirect stderr to /dev/null to hide TF noise.
        file=sys.stdout,
        dynamic_ncols=True,
    )

    from synthesis.job_log import (
        close_job_log,
        get_job_log,
        job_log_enabled,
        open_job_log,
    )

    pass_name = getattr(args, "ddsp_pass", None) or "synthesis"
    job_log = None
    if job_log_enabled():
        log_path = (
            os.environ.get("SPDMX_SYNTH_LOG_PATH")
            or str(Path(stems_output_filepath).parent / f"synthesis.{pass_name}.log")
        )
        job_log = open_job_log(
            log_path,
            verbose=bool(getattr(args, "verbose", False)),
            synthesis_pass=pass_name,
            desc=desc,
            songs=len(work_indices),
            tracks=track_total,
            workers=jobs,
            threads=use_threads,
        )
        print(f"Synthesis job log: {log_path}", flush=True)

    args._use_tracks_progress = use_tracks

    def _consume(result) -> None:
        song_info, stem_rows, routing_rows, recipe_rows, n_progress = result
        inc = int(n_progress) if use_tracks else 1
        # tqdm drops the percentage bar once n exceeds total; grow the
        # denominator if the initial remaining estimate was low.
        if use_tracks and pbar.total is not None and pbar.n + inc > pbar.total:
            pbar.total = pbar.n + inc
        pbar.update(inc)
        if recipe_rows and recipe_output_filepath is not None:
            write_row(
                recipe_output_filepath,
                STEM_RECIPE_COLUMNS,
                recipe_rows,
                **write_kw,
            )
            index = getattr(args, "stem_recipe_index", None)
            if isinstance(index, dict):
                for row in recipe_rows:
                    index[(str(row["path"]), int(row["track"]))] = row
        if stem_rows:
            write_row(
                stems_output_filepath,
                STEMS_TABLE_COLUMNS,
                stem_rows,
                **write_kw,
            )
        if routing_rows and routing_output_filepath is not None:
            write_row(
                routing_output_filepath,
                DDSP_ROUTING_COLUMNS,
                routing_rows,
                **write_kw,
            )
        if not write_tables or song_info is None:
            return
        append_rows_deduped(
            output_filepath,
            SONGS_TABLE_COLUMNS,
            [song_info],
        )

    # Dynamic --refresh-every only works with thread pools (MIDI-DDSP /
    # DDSP-Piano). Fluidsynth uses multiprocessing.Pool + imap; ignore the
    # flag there rather than plumbing apply_async / cross-process state.
    requested_refresh = int(getattr(args, "refresh_every", 0) or 0)
    refresh_every = 0 if not use_threads else requested_refresh
    if requested_refresh > 0 and not use_threads:
        print(
            "Ignoring --refresh-every (Fluidsynth process pool).",
            flush=True,
        )
    elif refresh_every > 0:
        print(
            f"Reloading progress from shared storage every {refresh_every} song(s).",
            flush=True,
        )
    audio_format = synthesis_audio_format(args.flac)
    args._progress_output_filepath = output_filepath
    args._progress_routing_filepath = routing_output_filepath
    args._progress_recipe_filepath = recipe_output_filepath
    shared_completed = completed_paths
    songs_since_refresh = 0
    skipped = 0

    def _maybe_refresh_progress() -> None:
        nonlocal shared_completed, songs_since_refresh
        if refresh_every <= 0:
            return
        songs_since_refresh += 1
        if songs_since_refresh < refresh_every:
            return
        songs_since_refresh = 0
        shared_completed = _reload_progress_from_disk(
            args,
            output_filepath=output_filepath,
            routing_output_filepath=routing_output_filepath,
            recipe_output_filepath=recipe_output_filepath,
            completed_paths=shared_completed,
        )
        if use_threads and _WORKER_CTX:
            _WORKER_CTX["completed_paths"] = shared_completed
        # Keep the bar denominator aligned with remaining work on shared storage.
        if use_tracks and pass_name in PASS_TRACK_COLUMNS:
            _, rem = _work_for_pass(
                dataset,
                work_indices,
                pass_name,
                stem_recipe_index=getattr(args, "stem_recipe_index", None),
                tables_dir=getattr(args, "_tables_dir", None),
            )
            if rem is not None:
                pbar.total = max(pbar.n + int(rem), pbar.n)

    def _next_work_index(todo_iter):
        nonlocal skipped
        while True:
            try:
                idx = next(todo_iter)
            except StopIteration:
                return None
            if _song_index_needs_work(
                idx, dataset, args, shared_completed, audio_format,
            ):
                return idx
            skipped += 1

    def _run_dynamic_thread_pool() -> None:
        """Fill a thread pool dynamically so ``--refresh-every`` can skip peers' work."""
        workers = max(1, int(jobs))
        pending: set = set()
        todo = iter(work_indices)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while len(pending) < workers:
                    idx = _next_work_index(todo)
                    if idx is None:
                        break
                    pending.add(pool.submit(_synthesis_worker_logged, idx))
                if not pending:
                    break
                try:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    if job_log is not None:
                        job_log.interrupted(where="song pool wait")
                    raise
                pending = set(pending)
                for fut in done:
                    try:
                        _consume(fut.result())
                    except Exception as exc:
                        if job_log is not None:
                            job_log.fatal(exc, where="song pool result")
                        raise
                    _maybe_refresh_progress()
        if skipped:
            print(
                f"Skipped {skipped} song(s) already complete on shared storage.",
                flush=True,
            )

    try:
        if use_threads:
            _preload_track_maps(args)
            _init_synthesis_worker(dataset, shared_completed, args)
            _run_dynamic_thread_pool()
            return

        pool_ctx = (
            multiprocessing.get_context("spawn")
            if _pool_should_spawn(args)
            else multiprocessing
        )
        with pool_ctx.Pool(
            processes=jobs,
            initializer=_init_synthesis_worker,
            initargs=(dataset, shared_completed, args),
        ) as pool:
            for result in pool.imap(
                _synthesis_worker_logged, work_indices, chunksize=CHUNK_SIZE
            ):
                _consume(result)
    except KeyboardInterrupt:
        if job_log is not None:
            job_log.interrupted(where="song pool")
        raise
    except Exception as exc:
        if job_log is not None:
            job_log.fatal(exc, where="song pool")
        raise
    finally:
        pbar.close()
        if job_log is not None:
            close_job_log()


def _hybrid_neural_song_workers() -> int:
    """Songs in flight for midi_ddsp / ddsp_piano (one thread per GPU)."""
    from synthesis.ddsp.env import parse_ddsp_gpu_ids
    from synthesis.ddsp.pool import ddsp_oneshot_enabled

    if ddsp_oneshot_enabled():
        return 1
    return max(1, len(parse_ddsp_gpu_ids()))


def _jobs(args, default: int = 1) -> int:
    return max(1, int(getattr(args, "jobs", default) or default))


def _parallel_map(fn, items, *, jobs: int, desc: str, unit: str = "song"):
    """Map ``fn`` over ``items`` with a thread pool when ``jobs > 1``.

    Used for mkdir / exists I/O. Process pools are a poor fit (tiny tasks, NFS).
    """
    items = list(items)
    n_jobs = max(1, int(jobs))
    label = desc if n_jobs <= 1 else f"{desc} (-j {n_jobs})"
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in tqdm(items, total=len(items), desc=label, unit=unit)]
    chunksize = 8
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(
            tqdm(
                executor.map(fn, items, chunksize=chunksize),
                total=len(items),
                desc=label,
                unit=unit,
            )
        )


def reset_synthesis_output(output_dir: str) -> None:
    """Remove all prior synthesis artifacts under output_dir."""
    if exists(output_dir):
        shutil.rmtree(output_dir)
    makedirs(output_dir, exist_ok=True)


def prepare_render_dataset(
    args,
    output_dir: str,
    *,
    register_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load PDMX rows for this run and attach ``path_output`` song directories."""
    print(f"Loading PDMX from {args.dataset_filepath} ...", flush=True)
    skip_cols = {"metadata", "mxl", "pdf", "version"}
    dataset = pd.read_csv(
        args.dataset_filepath,
        sep=",",
        header=0,
        index_col=False,
        usecols=lambda col: col not in skip_cols,
    )
    dataset = dataset[dataset["subset:all_valid"]].reset_index(drop=True)
    dataset = dataset.drop(columns=["subset:all_valid"], errors="ignore")
    print(f"Using {len(dataset)} valid songs", flush=True)
    if args.full:
        dataset = prepare_full_dataset(dataset)
    else:
        sample_file = listening_sample_path(args.output_dir)
        dataset = prepare_ablation_dataset(
            dataset,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            min_stems_per_category=args.min_stems_per_category,
            register_df=register_df,
            listening_sample_file=sample_file,
            persist_sample=True,
        )
        if sample_file.is_file():
            print(f"Ablation sample: {sample_file} ({len(dataset)} songs)")
    original_dataset_dir = dirname(args.dataset_filepath)
    dataset["path"] = [original_dataset_dir + p[1:] for p in dataset["path"]]
    dataset["mid"] = [original_dataset_dir + p[1:] for p in dataset["mid"]]
    dataset["mid_pdmx"] = dataset["mid"]
    dataset["path_output"] = [
        song_output_dir(
            output_dir,
            original_dataset_dir,
            p,
            tree_dir_name=render_tree_dir_name(args),
        )
        for p in dataset["path"]
    ]
    return dataset.reset_index(drop=True)


def _hybrid_corrected_midi_root(args) -> Path:
    from synthesis.dense_midi import default_corrected_midi_dir

    return Path(
        getattr(args, "corrected_midi_dir", None)
        or default_corrected_midi_dir(args.output_dir)
    )


def _midi_index_path(output_dir: str) -> Path:
    return Path(output_dir) / MIDI_INDEX_FILE_NAME


def build_midi_index(args) -> pd.DataFrame | None:
    """One row per SPDMX.csv song: dense MIDI path, n_tracks, and per-pass track counts."""
    from analysis.corrected_midi import resolve_track_map_csv
    from synthesis.recipe import hybrid_pass_for_track

    corrected_root = _hybrid_corrected_midi_root(args)
    csv_path = resolve_track_map_csv(corrected_root)
    if not csv_path.is_file():
        return None
    usecols = ["song_id", "program", "is_drum", "name"]
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in usecols if c in header.columns]
    if "song_id" not in cols:
        return None
    tracks = pd.read_csv(csv_path, usecols=cols)
    if tracks.empty:
        return None
    n_by_id = tracks.groupby("song_id", sort=False).size()
    root = str(corrected_root).rstrip("/")
    song_ids = n_by_id.index.astype(str)
    index = pd.DataFrame({
        "song_id": song_ids.to_numpy(),
        "mid": root + "/" + song_ids + ".mid",
        "n_tracks": n_by_id.to_numpy(),
        "n_fluidsynth": 0,
        "n_ddsp_piano": 0,
        "n_midi_ddsp": 0,
    })
    recipe = _resolved_recipe(args)
    if recipe is not None and "program" in tracks.columns:
        print("Counting Fluidsynth / DDSP-piano / MIDI-DDSP tracks from SPDMX.csv ...", flush=True)
        programs = tracks["program"].fillna(0).astype(int)
        drum_col = (
            tracks["is_drum"] if "is_drum" in tracks.columns
            else pd.Series(False, index=tracks.index)
        )
        name_col = (
            tracks["name"] if "name" in tracks.columns
            else pd.Series([None] * len(tracks), index=tracks.index)
        )
        assigned = []
        for i in range(len(tracks)):
            raw_name = name_col.iloc[i]
            assigned.append(hybrid_pass_for_track(
                recipe,
                program=int(programs.iloc[i]),
                is_drum=bool(drum_col.iloc[i]) if pd.notna(drum_col.iloc[i]) else False,
                track_name=None if pd.isna(raw_name) else str(raw_name),
            ))
        tracks = tracks.copy()
        tracks["pass"] = assigned
        counts = (
            tracks.groupby(["song_id", "pass"]).size().unstack(fill_value=0)
        )
        for pass_name, col in PASS_TRACK_COLUMNS.items():
            if pass_name in counts.columns:
                index[col] = index["song_id"].map(counts[pass_name]).fillna(0).astype(int)
    return index


def write_midi_index(args, output_dir: str) -> pd.DataFrame | None:
    index = build_midi_index(args)
    if index is None:
        return None
    path = _midi_index_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
    print(f"Wrote {path} ({len(index)} songs)", flush=True)
    return index


def load_midi_index(args, output_dir: str) -> pd.DataFrame | None:
    from analysis.corrected_midi import resolve_track_map_csv

    path = _midi_index_path(output_dir)
    if not path.is_file():
        return None
    csv_path = resolve_track_map_csv(_hybrid_corrected_midi_root(args))
    if csv_path.is_file() and path.stat().st_mtime < csv_path.stat().st_mtime:
        print(f"Rebuilding midi index ({path} older than {csv_path})", flush=True)
        return None
    recipe = _resolved_recipe(args)
    recipe_path = getattr(recipe, "path", None) if recipe is not None else None
    if recipe_path and Path(recipe_path).is_file() and path.stat().st_mtime < Path(recipe_path).stat().st_mtime:
        print(f"Rebuilding midi index ({path} older than recipe {recipe_path})", flush=True)
        return None
    index = pd.read_csv(path)
    required = {"song_id", "mid", "n_tracks", *PASS_TRACK_COLUMNS.values()}
    if not required <= set(index.columns):
        return None
    if (
        recipe is not None
        and not recipe.uses_ddsp_piano()
        and "n_ddsp_piano" in index.columns
        and int(index["n_ddsp_piano"].sum()) > 0
    ):
        print(
            f"Rebuilding midi index (DDSP-Piano unused; piano recipe is not midi-ddsp)",
            flush=True,
        )
        return None
    return index


def _resolve_corrected_midi_slow(dataset: pd.DataFrame, args) -> pd.DataFrame:
    """Per-song path resolve + exists check (used when SPDMX.csv is missing)."""
    from analysis.corrected_midi import (
        load_track_maps,
        resolve_corrected_midi_path,
        resolve_track_map_csv,
        song_id_from_mid,
    )

    original_dataset_dir = dirname(args.dataset_filepath)
    corrected_root = _hybrid_corrected_midi_root(args)
    track_maps = load_track_maps(resolve_track_map_csv(corrected_root))

    def _resolve_one(mid: str) -> tuple[str, int]:
        song_id = song_id_from_mid(mid)
        corrected = resolve_corrected_midi_path(
            mid,
            pdmx_root=original_dataset_dir,
            corrected_midi_dir=corrected_root,
        )
        if not corrected.is_file():
            raise FileNotFoundError(
                f"Corrected MIDI missing: {corrected}\n"
                "Generate corrected midis first:\n"
                "  uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
            )
        return str(corrected), len(track_maps[song_id])

    resolved = _parallel_map(
        _resolve_one,
        dataset["mid_pdmx"].tolist(),
        jobs=_jobs(args),
        desc="Resolving corrected MIDI",
    )
    dataset = dataset.copy()
    dataset["mid"] = [mid for mid, _ in resolved]
    dataset["n_tracks"] = [n for _, n in resolved]
    return dataset


def attach_corrected_midi(dataset: pd.DataFrame, args, output_dir: str) -> pd.DataFrame:
    """Set dense ``mid`` / ``n_tracks`` from midi_index.csv (built from SPDMX.csv)."""
    from analysis.corrected_midi import song_id_from_mid

    index = load_midi_index(args, output_dir)
    if index is None:
        index = write_midi_index(args, output_dir)
    if index is None:
        return _resolve_corrected_midi_slow(dataset, args)

    lookup = index.drop_duplicates("song_id").set_index("song_id")
    mid_col = dataset["mid_pdmx"] if "mid_pdmx" in dataset.columns else dataset["mid"]
    keys = mid_col.map(song_id_from_mid)
    dataset = dataset.copy()
    dataset["mid"] = keys.map(lookup["mid"])
    dataset["n_tracks"] = keys.map(lookup["n_tracks"])
    dataset["song_id"] = keys.astype(str)
    for col in PASS_TRACK_COLUMNS.values():
        if col in lookup.columns:
            dataset[col] = pd.to_numeric(keys.map(lookup[col]), errors="coerce").fillna(0).astype(int)
    missing = int(dataset["mid"].isna().sum())
    if missing:
        raise FileNotFoundError(
            f"{missing} songs missing from { _midi_index_path(output_dir) }. "
            "Re-run: uv run python -m synthesis.final --only-pass layout"
        )
    dataset["n_tracks"] = dataset["n_tracks"].astype(int)
    print(
        f"Using midi index ({len(index)} songs) from {_midi_index_path(output_dir)}",
        flush=True,
    )
    return dataset


def _restrict_dataset_to_spdmx_csv(dataset: pd.DataFrame, song_ids: set[str]) -> pd.DataFrame:
    """Keep PDMX rows whose song_id is in the released ``SPDMX.csv``."""
    from analysis.corrected_midi import song_id_from_mid

    mid_col = dataset["mid_pdmx"] if "mid_pdmx" in dataset.columns else dataset["mid"]
    keep = mid_col.map(song_id_from_mid).isin(song_ids)
    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"Restricting to SPDMX.csv ({len(song_ids)} songs); "
            f"dropped {n_drop} PDMX rows",
            flush=True,
        )
    return dataset.loc[keep].reset_index(drop=True)


def _spdmx_csv_song_ids(args) -> set[str] | None:
    from analysis.corrected_midi import resolve_track_map_csv

    csv_path = resolve_track_map_csv(_hybrid_corrected_midi_root(args))
    if not csv_path.is_file():
        return None
    return set(pd.read_csv(csv_path, usecols=["song_id"])["song_id"].astype(str))


def ensure_synthesis_tables(output_dir: str, args) -> None:
    """Create empty data/stems/routing/recipe CSVs when missing (or after ``--reset``)."""
    output_filepath = f"{output_dir}/{DATA_DIR_NAME}.csv"
    mistaken_release_name = f"{output_dir}/{SPDMX_FILE_NAME}.csv"
    if exists(mistaken_release_name) and not exists(output_filepath):
        Path(mistaken_release_name).rename(output_filepath)
    stems_output_filepath = f"{output_dir}/{STEMS_FILE_NAME}.csv"
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns=SONGS_TABLE_COLUMNS).to_csv(
            output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )
    if not exists(stems_output_filepath) or args.reset:
        pd.DataFrame(columns=STEMS_TABLE_COLUMNS).to_csv(
            stems_output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )
    routing_output_filepath = f"{output_dir}/{DDSP_ROUTING_FILE_NAME}"
    if _needs_ddsp_routing(args) and (not exists(routing_output_filepath) or args.reset):
        pd.DataFrame(columns=DDSP_ROUTING_COLUMNS).to_csv(
            routing_output_filepath, sep=",", na_rep=NA_STRING, header=True, index=False, mode="w",
        )
    from synthesis.recipe import STEM_RECIPE_COLUMNS, STEM_RECIPE_FILE_NAME

    recipe = _hybrid_recipe(args)
    if recipe is not None:
        recipe_output_filepath = f"{output_dir}/{STEM_RECIPE_FILE_NAME}"
        if not exists(recipe_output_filepath) or args.reset:
            pd.DataFrame(columns=STEM_RECIPE_COLUMNS).to_csv(
                recipe_output_filepath,
                sep=",",
                na_rep=NA_STRING,
                header=True,
                index=False,
                mode="w",
            )


def run_layout_pass(
    args,
    output_dir: str,
    *,
    media_dir: str | None = None,
) -> pd.DataFrame:
    """Pass 0: mkdir the PDMX-mirrored audio (and mid) tree, no media yet.

    ``output_dir`` holds data/stems/recipe CSVs. Hybrid ``--full`` uses
    ``media_dir={SPDMX}`` so those tables stay out of the released tree.
    """
    media_dir = media_dir or output_dir
    if args.reset:
        print(f"Reset: clearing tables under {output_dir} ...", flush=True)
        reset_synthesis_output(output_dir)
        if Path(media_dir).resolve() != Path(output_dir).resolve():
            audio_root = Path(media_dir) / SPDMX_RAW_DIR_NAME
            if audio_root.exists():
                print(
                    f"Reset: deleting {audio_root} (can take several minutes on NFS) ...",
                    flush=True,
                )
                shutil.rmtree(audio_root)
                print("Reset: raw stem tree removed.", flush=True)
    else:
        makedirs(output_dir, exist_ok=True)
        makedirs(media_dir, exist_ok=True)
    leaf = render_tree_dir_name(args)
    makedirs(f"{media_dir}/{leaf}", exist_ok=True)
    if _hybrid_recipe(args) is not None:
        makedirs(f"{media_dir}/{SPDMX_MID_DIR_NAME}", exist_ok=True)

    register_df = None
    if not args.full and not getattr(args, "no_register", False):
        register_path = getattr(args, "register", None) or default_gm_register_path(args.output_dir)
        if exists(register_path):
            register_df = pd.read_csv(register_path)

    dataset = prepare_render_dataset(args, media_dir, register_df=register_df)
    hybrid = _hybrid_recipe(args) is not None
    if hybrid:
        song_ids = _spdmx_csv_song_ids(args)
        if song_ids is not None:
            dataset = _restrict_dataset_to_spdmx_csv(dataset, song_ids)
    print(f"Planning {len(dataset)} song directories ...", flush=True)
    dirs: list[str] = list(dataset["path_output"])
    if hybrid:
        pdmx_root = str(Path(dirname(args.dataset_filepath))).rstrip("/")
        prefix = pdmx_root + "/"
        media = Path(media_dir)
        for mid in dataset["mid"]:
            mid_s = str(mid)
            rel = mid_s[len(prefix):] if mid_s.startswith(prefix) else mid_s.lstrip("/")
            dirs.append(str((media / rel).parent))
    seen: set[str] = set()
    unique_dirs: list[str] = []
    for path in dirs:
        if path not in seen:
            seen.add(path)
            unique_dirs.append(path)
    print(f"Creating {len(unique_dirs)} directories ...", flush=True)
    _parallel_map(lambda path: makedirs(path, exist_ok=True), unique_dirs, jobs=_jobs(args), desc="Pass 0 layout")
    ensure_synthesis_tables(output_dir, args)
    if hybrid:
        write_midi_index(args, output_dir)
        from synthesis.spdmx_release import maybe_write_spdmx_release_docs

        maybe_write_spdmx_release_docs(media_dir)
    print(
        f"Pass 0 layout: {len(dataset)} song directories under {media_dir}/{leaf}",
        flush=True,
    )
    return dataset


def _run_hybrid_synthesis(
    *,
    args,
    recipe,
    dataset: pd.DataFrame,
    completed_paths: set[str],
    work_indices: list,
    stems_output_filepath: str,
    routing_output_filepath: str | None,
    output_filepath: str,
    recipe_output_filepath: str | None,
) -> None:
    """Fluidsynth / DDSP write per-pass CSVs; mix merges them into canonical tables."""
    from synthesis.ddsp.pool import shutdown_ddsp_pool
    from synthesis.pass_tables import pass_recipe_csv, pass_routing_csv, pass_stems_csv
    from synthesis.recipe import load_stem_recipe_index

    only = getattr(args, "only_pass", None)
    uses_neural = recipe.uses_ddsp()
    run_fluidsynth = only in (None, "fluidsynth")
    run_ddsp_piano = recipe.uses_ddsp_piano() and only in (None, "ddsp_piano")
    run_midi_ddsp = uses_neural and only in (None, "midi_ddsp")
    tables = Path(output_filepath).parent
    args._tables_dir = tables

    def _one(pass_name: str, desc: str, pass_jobs: int, *, use_threads: bool = False) -> None:
        args.ddsp_pass = pass_name
        recipe_path = pass_recipe_csv(tables, pass_name)
        args.stem_recipe_index = load_stem_recipe_index(
            tables, filename=recipe_path.name,
        )
        if pass_name == "midi_ddsp":
            args._pending_midi_ddsp_keys = _fluidsynth_pending_unclaimed_keys(
                tables, args.stem_recipe_index,
            )
            n_pending = len(args._pending_midi_ddsp_keys)
            if n_pending:
                n_songs = len({p for p, _ in args._pending_midi_ddsp_keys})
                print(
                    f"Fluidsynth pending_midi_ddsp still needing neural: "
                    f"{n_pending} tracks across {n_songs} songs",
                    flush=True,
                )
        else:
            args._pending_midi_ddsp_keys = frozenset()
        indices, track_total = _work_for_pass(
            dataset, work_indices, pass_name,
            stem_recipe_index=args.stem_recipe_index,
            tables_dir=tables,
        )
        extra = ""
        if track_total is not None:
            extra = f", {track_total} tracks left, {len(indices)} songs"
        workers = (
            f"{pass_jobs} GPUs across songs" if use_threads else f"-j {pass_jobs}"
        )
        print(f"Hybrid pass: {pass_name} ({workers}{extra})", flush=True)
        if not indices:
            print(f"No {pass_name} tracks to render.", flush=True)
            return
        _run_song_pool(
            dataset=dataset,
            completed_paths=completed_paths,
            args=args,
            work_indices=indices,
            jobs=pass_jobs,
            desc=desc,
            stems_output_filepath=str(pass_stems_csv(tables, pass_name)),
            routing_output_filepath=(
                str(pass_routing_csv(tables, pass_name))
                if routing_output_filepath is not None
                else None
            ),
            output_filepath=output_filepath,
            write_tables=False,
            recipe_output_filepath=str(recipe_path),
            track_total=track_total,
            append_only=True,
            use_threads=use_threads,
        )
        if pass_name in ("ddsp_piano", "midi_ddsp"):
            shutdown_ddsp_pool()

    if run_fluidsynth:
        _one("fluidsynth", "Fluidsynth stems", max(1, int(args.jobs)))
    if run_ddsp_piano or run_midi_ddsp:
        gpu_jobs = _hybrid_neural_song_workers()
        if gpu_jobs > 1:
            print(
                f"MIDI-DDSP / DDSP-Piano: {gpu_jobs} songs in flight "
                "(one thread per CUDA_VISIBLE_DEVICES id).",
                flush=True,
            )
        if run_ddsp_piano:
            args.neural_inner_workers = 1
            _one("ddsp_piano", "DDSP-Piano stems", gpu_jobs, use_threads=True)
        if run_midi_ddsp:
            args.neural_inner_workers = 1
            _one("midi_ddsp", "MIDI-DDSP stems", gpu_jobs, use_threads=True)
    elif only == "ddsp_piano" and not recipe.uses_ddsp_piano():
        print("Hybrid DDSP-Piano pass skipped (piano recipe is not midi-ddsp).", flush=True)
    elif only == "midi_ddsp" and not uses_neural:
        print("Hybrid DDSP pass skipped (no category uses midi-ddsp).", flush=True)
    args.ddsp_pass = None


def run_synthesis(args, output_dir: str, *, media_dir: str | None = None):
    media_dir = media_dir or output_dir
    if args.reset and not getattr(args, "skip_output_reset", False):
        reset_synthesis_output(output_dir)
        if Path(media_dir).resolve() != Path(output_dir).resolve():
            audio_root = Path(media_dir) / SPDMX_RAW_DIR_NAME
            if audio_root.exists():
                shutil.rmtree(audio_root)
    else:
        makedirs(output_dir, exist_ok=True)
        makedirs(media_dir, exist_ok=True)
    output_filepath = f"{output_dir}/{DATA_DIR_NAME}.csv"
    stems_output_filepath = f"{output_dir}/{STEMS_FILE_NAME}.csv"
    makedirs(f"{media_dir}/{render_tree_dir_name(args)}", exist_ok=True)

    if args.soundfont_filepath is None:
        args.soundfont_filepath = f"{expanduser('~')}/.muspy/musescore-general/MuseScore_General.sf3"
    if not exists(args.soundfont_filepath):
        raise RuntimeError("Soundfont not found.")

    # GM register: required step-0 corrections unless --no-register.
    args.gm_register_lookup = None
    register_df = None
    register_path = getattr(args, "register", None) or default_gm_register_path(args.output_dir)
    if not getattr(args, "no_register", False):
        from analysis.gm_register import load_register_lookup

        if not exists(register_path):
            raise RuntimeError(
                f"GM register not found at {register_path}\n"
                "Run synthesis setup before any ablation:\n"
                "  uv run python -m analysis.prepare_synthesis --subset all_valid -j 8\n"
                "Or pass --no-register to synthesize with raw MIDI programs."
            )
        pdmx_root = dirname(args.dataset_filepath)
        args.gm_register_lookup = load_register_lookup(register_path, pdmx_root=pdmx_root)
        register_df = pd.read_csv(register_path)
        print(f"Loaded GM register ({len(args.gm_register_lookup)} keys) from {register_path}")

    if uses_ddsp(getattr(args, "render_mode", "") or "") and not args.full:
        if _hybrid_recipe(args) is None:
            require_donor_ablation(args, realify=False)

    dataset = prepare_render_dataset(args, media_dir, register_df=register_df)
    hybrid = _hybrid_recipe(args) is not None
    print(f"Using dense corrected midis under {_hybrid_corrected_midi_root(args)}")
    if hybrid:
        song_ids = _spdmx_csv_song_ids(args)
        if song_ids is not None:
            dataset = _restrict_dataset_to_spdmx_csv(dataset, song_ids)
    dataset = attach_corrected_midi(dataset, args, output_dir)

    if not hybrid:
        _parallel_map(
            lambda path: makedirs(path, exist_ok=True),
            list(dict.fromkeys(dataset["path_output"])),
            jobs=_jobs(args),
            desc="Ensuring song directories",
        )

    ensure_synthesis_tables(output_dir, args)
    output_filepath = f"{output_dir}/{DATA_DIR_NAME}.csv"
    stems_output_filepath = f"{output_dir}/{STEMS_FILE_NAME}.csv"
    routing_output_filepath = f"{output_dir}/{DDSP_ROUTING_FILE_NAME}"
    needs_routing = _needs_ddsp_routing(args)
    from synthesis.recipe import (
        STEM_RECIPE_FILE_NAME,
        load_stem_recipe_index,
        require_recipe_conflicts_ok,
        scan_recipe_conflicts,
    )
    recipe = _hybrid_recipe(args)
    if recipe is not None:
        from synthesis.pass_tables import drop_canonical_tables

        drop_canonical_tables(output_dir)
    recipe_output_filepath = (
        f"{output_dir}/{STEM_RECIPE_FILE_NAME}" if recipe is not None else None
    )
    audio_format = synthesis_audio_format(args.flac)
    if recipe is not None and not args.reset:
        require_recipe_conflicts_ok(
            scan_recipe_conflicts(
                output_dir, recipe, audio_format=audio_format, stage="raw",
            ),
            yes=bool(getattr(args, "yes", False)),
        )
        args.stem_recipe_index = load_stem_recipe_index(output_dir)
    else:
        args.stem_recipe_index = {}

    completed_paths = set()
    if exists(output_filepath) and not args.reset:
        routing_for_completed = (
            routing_output_filepath if needs_routing else None
        )
        # For DDSP, songs with stems/data but incomplete routing are not "done".
        completed_paths = load_completed_song_paths(
            output_filepath,
            routing_csv=routing_for_completed,
        )

    # Neural DDSP needs Torch in workers for resample. Fork-after-CUDA causes
    # SIGKILL (exit -9); use spawn so each worker initializes cleanly.
    ablation_ddsp = uses_ddsp(getattr(args, "render_mode", "") or "") and recipe is None
    jobs = 1 if ablation_ddsp else args.jobs
    if ablation_ddsp and args.jobs > 1:
        print(
            f"Note: {args.render_mode} uses spawn + -j 1 (was {args.jobs}) to avoid "
            "CUDA-after-fork kills (exit -9)."
        )

    work_indices = []
    for i in dataset.index:
        if args.reset:
            work_indices.append(i)
            continue
        path_output = dataset.at[i, "path_output"]
        if path_output not in completed_paths:
            work_indices.append(i)
            continue
        n_tracks = int(dataset.at[i, "n_tracks"])
        if not song_is_complete(
            Path(path_output), n_tracks, audio_format, require_mixture=False,
        ):
            work_indices.append(i)
            continue
        if recipe is not None and not _hybrid_song_raw_current(
            args, path_output, n_tracks, Path(path_output), audio_format, recipe,
        ):
            work_indices.append(i)

    refresh_every = int(getattr(args, "refresh_every", 10) or 0)
    if refresh_every < 0:
        refresh_every = 0
    args.refresh_every = refresh_every

    shard_count = int(getattr(args, "shard_count", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    try:
        validate_shard_args(shard_count, shard_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    work_indices, assigned_count = filter_work_indices_by_shard(
        work_indices,
        dataset,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    if shard_count > 1:
        print(
            format_shard_summary(
                shard_count, shard_index, assigned_count, len(work_indices),
            ),
            flush=True,
        )

    if not work_indices:
        return

    if recipe is not None:
        _run_hybrid_synthesis(
            args=args,
            recipe=recipe,
            dataset=dataset,
            completed_paths=completed_paths,
            work_indices=work_indices,
            stems_output_filepath=stems_output_filepath,
            routing_output_filepath=routing_output_filepath if needs_routing else None,
            output_filepath=output_filepath,
            recipe_output_filepath=recipe_output_filepath,
        )
        return

    if uses_ddsp(args.render_mode):
        from synthesis.ddsp.pool import shutdown_ddsp_pool

        # Global two-pass: keep one neural backend hot per phase, then finalize.
        print(
            "DDSP schedule: pass1=ddsp_piano, pass2=midi_ddsp, "
            "pass3=donors/soundfont (pool restarts between neural passes).",
            flush=True,
        )
        ddsp_passes = (
            ("ddsp_piano", "DDSP piano stems", False),
            ("midi_ddsp", "DDSP mono stems", False),
            ("finalize", "DDSP finalize", True),
        )
        for pass_name, desc, write_tables in ddsp_passes:
            args.ddsp_pass = pass_name
            if pass_name == "finalize":
                shutdown_ddsp_pool()
            _run_song_pool(
                dataset=dataset,
                completed_paths=completed_paths,
                args=args,
                work_indices=work_indices,
                jobs=jobs,
                desc=desc,
                stems_output_filepath=stems_output_filepath,
                routing_output_filepath=routing_output_filepath,
                output_filepath=output_filepath,
                write_tables=write_tables,
            )
            if pass_name in ("ddsp_piano", "midi_ddsp"):
                # Drop resident TF models before the next backend.
                shutdown_ddsp_pool()
        args.ddsp_pass = None
        return

    _run_song_pool(
        dataset=dataset,
        completed_paths=completed_paths,
        args=args,
        work_indices=work_indices,
        jobs=jobs,
        desc="Synthesizing songs",
        stems_output_filepath=stems_output_filepath,
        routing_output_filepath=None,
        output_filepath=output_filepath,
        write_tables=True,
    )


def _render_passes_for_recipe(recipe) -> tuple[str, ...]:
    """Hybrid render engines that should be complete before verify/mix."""
    from synthesis.pass_tables import RENDER_PASSES

    if recipe is None:
        return RENDER_PASSES
    passes = ["fluidsynth"]
    if recipe.uses_ddsp_piano():
        passes.append("ddsp_piano")
    if recipe.uses_ddsp():
        passes.append("midi_ddsp")
    return tuple(passes)


def count_pass_remaining(
    tables_dir: str | Path,
    *,
    recipe=None,
    sample_limit: int = 25,
) -> list[dict]:
    """Per-pass assigned / done / remaining track counts from midi_index + recipes.

    ``remaining`` matches ``_work_for_pass`` resume logic (assigned tracks not yet
    recorded in ``stem_recipe.<pass>.csv``). Each row includes ``examples``:
    ``[{song_id, remaining}, ...]`` (up to ``sample_limit``) for incomplete songs.
    """
    from synthesis.pass_tables import _song_id_from_audio_dir, pass_recipe_csv
    from synthesis.recipe import load_stem_recipe_index

    root = Path(tables_dir)
    index_path = root / MIDI_INDEX_FILE_NAME
    if not index_path.is_file():
        return []
    index = pd.read_csv(index_path)
    if index.empty or "song_id" not in index.columns:
        return []

    rows: list[dict] = []
    for pass_name in _render_passes_for_recipe(recipe):
        col = PASS_TRACK_COLUMNS.get(pass_name)
        if col is None or col not in index.columns:
            continue
        recipe_path = pass_recipe_csv(root, pass_name)
        stem_index = load_stem_recipe_index(root, filename=recipe_path.name)
        done_by_path = _work_done_by_path(
            pass_name,
            stem_recipe_index=stem_index,
            tables_dir=root,
        )
        pending_by_path: dict[str, int] = {}
        if pass_name == "midi_ddsp":
            pending_by_path = _fluidsynth_pending_unclaimed_counts_by_path(
                root,
                stem_index,
            )
        done_by_sid: dict[str, int] = {}
        for path, n in done_by_path.items():
            sid = _song_id_from_audio_dir(str(path))
            done_by_sid[sid] = done_by_sid.get(sid, 0) + int(n)
        pending_by_sid: dict[str, int] = {}
        for path, n in pending_by_path.items():
            sid = _song_id_from_audio_dir(str(path))
            pending_by_sid[sid] = pending_by_sid.get(sid, 0) + int(n)
        assigned = 0
        remaining = 0
        songs_left = 0
        neural_done = 0
        examples: list[dict] = []
        for _, row in tqdm(
            index.iterrows(),
            total=len(index),
            desc=f"verify remaining ({pass_name})",
            unit="song",
            leave=False,
        ):
            n = int(row[col])
            sid = str(row["song_id"])
            rem_pending = int(pending_by_sid.get(sid, 0))
            if n <= 0 and rem_pending <= 0:
                continue
            if n > 0:
                assigned += n
                done_cap = min(n, int(done_by_sid.get(sid, 0)))
            else:
                assigned += rem_pending
                done_cap = 0
            neural_done += done_cap
            rem = (
                _midi_ddsp_remaining_for_path(
                    assigned=n,
                    done=done_cap,
                    pending_unclaimed=rem_pending,
                )
                if pass_name == "midi_ddsp"
                else max(0, n - done_cap)
            )
            if rem > 0:
                remaining += rem
                songs_left += 1
                if len(examples) < sample_limit:
                    examples.append({"song_id": sid, "remaining": rem})
        rows.append({
            "pass": pass_name,
            "assigned": assigned,
            "done": neural_done,
            "remaining": remaining,
            "songs_left": songs_left,
            "examples": examples,
        })
    return rows


def _verify_stem_path_ok(path_str: str) -> bool:
    """Picklable worker: True when claimed stem path is valid on disk."""
    return stem_is_valid(Path(path_str))


def _verify_stem_path_ok_indexed(item: tuple[int, str]) -> tuple[int, bool]:
    idx, path_str = item
    return idx, stem_is_valid(Path(path_str))


def _verify_song_complete_task(
    item: tuple[str, int, str, bool],
) -> bool:
    song_dir, n_tracks, audio_format, require_mixture = item
    return song_is_complete(
        Path(song_dir),
        int(n_tracks),
        audio_format,
        require_mixture=require_mixture,
    )


def _missing_from_stem_paths(
    path_strs: list[str],
    *,
    jobs: int = 1,
    desc: str = "verify disk",
    limit: int | None = None,
) -> list[str]:
    """Return invalid paths from ``path_strs`` (optional early stop via ``limit``)."""
    if not path_strs:
        return []
    n_jobs = max(1, int(jobs))
    label = f"{desc} (-j {n_jobs})" if n_jobs > 1 else desc
    missing: list[str] = []
    if n_jobs <= 1 or len(path_strs) <= 1:
        for path_str in tqdm(path_strs, total=len(path_strs), desc=label, unit="stem", leave=False):
            if not _verify_stem_path_ok(path_str):
                missing.append(path_str)
                if limit is not None and len(missing) >= limit:
                    break
        return missing

    chunksize = max(4, min(32, len(path_strs) // (n_jobs * 8) or 4))
    pbar = tqdm(
        total=len(path_strs),
        desc=label,
        unit="stem",
        leave=False,
        miniters=1,
        smoothing=0.05,
    )
    try:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for idx, ok in pool.imap_unordered(
                _verify_stem_path_ok_indexed,
                enumerate(path_strs),
                chunksize=chunksize,
            ):
                pbar.update(1)
                if not ok:
                    missing.append(path_strs[idx])
                    if limit is not None and len(missing) >= limit:
                        pool.terminate()
                        break
    finally:
        pbar.close()
    # imap_unordered order is nondeterministic; stable report order.
    missing.sort()
    return missing


def find_missing_claimed_stems(
    tables_dir: str | Path,
    audio_format: str,
    *,
    limit: int | None = None,
    by_pass: bool = False,
    jobs: int = 1,
) -> list[str] | dict[str, list[str]]:
    """Paths claimed in ``stems.csv`` (or per-pass shards) that lack a valid FLAC.

    Used as the mix-time verify so CSV-only resume cannot ship missing audio.
    When ``by_pass`` is True, return ``{pass_name: [paths...]}`` (canonical
    ``stems.csv`` is checked under the key ``\"merged\"``).
    ``jobs`` parallelizes stem validation via ``multiprocessing.Pool``.
    """
    from synthesis.pass_tables import RENDER_PASSES, pass_stems_csv

    root = Path(tables_dir)

    def _missing_from_frame(frame: pd.DataFrame, *, desc: str) -> list[str]:
        if frame is None or frame.empty:
            return []
        uniq = frame.drop_duplicates(["path", "track"])
        path_strs = [
            str(stem_path(Path(str(p)), int(t), audio_format))
            for p, t in zip(uniq["path"].tolist(), uniq["track"].tolist(), strict=True)
        ]
        return _missing_from_stem_paths(
            path_strs, jobs=jobs, desc=desc, limit=limit,
        )

    if by_pass:
        out: dict[str, list[str]] = {}
        stems_csv = root / f"{STEMS_FILE_NAME}.csv"
        if stems_csv.is_file() and stems_csv.stat().st_size > 0:
            out["merged"] = _missing_from_frame(
                pd.read_csv(stems_csv, usecols=["path", "track"]),
                desc="verify disk (merged)",
            )
        for name in RENDER_PASSES:
            path = pass_stems_csv(root, name)
            if path.is_file() and path.stat().st_size > 0:
                out[name] = _missing_from_frame(
                    pd.read_csv(path, usecols=["path", "track"]),
                    desc=f"verify disk ({name})",
                )
        return out

    stems_csv = root / f"{STEMS_FILE_NAME}.csv"
    frames = []
    if stems_csv.is_file():
        frames.append(pd.read_csv(stems_csv, usecols=["path", "track"]))
    else:
        for name in RENDER_PASSES:
            path = pass_stems_csv(root, name)
            if path.is_file() and path.stat().st_size > 0:
                frames.append(pd.read_csv(path, usecols=["path", "track"]))
    if not frames:
        return []
    stems = pd.concat(frames, ignore_index=True).drop_duplicates(["path", "track"])
    return _missing_from_frame(stems, desc="verify disk")


def _format_verify_examples(label: str, items: list[str], *, total: int, limit: int) -> str:
    if not items and total <= 0:
        return f"{label}: (none shown)"
    shown = items[:limit]
    body = "\n".join(f"    {item}" for item in shown)
    leftover = total - len(shown)
    suffix = f"\n    ... and {leftover} more" if leftover > 0 else ""
    return f"{label}:\n{body}{suffix}"


def verify_claimed_stems_on_disk(
    tables_dir: str | Path,
    audio_format: str,
    *,
    sample_limit: int = 25,
    recipe=None,
    jobs: int = 1,
) -> None:
    """Raise if render work remains or claimed stems are missing/invalid on disk.

    Always prints a per-pass summary. On failure the exception lists counts and
    concrete examples (incomplete ``song_id``s and missing file paths).
    ``jobs`` parallelizes claimed-stem FLAC checks.
    """
    root = Path(tables_dir)
    remaining_rows = count_pass_remaining(
        root, recipe=recipe, sample_limit=sample_limit,
    )
    summary_lines: list[str] = []

    if remaining_rows:
        summary_lines.append("Pass remaining (assigned − stem_recipe):")
        for row in remaining_rows:
            summary_lines.append(
                f"  {row['pass']}: {row['remaining']} tracks left "
                f"({row['songs_left']} songs; "
                f"{row['done']}/{row['assigned']} recorded)"
            )
            if row["remaining"] and row["examples"]:
                summary_lines.append(
                    _format_verify_examples(
                        f"  examples ({row['pass']})",
                        [
                            f"{ex['song_id']} ({ex['remaining']} tracks)"
                            for ex in row["examples"]
                        ],
                        total=int(row["songs_left"]),
                        limit=sample_limit,
                    )
                )
    elif not (root / MIDI_INDEX_FILE_NAME).is_file():
        summary_lines.append(
            f"No {MIDI_INDEX_FILE_NAME} under {root}; "
            "skipping per-pass remaining counts."
        )

    missing_by_pass = find_missing_claimed_stems(
        root, audio_format, by_pass=True, jobs=jobs,
    )
    assert isinstance(missing_by_pass, dict)
    report_passes = [
        name for name in _render_passes_for_recipe(recipe)
        if name in missing_by_pass
    ]
    if not report_passes and "merged" in missing_by_pass:
        report_passes = ["merged"]

    missing_total = 0
    if report_passes:
        summary_lines.append("Claimed stems missing or invalid on disk:")
        for name in report_passes:
            paths = missing_by_pass.get(name) or []
            n = len(paths)
            missing_total += n
            summary_lines.append(f"  {name}: {n} missing")
            if n:
                summary_lines.append(
                    _format_verify_examples(
                        f"  examples ({name})",
                        paths,
                        total=n,
                        limit=sample_limit,
                    )
                )
    else:
        summary_lines.append(f"No stem tables under {root} to check on disk.")

    report = "\n".join(summary_lines)
    print(report, flush=True)

    unfinished = [r for r in remaining_rows if int(r["remaining"]) > 0]
    problems: list[str] = []
    if unfinished:
        cmds = "\n".join(
            f"  uv run python -m synthesis.final --only-pass {r['pass']}"
            for r in unfinished
        )
        detail = []
        for r in unfinished:
            detail.append(
                f"  {r['pass']}: {r['remaining']} tracks "
                f"({r['songs_left']} songs)"
            )
            if r["examples"]:
                detail.append(
                    _format_verify_examples(
                        f"  missing song_ids ({r['pass']})",
                        [
                            f"{ex['song_id']} ({ex['remaining']} tracks)"
                            for ex in r["examples"]
                        ],
                        total=int(r["songs_left"]),
                        limit=sample_limit,
                    )
                )
        problems.append(
            "Render passes still have stems left to do:\n"
            + "\n".join(detail)
            + f"\nRe-run:\n{cmds}"
        )
    if missing_total:
        detail = []
        for name in report_passes:
            paths = missing_by_pass.get(name) or []
            if not paths:
                continue
            detail.append(f"  {name}: {len(paths)} missing")
            detail.append(
                _format_verify_examples(
                    f"  missing files ({name})",
                    paths,
                    total=len(paths),
                    limit=sample_limit,
                )
            )
        problems.append(
            f"Stem tables claim files that are missing or invalid on disk "
            f"({missing_total} total):\n"
            + "\n".join(detail)
            + "\nFinish copying audio (rsync) or re-render missing stems before mix."
        )

    if problems:
        raise RuntimeError("\n\n".join(problems))
    print(
        f"Verified: assigned stems recorded and present on disk "
        f"under {root} ({audio_format}).",
        flush=True,
    )



def synthesis_is_complete(
    source_dir: str,
    audio_format: str,
    *,
    require_mixture: bool = False,
    expected_n_songs: int | None = None,
    progress: bool = False,
    jobs: int = 1,
) -> bool:
    """True when data/stems tables exist and every listed song has stem files on disk.

    When ``ddsp_routing.csv`` is present, every song must also have routing rows for
    all tracks (DDSP ablations). ``expected_n_songs`` (unique songs in SPDMX.csv)
    rejects a partial ``data.csv`` written while Fluidsynth/DDSP are still running.
    ``jobs`` parallelizes per-song completeness checks.
    """
    source = Path(source_dir)
    data_csv = source / f"{DATA_DIR_NAME}.csv"
    stems_csv = source / f"{STEMS_FILE_NAME}.csv"
    if not data_csv.exists() or not stems_csv.exists():
        return False

    songs = pd.read_csv(data_csv, sep=",", header=0, index_col=False)
    stems = pd.read_csv(
        stems_csv, sep=",", header=0, index_col=False, low_memory=False,
    )
    if len(songs) == 0 or len(stems) == 0:
        return False
    if expected_n_songs is not None and len(songs) < int(expected_n_songs):
        return False

    routing_csv = source / DDSP_ROUTING_FILE_NAME
    if routing_csv.is_file():
        routing = pd.read_csv(routing_csv, sep=",", header=0, index_col=False)
        if songs_missing_routing(songs, routing):
            return False

    tasks = [
        (str(row.path), int(row.n_tracks), audio_format, require_mixture)
        for row in songs[["path", "n_tracks"]].itertuples(index=False)
    ]
    n_jobs = max(1, int(jobs))
    label = "verify song completeness"
    if n_jobs > 1:
        label = f"{label} (-j {n_jobs})"

    if n_jobs <= 1 or len(tasks) <= 1:
        rows = tasks
        if progress:
            rows = tqdm(rows, total=len(tasks), desc=label, unit="song")
        for item in rows:
            if not _verify_song_complete_task(item):
                return False
        return True

    chunksize = max(1, min(16, len(tasks) // (n_jobs * 8) or 1))
    pbar = tqdm(
        total=len(tasks),
        desc=label,
        unit="song",
        miniters=1,
        smoothing=0.05,
        disable=not progress,
    )
    try:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            for ok in pool.imap_unordered(
                _verify_song_complete_task,
                tasks,
                chunksize=chunksize,
            ):
                pbar.update(1)
                if not ok:
                    pool.terminate()
                    return False
    finally:
        pbar.close()
    return True


def require_raw_synthesis(
    source_dir: str,
    *,
    run_command: str,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    expected_n_songs: int | None = None,
    jobs: int = 1,
) -> None:
    """Raise if the non-realify synthesis pass has not completed successfully."""
    if synthesis_is_complete(
        source_dir,
        audio_format,
        require_mixture=False,
        expected_n_songs=expected_n_songs,
        progress=True,
        jobs=jobs,
    ):
        return
    source = Path(source_dir)
    data_csv = source / f"{DATA_DIR_NAME}.csv"
    have = 0
    if data_csv.is_file():
        try:
            have = len(pd.read_csv(data_csv, usecols=["path"]))
        except Exception:
            have = 0
    detail_parts: list[str] = []
    if expected_n_songs is not None:
        detail_parts.append(
            f"Need {expected_n_songs} songs in data.csv "
            f"(have {have}). Fluidsynth and MIDI-DDSP must both finish first."
        )
        if have and have < int(expected_n_songs):
            pending = _fluidsynth_pending_unclaimed_counts_by_path(source)
            n_pending = sum(pending.values())
            if n_pending:
                detail_parts.append(
                    f"{n_pending} Fluidsynth pending_midi_ddsp tracks still lack "
                    "a MIDI-DDSP recipe/audio — re-run midi_ddsp."
                )
    detail = (" " + " ".join(detail_parts)) if detail_parts else ""
    raise RuntimeError(
        "Raw stems are missing or incomplete at "
        f"{source_dir}.{detail}\n"
        "Run the corresponding non-realify ablation first:\n"
        f"  {run_command}"
    )


def require_donor_ablation(args, *, realify: bool) -> None:
    """Ensure the soundfont-fallback donor ablation exists for DDSP modes."""
    donor_mode = fallback_donor_mode(args.render_mode)
    if donor_mode is None:
        return
    audio_format = synthesis_audio_format(args.flac)
    if realify:
        donor_dir = ablation_realify_dir(args.output_dir, donor_mode)
        cmd = (
            f"uv run python -m synthesis.synthesize --render-mode {donor_mode} --realify"
        )
    else:
        donor_dir = ablation_raw_dir(args.output_dir, donor_mode)
        cmd = f"uv run python -m synthesis.synthesize --render-mode {donor_mode}"
    if args.flac:
        cmd += " --flac"
    if synthesis_is_complete(donor_dir, audio_format, require_mixture=False):
        return
    if getattr(args, "allow_fallback_render", False) and not realify:
        print(
            f"Warning: donor ablation incomplete at {donor_dir}; "
            "--allow-fallback-render will Fluidsynth-render missing stems."
        )
        return
    kind = "realify" if realify else "raw"
    raise RuntimeError(
        f"Cannot run {args.render_mode}{' --realify' if realify else ''}: "
        f"donor {kind} ablation incomplete at {donor_dir}\n"
        f"Run first:\n  {cmd}"
    )


def raw_synthesis_command(args) -> str:
    cmd = f"uv run python -m synthesis.synthesize --render-mode {args.render_mode}"
    if args.full:
        cmd += " --full"
    if args.flac:
        cmd += " --flac"
    return cmd


def run_realify_pass(args, source_dir: str, dest_dir: str, *, allowed_song_ids: set[str] | None = None):
    from synthesis.realify.realify import run_realify

    audio_format = synthesis_audio_format(args.flac)
    content_fidelity_enforce = REALIFY_CONTENT_FIDELITY_ENFORCE
    if getattr(args, "content_fidelity_enforce", False):
        content_fidelity_enforce = True
    if getattr(args, "no_content_fidelity_enforce", False):
        content_fidelity_enforce = False

    in_place = Path(source_dir).resolve() == Path(dest_dir).resolve()
    run_realify(
        source_dir=source_dir,
        output_dir=dest_dir,
        model=args.model,
        limit=args.realify_limit,
        jobs=args.jobs,
        batch_size=(
            REALIFY_BATCH_SIZE
            if args.realify_batch_size is None
            else args.realify_batch_size
        ),
        audio_format=audio_format,
        sample_seed=args.sample_seed,
        reset=bool(args.reset) and not in_place,
        silence_enforce=REALIFY_SILENCE_ENFORCE and not args.no_silence_enforce,
        content_fidelity_enforce=content_fidelity_enforce,
        output_root=args.output_dir,
        render_mode=args.render_mode,
        category_allowlist=(
            set(_hybrid_recipe(args).realify_categories())
            if _hybrid_recipe(args) is not None
            else None
        ),
        recipe=_hybrid_recipe(args),
        allowed_song_ids=allowed_song_ids,
    )


def main():
    from synthesis.mix import print_mix_hint

    args = parse_args()
    if args.full:
        source_dir = full_stems_dir(args.output_dir)
        dest_dir = full_stems_realify_dir(args.output_dir)
    else:
        source_dir = ablation_raw_dir(args.output_dir, args.render_mode)
        dest_dir = ablation_realify_dir(args.output_dir, args.render_mode)

    stems_dir = dest_dir if args.realify else source_dir
    if args.realify:
        audio_format = synthesis_audio_format(args.flac)
        require_raw_synthesis(
            source_dir,
            run_command=raw_synthesis_command(args),
            audio_format=audio_format,
        )
        if uses_ddsp(args.render_mode) and not args.full:
            require_donor_ablation(args, realify=True)
        run_realify_pass(args, source_dir, dest_dir)
    else:
        run_synthesis(args, source_dir)

    link_ablations_in_repo(args.output_dir)
    print_mix_hint(stems_dir, jobs=args.jobs, flac=bool(args.flac))


if __name__ == "__main__":
    main()
