"""Build verification catalogs and resolve manual winner picks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from experiments.listening.aggregate import (
    DEFAULT_CONTENT_MEAN_THRESHOLD,
    DEFAULT_CONTENT_THRESHOLD,
    aggregate_winners,
    ratings_dataframe,
)

if TYPE_CHECKING:
    from experiments.listening.catalog import SweepCatalog

PRESET_VERIFY_SOURCE = "winners.yaml"
PATCH_VERIFY_SOURCE = "winners.yaml"

# Preferred probe stem when building one representative verify-swipe clip per soundfont.
VERIFY_SWIPE_PREFERRED_STEMS = {
    "organ": "organ_2",  # GM 19 church/hymn organ instead of GM 22 percussive
    "strings": "cello",  # GM 48 string ensemble instead of GM 44 tremolo probes
}


def variant_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    per_stem = (
        df.groupby(["stem_id", "category", "variant_id"], dropna=False)
        .agg(content=("content", "mean"), realism=("realism", "mean"))
        .reset_index()
    )

    return (
        per_stem.groupby(["category", "variant_id"], dropna=False)
        .agg(
            mean_content=("content", "mean"),
            mean_realism=("realism", "mean"),
            min_content=("content", "min"),
            n_stems=("stem_id", "nunique"),
        )
        .reset_index()
    )


def filter_eligible(
    stats: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
) -> pd.DataFrame:
    if stats.empty:
        return stats

    eligible = stats[
        (stats["min_content"] >= content_threshold)
        & (stats["mean_content"] >= content_mean_threshold)
    ]
    if eligible.empty:
        return stats.copy()
    return eligible


def auto_winners(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame()

    winners = []
    for category, group in eligible.groupby("category", dropna=False):
        best = group.sort_values(
            ["mean_realism", "mean_content"],
            ascending=[False, False],
        ).iloc[0]
        winners.append({
            "category": category,
            "variant_id": best["variant_id"],
            "mean_content": round(float(best["mean_content"]), 2),
            "mean_realism": round(float(best["mean_realism"]), 2),
            "n_stems": int(best["n_stems"]),
        })
    return pd.DataFrame(winners).sort_values("category")


def analyze_responses(
    responses: dict,
    *,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ratings_df, variant_stats, eligible, auto_winners)."""
    df = ratings_dataframe(responses)
    stats = variant_stats(df)
    eligible = filter_eligible(
        stats,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )
    _, winners = aggregate_winners(
        df,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )
    return df, stats, eligible, winners


def _variant_config(catalog: SweepCatalog, variant_id: str) -> dict:
    manifest = catalog._manifest
    if manifest.empty:
        return {"variant_id": variant_id}
    match = manifest[manifest["variant_id"] == variant_id]
    if match.empty:
        return {"variant_id": variant_id}
    row = match.iloc[0]
    if catalog.sweep_type == "preset":
        config = {
            "variant_id": variant_id,
            "init_noise_level": float(row["init_noise_level"]),
            "prompt_variant": row["prompt_variant"],
        }
        if "steps" in row and pd.notna(row["steps"]):
            config["steps"] = int(row["steps"])
        if "cfg_scale" in row and pd.notna(row["cfg_scale"]):
            config["cfg_scale"] = float(row["cfg_scale"])
        return config

    config = {"variant_id": variant_id}
    for key in ("soundfont_id", "fx_profile", "pool_id", "phase"):
        if key in row and str(row[key]):
            config[key] = row[key]
    return config


def _stats_row(stats: pd.DataFrame, category: str, variant_id: str) -> dict | None:
    if stats.empty:
        return None
    match = stats[
        (stats["category"] == category) & (stats["variant_id"] == variant_id)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    result = {
        "mean_content": round(float(row["mean_content"]), 2),
        "mean_realism": round(float(row["mean_realism"]), 2),
        "min_content": round(float(row["min_content"]), 2),
        "n_stems": int(row["n_stems"]),
    }
    if "mean_rating" in row and pd.notna(row["mean_rating"]):
        result["mean_rating"] = round(float(row["mean_rating"]), 2)
    return result


def build_patch_shortlist_verification_meta(
    catalog: SweepCatalog,
    responses: dict,
    *,
    source_responses: str,
    shortlists: dict[str, list[str]],
    verification_phase: str | None = None,
) -> dict:
    """Build verification meta for reviewing phase-1 soundfont shortlists."""
    from experiments.listening.aggregate import (
        DEFAULT_MEAN_RATING_THRESHOLD,
        ratings_dataframe,
        shortlist_dataframe,
    )

    df = ratings_dataframe(responses)
    stats = shortlist_dataframe(df)

    categories = []
    for category in sorted(shortlists):
        shortlist = [str(v) for v in shortlists[category] if v]
        if not shortlist:
            continue
        category_stats = stats[stats["category"] == category] if not stats.empty else stats
        variants = []
        for variant_id in shortlist:
            stat_row = None
            if not category_stats.empty:
                match = category_stats[category_stats["variant_id"] == variant_id]
                if not match.empty:
                    stat_row = match.iloc[0]
            variant_stats = _stats_row(stats, category, variant_id) if not stats.empty else None
            variants.append({
                "variant_id": variant_id,
                "passed_filter": True,
                "is_auto_winner": False,
                "stats": variant_stats,
                "config": {
                    "variant_id": variant_id,
                    "soundfont_id": variant_id,
                    "fx_profile": "dry",
                },
            })
        categories.append({
            "category": category,
            "shortlist": shortlist,
            "auto_winner_variant_id": None,
            "variants": variants,
        })

    return {
        "sweep_type": catalog.sweep_type,
        "manifest_id": catalog.manifest_id(),
        "source_responses": source_responses,
        "verification_phase": verification_phase,
        "verification_mode": "soundfont_shortlist",
        "mode": "final",
        "mean_rating_threshold": DEFAULT_MEAN_RATING_THRESHOLD,
        "categories": categories,
    }


def build_patch_verify_swipe_cards(
    catalog: SweepCatalog,
    shortlists: dict[str, list[str]],
) -> list[dict]:
    """One listening card per shortlisted soundfont (representative 10s clip)."""
    from experiments.listening.catalog import song_id_from_manifest_row

    clip_manifest = catalog._clip_manifest
    if clip_manifest.empty:
        raise RuntimeError(
            "Clip manifest missing. Run make_clips on phase-1 soundfont output first."
        )

    probes = catalog._probe_by_id
    cards: list[dict] = []
    for category in sorted(shortlists):
        for soundfont_id in shortlists[category]:
            if not soundfont_id:
                continue
            rows = clip_manifest[
                (clip_manifest["category"].astype(str) == str(category))
                & (clip_manifest["variant_id"].astype(str) == str(soundfont_id))
            ]
            if rows.empty:
                continue
            preferred_stem = VERIFY_SWIPE_PREFERRED_STEMS.get(str(category))
            if preferred_stem:
                preferred_rows = rows[rows["stem_id"].astype(str) == str(preferred_stem)]
                if not preferred_rows.empty:
                    rows = preferred_rows
            rows = rows.sort_values(["stem_id", "clip_index"])
            row = rows.iloc[0]
            variant_id = str(row["variant_id"])
            clip_id = str(row["clip_id"])
            stem_id = str(row["stem_id"])
            probe = probes.get(stem_id, {})
            clip_index = int(row.get("clip_index") or 0)
            n_clips = int(
                clip_manifest[
                    (clip_manifest["stem_id"] == stem_id)
                    & (clip_manifest["variant_id"] == variant_id)
                ]["clip_id"].nunique()
            )
            out_path = Path(str(row["out_path"]))
            cards.append({
                "card_id": f"{category}|{variant_id}|{clip_id}",
                "category": str(category),
                "soundfont_id": variant_id,
                "variant_id": variant_id,
                "clip_id": clip_id,
                "stem_id": stem_id,
                "track": int(row.get("track") or probe.get("track") or 0),
                "clip_index": clip_index,
                "clips_per_stem": n_clips,
                "label": (
                    f"{category} · {variant_id} · "
                    f"{probe.get('note') or stem_id} · clip {clip_index + 1}/{n_clips}"
                ),
                "audio": catalog._variant_cell_from_row(row),
                "filename": out_path.name,
                "song_id": song_id_from_manifest_row(row),
            })
    return cards


def _safe_verification_token(name: str | None) -> str:
    if not name:
        return ""
    safe = str(name).replace("/", "_").replace("\\", "_")
    if safe.endswith(".json"):
        safe = safe[: -len(".json")]
    return safe


def patch_verify_swipe_session_id(
    catalog: SweepCatalog,
    *,
    pass_number: int = 1,
    source_verification: str | None = None,
) -> str:
    session_id = f"{catalog.sweep_dir.resolve().name}_verify_pass{int(pass_number)}"
    token = _safe_verification_token(source_verification)
    if token:
        session_id = f"{session_id}_from_{token}"
    return session_id


def shortlists_from_verification_doc(doc: dict) -> dict[str, list[str]]:
    """Approved soundfonts per category from a prior verification pass."""
    return winners_from_verification(doc, sweep_type="patch")


def resolve_verification_path(catalog: SweepCatalog, name: str) -> Path | None:
    """Resolve a verification JSON under the catalog responses dir."""
    return resolve_responses_path(catalog, name)


def list_verification_files(catalog: SweepCatalog) -> list[dict]:
    """Final verification JSON files available as multi-pass inputs."""
    responses_dir = catalog.responses_dir()
    files = []
    for path in sorted(responses_dir.glob("verification_final_*.json"), reverse=True):
        if "in_progress" in path.name:
            continue
        stat = path.stat()
        files.append({
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return files


def category_counts_from_votes(
    cards: list[dict],
    votes: list[dict] | dict[str, dict],
) -> dict[str, dict[str, int]]:
    """Per-category left / accepted / rejected counts for verify swipe."""
    vote_by_card: dict[str, dict]
    if isinstance(votes, dict):
        vote_by_card = {str(k): v for k, v in votes.items() if isinstance(v, dict)}
    else:
        vote_by_card = {}
        for vote in votes:
            card_id = vote.get("card_id")
            if card_id:
                vote_by_card[str(card_id)] = vote

    counts: dict[str, dict[str, int]] = {}
    for card in cards:
        category = str(card.get("category") or "")
        if not category:
            continue
        entry = counts.setdefault(
            category,
            {"total": 0, "left": 0, "accepted": 0, "rejected": 0},
        )
        entry["total"] += 1
        vote = vote_by_card.get(str(card.get("card_id") or ""))
        if not vote:
            entry["left"] += 1
        elif str(vote.get("tier")) == "strong_accept":
            entry["accepted"] += 1
        else:
            entry["rejected"] += 1
    return counts


def build_patch_verify_swipe_meta(
    catalog: SweepCatalog,
    shortlists: dict[str, list[str]],
    *,
    order: str = "sequential",
    seed: int = 42,
    verification_phase: str | None = None,
    pass_number: int = 1,
    source_verification: str | None = None,
) -> dict:
    from experiments.listening.swipe import order_swipe_cards, swipe_storage_key

    cards = build_patch_verify_swipe_cards(catalog, shortlists)
    cards = order_swipe_cards(cards, order=order, seed=seed)
    session_id = patch_verify_swipe_session_id(
        catalog,
        pass_number=pass_number,
        source_verification=source_verification,
    )
    categories = sorted({str(card["category"]) for card in cards if card.get("category")})
    normalized_shortlists = {
        str(category): [str(v) for v in values if v]
        for category, values in shortlists.items()
        if values
    }
    return {
        "sweep_type": catalog.sweep_type,
        "mode": "verification",
        "verification_mode": "soundfont_shortlist_swipe",
        "available": catalog.available(),
        "manifest_id": catalog.clip_manifest_id() or catalog.manifest_id(),
        "session_id": session_id,
        "storage_key": swipe_storage_key(catalog.sweep_type, session_id),
        "source_responses": PATCH_VERIFY_SOURCE,
        "source_verification": source_verification,
        "pass": int(pass_number),
        "verification_phase": verification_phase,
        "order": order,
        "seed": seed,
        "total_cards": len(cards),
        "categories": categories,
        "shortlists": normalized_shortlists,
        "category_totals": {
            category: len(normalized_shortlists.get(category, []))
            for category in categories
        },
        "cards": cards,
        "keyboard": {
            "strong_reject": "ArrowLeft",
            "strong_accept": "ArrowRight",
            "replay": "Space",
            "undo": "Backspace",
        },
    }


def verification_from_patch_swipe_votes(
    votes: list[dict],
    shortlists: dict[str, list[str]],
    *,
    source_responses: str = PATCH_VERIFY_SOURCE,
    pass_number: int = 1,
    source_verification: str | None = None,
) -> dict:
    """Convert binary swipe votes into patch verification JSON for lock."""
    vote_index = {
        (str(vote.get("category")), str(vote.get("variant_id"))): vote
        for vote in votes
        if vote.get("category") and vote.get("variant_id")
    }
    categories = []
    for category in sorted(shortlists):
        approved: list[str] = []
        for soundfont_id in shortlists[category]:
            vote = vote_index.get((str(category), str(soundfont_id)))
            if vote and str(vote.get("tier")) == "strong_accept":
                approved.append(str(soundfont_id))
        categories.append({
            "category": str(category),
            "approved": approved,
        })
    return {
        "mode": "verification",
        "sweep_type": "patch",
        "verification_mode": "soundfont_shortlist_swipe",
        "source_responses": source_responses,
        "source_verification": source_verification,
        "pass": int(pass_number),
        "categories": categories,
    }


def patch_verify_swipe_session_path(
    catalog: SweepCatalog,
    *,
    pass_number: int = 1,
    source_verification: str | None = None,
) -> Path:
    """In-progress swipe session; pass-aware so multi-pass runs don't collide.

    Pass 1 with no prior verification keeps the legacy filename for compatibility.
    """
    if int(pass_number) == 1 and not source_verification:
        return catalog.responses_dir() / "verification_swipe_in_progress.json"
    token = _safe_verification_token(source_verification)
    if token:
        return (
            catalog.responses_dir()
            / f"verification_swipe_pass{int(pass_number)}_from_{token}_in_progress.json"
        )
    return (
        catalog.responses_dir()
        / f"verification_swipe_pass{int(pass_number)}_in_progress.json"
    )


def build_preset_realify_verification_meta(
    catalog: SweepCatalog,
    *,
    category_winners: dict[str, str],
    source_responses: str = PRESET_VERIFY_SOURCE,
    composed_config_fn=None,
    verification_phase: str | None = None,
) -> dict:
    """Build verification meta from locked phase winners (no blind-test responses)."""
    categories = []
    for category in sorted(category_winners):
        variant_id = str(category_winners[category])
        config = (
            composed_config_fn(category, variant_id)
            if composed_config_fn is not None
            else {"variant_id": variant_id}
        )
        categories.append({
            "category": category,
            "auto_winner_variant_id": variant_id,
            "variants": [{
                "variant_id": variant_id,
                "passed_filter": True,
                "is_auto_winner": True,
                "stats": None,
                "config": config,
            }],
        })

    return {
        "sweep_type": catalog.sweep_type,
        "manifest_id": catalog.manifest_id(),
        "source_responses": source_responses,
        "verification_phase": verification_phase,
        "verification_mode": "preset_realify",
        "mode": "final",
        "categories": categories,
    }


def build_verification_meta(
    catalog: SweepCatalog,
    responses: dict,
    *,
    source_responses: str,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
    composed_config_fn=None,
    verification_phase: str | None = None,
) -> dict:
    _, stats, eligible, winners = analyze_responses(
        responses,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )

    eligible_keys = set()
    if not eligible.empty:
        for _, row in eligible.iterrows():
            eligible_keys.add((str(row["category"]), str(row["variant_id"])))

    auto_winner_by_category = {}
    if not winners.empty:
        for _, row in winners.iterrows():
            auto_winner_by_category[str(row["category"])] = str(row["variant_id"])

    categories = []
    if not stats.empty:
        for category in sorted(stats["category"].dropna().unique()):
            category = str(category)
            category_stats = stats[stats["category"] == category].sort_values(
                ["mean_realism", "mean_content"],
                ascending=[False, False],
            )
            variants = []
            for _, row in category_stats.iterrows():
                variant_id = str(row["variant_id"])
                passed_filter = (category, variant_id) in eligible_keys
                variants.append({
                    "variant_id": variant_id,
                    "passed_filter": passed_filter,
                    "is_auto_winner": auto_winner_by_category.get(category) == variant_id,
                    "stats": _stats_row(stats, category, variant_id),
                    "config": (
                        composed_config_fn(category, variant_id)
                        if composed_config_fn is not None
                        else _variant_config(catalog, variant_id)
                    ),
                })
            categories.append({
                "category": category,
                "auto_winner_variant_id": auto_winner_by_category.get(category),
                "variants": variants,
            })

    return {
        "sweep_type": catalog.sweep_type,
        "manifest_id": catalog.manifest_id(),
        "source_responses": source_responses,
        "verification_phase": verification_phase,
        "mode": "final",
        "content_threshold": content_threshold,
        "content_mean_threshold": content_mean_threshold,
        "categories": categories,
    }


def _stem_metadata_lookup(catalog: SweepCatalog) -> dict[tuple[str, int], dict]:
    from shared.config import STEMS_FILE_NAME
    from synthesis.listening.catalog import song_id_from_path

    stems_csv = catalog.source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.is_file():
        return {}

    lookup: dict[tuple[str, int], dict] = {}
    stems = pd.read_csv(stems_csv)
    for _, row in stems.iterrows():
        song_id = song_id_from_path(str(row["path"]))
        track = int(row["track"])
        lookup[(song_id, track)] = {
            "track_name": str(row.get("name") or "").strip() or None,
            "program": int(row.get("program", 0) or 0),
            "is_drum": bool(row.get("is_drum", False)),
        }
    return lookup


def build_category_verification(
    catalog: SweepCatalog,
    category: str,
    variant_ids: list[str],
    *,
    composed_config_fn=None,
) -> dict | None:
    if catalog._manifest.empty:
        return None

    group = catalog._manifest[catalog._manifest["category"] == category]
    if group.empty:
        return None

    metadata_lookup = _stem_metadata_lookup(catalog)
    stems = []
    for stem_id, stem_group in group.groupby("stem_id"):
        first = stem_group.iloc[0]
        from synthesis.listening.catalog import song_id_from_path

        track = int(first["track"])
        from synthesis.audio import stem_filename

        filename = catalog._reference_filename(stem_id, track)
        probe = catalog._probe_by_id.get(stem_id, {})
        song_id = song_id_from_path(str(first["path"]))
        meta = metadata_lookup.get((song_id, track), {})
        track_name = probe.get("note") or meta.get("track_name")
        stems.append({
            "id": stem_id,
            "track": track,
            "note": probe.get("note"),
            "track_name": track_name,
            "program": meta.get("program", 0),
            "is_drum": meta.get("is_drum", False),
            "reference": catalog._reference_cell(stem_id, filename),
        })

    stems.sort(key=lambda row: row["id"])
    variants = []
    for variant_id in variant_ids:
        row = group[group["variant_id"] == variant_id]
        if row.empty:
            continue
        first = row.iloc[0]
        from synthesis.listening.catalog import song_id_from_path

        audio_by_stem = {}
        for stem in stems:
            stem_id = stem["id"]
            stem_row = row[row["stem_id"] == stem_id]
            if stem_row.empty:
                continue
            song_id = song_id_from_path(stem_row.iloc[0]["path"])
            track = int(stem_row.iloc[0]["track"])
            from synthesis.audio import stem_filename

            filename = stem_filename(track, catalog._audio_format)
            audio_by_stem[stem_id] = catalog._variant_cell(variant_id, song_id, filename)

        variants.append({
            "variant_id": variant_id,
            "config": (
                composed_config_fn(category, variant_id)
                if composed_config_fn is not None
                else _variant_config(catalog, variant_id)
            ),
            "audio_by_stem": audio_by_stem,
        })

    return {
        "category": category,
        "stems": stems,
        "variants": variants,
    }


def list_response_files(catalog: SweepCatalog) -> list[dict]:
    responses_dir = catalog.responses_dir()
    files = []
    for path in sorted(responses_dir.glob("responses_*.json"), reverse=True):
        if path.name == "responses_in_progress.json":
            continue
        stat = path.stat()
        files.append({
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return files


def resolve_responses_path(catalog: SweepCatalog, name: str) -> Path | None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    path = (catalog.responses_dir() / name).resolve()
    if not str(path).startswith(str(catalog.responses_dir().resolve())):
        return None
    return path if path.is_file() else None


def winners_from_verification(doc: dict, *, sweep_type: str | None = None) -> dict:
    sweep_type = sweep_type or doc.get("sweep_type")
    if sweep_type == "patch":
        shortlists: dict[str, list[str]] = {}
        for entry in doc.get("categories", []):
            category = entry.get("category")
            approved = entry.get("approved") or []
            if category and approved:
                shortlists[str(category)] = [str(value) for value in approved]
        return shortlists

    winners = {}
    for entry in doc.get("categories", []):
        category = entry.get("category")
        if not category or entry.get("bypass_realify"):
            continue
        winner = entry.get("winner_variant_id")
        if winner:
            winners[str(category)] = str(winner)
    return winners


def bypass_realify_from_verification(doc: dict) -> dict[str, bool]:
    """Categories where every stem bypasses SA3 (or category shortcut was used)."""
    bypass = {}
    for entry in doc.get("categories", []):
        category = entry.get("category")
        if not category:
            continue
        stems = entry.get("stems") or []
        if stems:
            if all(bool(stem.get("bypass_realify")) for stem in stems):
                bypass[str(category)] = True
            continue
        if entry.get("bypass_realify"):
            bypass[str(category)] = True
    return bypass


def bypass_routing_rules_from_verification(doc: dict) -> list[dict]:
    """Per-instrument bypass rules for partial category bypass."""
    del doc
    from experiments.listening_shared.clips import PRESET_SWEEP_REMOVED

    raise RuntimeError(PRESET_SWEEP_REMOVED)


def validate_verification_entry(
    entry: dict,
    *,
    sweep_type: str | None = None,
) -> list[str]:
    errors = []
    category = entry.get("category")
    if not category:
        errors.append("missing category")
        return errors

    approved = entry.get("approved") or []
    if sweep_type == "patch":
        if not approved:
            errors.append(f"{category}: keep at least one soundfont")
        return errors

    verification_mode = entry.get("_verification_mode")
    if verification_mode == "preset_realify" or entry.get("bypass_realify"):
        return errors

    winner = entry.get("winner_variant_id")
    if not approved:
        errors.append(f"{category}: approve at least one variant")
    if not winner:
        errors.append(f"{category}: pick a winner")
    elif winner not in approved:
        errors.append(f"{category}: winner must be among approved variants")
    return errors


def validate_verification(doc: dict) -> list[str]:
    errors = []
    sweep_type = doc.get("sweep_type")
    verification_mode = doc.get("verification_mode")
    for entry in doc.get("categories", []):
        entry_with_mode = dict(entry)
        entry_with_mode["_verification_mode"] = verification_mode
        errors.extend(validate_verification_entry(entry_with_mode, sweep_type=sweep_type))
    return errors


def verification_in_progress_path(catalog: SweepCatalog, source_responses: str) -> Path:
    safe = source_responses.replace("/", "_").replace("\\", "_")
    if safe.endswith(".json"):
        safe = safe[: -len(".json")]
    return catalog.responses_dir() / f"verification_in_progress_{safe}.json"
