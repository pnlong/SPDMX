"""Aggregate sweep listening-test responses into per-category winners."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from experiments.listening.catalog import SweepCatalog, default_sweep_dir
from experiments.paths import DEFAULT_PROBE_STEMS

DEFAULT_CONTENT_THRESHOLD = 3.0
DEFAULT_CONTENT_MEAN_THRESHOLD = 3.5
DEFAULT_MEAN_RATING_THRESHOLD = 4.1
DEFAULT_REALISM_THRESHOLD = 4.0
DEFAULT_NOISE_CONTENT_THRESHOLD = 4.5
DEFAULT_MIN_SWIPE_WINNERS = 3
SWIPE_TIERS = ("strong_reject", "weak_reject", "weak_accept", "strong_accept")
SWIPE_ACCEPT_TIERS = frozenset({"weak_accept", "strong_accept"})


def load_responses(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ratings_dataframe(responses: dict) -> pd.DataFrame:
    rows = []
    for entry in responses.get("ratings", []):
        stem_id = entry["stem_id"]
        category = entry.get("category")
        for rating in entry.get("samples", []):
            rows.append({
                "stem_id": stem_id,
                "category": category,
                "variant_id": rating["variant_id"],
                "content": float(rating["content"]),
                "realism": float(rating["realism"]),
            })
    return pd.DataFrame(rows)


def aggregate_winners(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_CONTENT_THRESHOLD,
    content_mean_threshold: float = DEFAULT_CONTENT_MEAN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_stem_scores, per_category_winners)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    from experiments.listening.verification import auto_winners, filter_eligible, variant_stats

    per_stem = (
        df.groupby(["stem_id", "category", "variant_id"], dropna=False)
        .agg(content=("content", "mean"), realism=("realism", "mean"))
        .reset_index()
    )

    stats = variant_stats(df)
    eligible = filter_eligible(
        stats,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )
    winners = auto_winners(eligible)

    return per_stem, winners


def _noise_level_from_variant_id(variant_id: str) -> float:
    prefix = "noise"
    if str(variant_id).startswith(prefix):
        return float(str(variant_id)[len(prefix):])
    return float("-inf")


def noise_level_winners(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> pd.DataFrame:
    """Pick phase-1 noise winners: mean content >= threshold, then highest realism.

    Tie-break equal realism by higher init_noise_level.
    """
    if df.empty:
        return pd.DataFrame()

    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["init_noise_level"] = stats["variant_id"].map(_noise_level_from_variant_id)

    def _rank(group: pd.DataFrame) -> pd.DataFrame:
        return group.sort_values(
            ["mean_realism", "init_noise_level", "mean_content", "variant_id"],
            ascending=[False, False, False, True],
        )

    winners = []
    for category, group in stats.groupby("category", dropna=False):
        passing = group[group["mean_content"] >= content_threshold]
        ranked = _rank(passing if not passing.empty else group)
        best = ranked.iloc[0]
        winners.append({
            "category": category,
            "variant_id": best["variant_id"],
            "mean_content": round(float(best["mean_content"]), 2),
            "mean_realism": round(float(best["mean_realism"]), 2),
            "n_stems": int(best["n_stems"]),
            "passed_content_threshold": bool(best["mean_content"] >= content_threshold),
        })
    return pd.DataFrame(winners).sort_values("category")


def noise_level_dataframe(
    df: pd.DataFrame,
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> pd.DataFrame:
    """Tabular phase-1 noise stats with threshold flag for reporting."""
    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["passed_content_threshold"] = stats["mean_content"] >= content_threshold
    return stats.sort_values(["category", "mean_realism"], ascending=[True, False])


def noise_audit_candidates(
    phase1_winners: dict[str, str],
) -> dict[str, set[str]]:
    """Per-category variant ids to compare in phase-1b (winner vs one-step-lower)."""
    from experiments.listening_shared.clips import PRESET_SWEEP_REMOVED

    raise RuntimeError(PRESET_SWEEP_REMOVED)


def noise_audit_winners(
    df: pd.DataFrame,
    phase1_winners: dict[str, str],
    *,
    content_threshold: float = DEFAULT_NOISE_CONTENT_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pick audit winners after production silence enforcement: content gate, then realism."""
    del df, phase1_winners, content_threshold
    from experiments.listening_shared.clips import PRESET_SWEEP_REMOVED

    raise RuntimeError(PRESET_SWEEP_REMOVED)


def shortlist_variants(
    df: pd.DataFrame,
    *,
    mean_rating_threshold: float = DEFAULT_MEAN_RATING_THRESHOLD,
    realism_threshold: float | None = None,
) -> dict[str, list[str]]:
    """Return per-category variant shortlists.

    When ``realism_threshold`` is set, shortlist by mean realism only (patch phase 1).
    Otherwise use mean(content, realism) / 2 >= ``mean_rating_threshold``.
    """
    if df.empty:
        return {}

    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return {}

    stats = stats.copy()
    stats["mean_rating"] = (stats["mean_content"] + stats["mean_realism"]) / 2

    shortlists: dict[str, list[str]] = {}
    for category, group in stats.groupby("category", dropna=False):
        if realism_threshold is not None:
            passing = group[group["mean_realism"] >= realism_threshold].sort_values(
                ["mean_realism", "variant_id"],
                ascending=[False, True],
            )
        else:
            passing = group[group["mean_rating"] >= mean_rating_threshold].sort_values(
                ["mean_rating", "mean_realism", "mean_content"],
                ascending=[False, False, False],
            )
        if passing.empty:
            if realism_threshold is not None:
                best = group.sort_values(
                    ["mean_realism", "variant_id"],
                    ascending=[False, True],
                ).iloc[0]
            else:
                best = group.sort_values(
                    ["mean_rating", "mean_realism", "mean_content"],
                    ascending=[False, False, False],
                ).iloc[0]
            passing = group[group["variant_id"] == best["variant_id"]]

        shortlists[str(category)] = [str(v) for v in passing["variant_id"].tolist()]

    return shortlists


def shortlist_dataframe(
    df: pd.DataFrame,
    *,
    mean_rating_threshold: float = DEFAULT_MEAN_RATING_THRESHOLD,
    realism_threshold: float | None = None,
) -> pd.DataFrame:
    """Tabular shortlist with stats for reporting."""
    from experiments.listening.verification import variant_stats

    stats = variant_stats(df)
    if stats.empty:
        return stats

    stats = stats.copy()
    stats["mean_rating"] = (stats["mean_content"] + stats["mean_realism"]) / 2
    if realism_threshold is not None:
        stats["shortlisted"] = stats["mean_realism"] >= realism_threshold
    else:
        stats["shortlisted"] = stats["mean_rating"] >= mean_rating_threshold
    return stats.sort_values(["category", "mean_rating"], ascending=[True, False])


def votes_dataframe(responses: dict) -> pd.DataFrame:
    rows = []
    for vote in responses.get("votes", []):
        rows.append({
            "card_id": vote.get("card_id"),
            "category": vote.get("category"),
            "stem_id": vote.get("stem_id"),
            "clip_id": vote.get("clip_id"),
            "variant_id": vote.get("variant_id"),
            "tier": vote.get("tier"),
        })
    return pd.DataFrame(rows)


def _variant_tier_vote_count(
    group: pd.DataFrame,
    variant_id: str,
    *,
    tiers: frozenset[str],
) -> int:
    return sum(
        1
        for _, row in group.iterrows()
        if str(row["variant_id"]) == variant_id and str(row["tier"]) in tiers
    )


def _rank_variants_by_tier_votes(
    group: pd.DataFrame,
    variant_ids: list[str],
    *,
    tiers: frozenset[str],
) -> list[str]:
    return sorted(
        variant_ids,
        key=lambda variant_id: (
            -_variant_tier_vote_count(group, variant_id, tiers=tiers),
            variant_id,
        ),
    )


def swipe_winners(
    responses: dict,
    *,
    min_winners: int = DEFAULT_MIN_SWIPE_WINNERS,
    max_winners: int | None = None,
    strict: bool = True,
    exclude_legacy_stems: bool = True,
) -> dict[str, list[str]]:
    """Select per-category variant shortlists from swipe votes.

    When ``max_winners`` is set, cap the strong/weak accept pools after ranking.
    Phase 1 and phase 2 both keep every variant with at least one strong accept
    unless ``max_winners`` is passed explicitly.
    """
    from experiments.probe_stems import legacy_stem_ids

    df = votes_dataframe(responses)
    if df.empty:
        return {}

    if exclude_legacy_stems:
        legacy = legacy_stem_ids()
        if legacy and "stem_id" in df.columns:
            df = df[~df["stem_id"].astype(str).isin(legacy)]
        if df.empty:
            return {}

    winners: dict[str, list[str]] = {}
    for category, group in df.groupby("category", dropna=False):
        category = str(category)
        variant_tiers: dict[str, set[str]] = {}
        for _, row in group.iterrows():
            variant_id = str(row["variant_id"])
            tier = str(row["tier"])
            if tier not in SWIPE_TIERS:
                continue
            variant_tiers.setdefault(variant_id, set()).add(tier)

        strong_pool = sorted(
            variant_id
            for variant_id, tiers in variant_tiers.items()
            if "strong_accept" in tiers
        )
        if strong_pool:
            if max_winners == 1:
                strong_pool = _rank_variants_by_tier_votes(
                    group,
                    strong_pool,
                    tiers=frozenset({"strong_accept"}),
                )[:1]
            elif max_winners is not None:
                strong_pool = _rank_variants_by_tier_votes(
                    group,
                    strong_pool,
                    tiers=frozenset({"strong_accept"}),
                )[:max_winners]
            winners[category] = strong_pool
            continue

        weak_pool = [
            variant_id
            for variant_id, tiers in variant_tiers.items()
            if "weak_accept" in tiers
        ]
        weak_pool = _rank_variants_by_tier_votes(
            group,
            weak_pool,
            tiers=SWIPE_ACCEPT_TIERS,
        )
        if weak_pool:
            cap = max(min_winners, 1)
            if max_winners is not None:
                cap = min(cap, max_winners)
            winners[category] = weak_pool[:cap]
            continue

        weak_reject_pool = sorted(
            variant_id
            for variant_id, tiers in variant_tiers.items()
            if "weak_reject" in tiers
        )
        strong_reject_pool = sorted(
            variant_id
            for variant_id, tiers in variant_tiers.items()
            if "strong_reject" in tiers
        )
        if strict:
            raise RuntimeError(
                f"{category}: no strong/weak accepts; would require reject-tier fallback"
            )
        if weak_reject_pool:
            cap = max(min_winners, 1)
            if max_winners is not None:
                cap = min(cap, max_winners)
            winners[category] = weak_reject_pool[:cap]
            continue
        if strong_reject_pool:
            cap = max(min_winners, 1)
            if max_winners is not None:
                cap = min(cap, max_winners)
            winners[category] = strong_reject_pool[:cap]
            continue
        raise RuntimeError(f"{category}: no swipe votes")

    return winners


def preset_config_suggestions(
    winners: pd.DataFrame,
    catalog: SweepCatalog,
) -> str:
    if winners.empty or catalog._manifest.empty:
        return ""

    manifest = catalog._manifest
    lines = ["# Suggested categories.yaml overrides", ""]
    for _, row in winners.iterrows():
        variant_id = row["variant_id"]
        match = manifest[manifest["variant_id"] == variant_id].iloc[0]
        lines.append(f"  {row['category']}:")
        lines.append(f"    init_noise_level: {float(match['init_noise_level'])}")
        lines.append(f"    prompt_variant: {match['prompt_variant']}")
        if "steps" in match and pd.notna(match["steps"]):
            lines.append(f"    steps: {int(match['steps'])}")
        if "cfg_scale" in match and pd.notna(match["cfg_scale"]):
            lines.append(f"    cfg_scale: {float(match['cfg_scale'])}")
        lines.append("")
    lines.append("# After all phases: uv run python -m experiments.patch_sweep.lock")
    return "\n".join(lines)


def patch_config_suggestions(winners: pd.DataFrame, catalog: SweepCatalog) -> str:
    if winners.empty or catalog._manifest.empty:
        return ""

    manifest = catalog._manifest
    lines = ["# Suggested production config", ""]
    for _, row in winners.iterrows():
        variant_id = row["variant_id"]
        match = manifest[manifest["variant_id"] == variant_id].iloc[0]
        parts = [f"variant {variant_id}"]
        if "soundfont_id" in match and str(match["soundfont_id"]):
            parts.append(f"soundfont={match['soundfont_id']}")
        if "fx_profile" in match and str(match["fx_profile"]):
            parts.append(f"fx={match['fx_profile']}")
        if "pool_id" in match and str(match["pool_id"]):
            parts.append(f"pool={match['pool_id']}")
        lines.append(f"# {row['category']}: {', '.join(parts)}")
    lines.append("")
    lines.append("# After all phases: uv run python -m experiments.patch_sweep.lock")
    lines.append("# Or merge per-category winners into experiments/patch_sweep/winners.yaml")
    return "\n".join(lines)


def format_results_markdown(
    *,
    sweep_type: str,
    winners: pd.DataFrame,
    catalog: SweepCatalog,
    responses_path: Path,
) -> str:
    lines = [
        "# Sweep Listening Test Results",
        "",
        f"Source: `{responses_path}`",
        f"Sweep: {sweep_type}",
        "",
        "## Per instrument class",
        "",
    ]

    manifest = catalog._manifest

    if sweep_type == "preset":
        has_diffusion = "steps" in manifest.columns and "cfg_scale" in manifest.columns
        if has_diffusion:
            lines.append(
                "| Category | Winner variant_id | init_noise_level | prompt_variant "
                "| steps | cfg_scale | mean_content | mean_realism | Notes |"
            )
            lines.append(
                "|----------|-------------------|------------------|----------------|"
                "-------|-----------|--------------|--------------|-------|"
            )
        else:
            lines.append(
                "| Category | Winner variant_id | init_noise_level | prompt_variant "
                "| mean_content | mean_realism | Notes |"
            )
            lines.append(
                "|----------|-------------------|------------------|----------------|"
                "--------------|--------------|-------|"
            )
        for _, row in winners.iterrows():
            match = manifest[manifest["variant_id"] == row["variant_id"]].iloc[0]
            if has_diffusion:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{float(match['init_noise_level'])} | {match['prompt_variant']} | "
                    f"{int(match['steps'])} | {float(match['cfg_scale'])} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
            else:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{float(match['init_noise_level'])} | {match['prompt_variant']} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
    else:
        has_sf = "soundfont_id" in manifest.columns
        has_fx = "fx_profile" in manifest.columns
        if has_sf and has_fx:
            header = (
                "| Category | Winner variant_id | soundfont_id | fx_profile | pool_id "
                "| mean_content | mean_realism | Notes |"
            )
            sep = (
                "|----------|-------------------|--------------|------------|---------|"
                "--------------|--------------|-------|"
            )
        else:
            header = (
                "| Category | Winner variant_id | pool_id | mean_content | mean_realism | Notes |"
            )
            sep = (
                "|----------|-------------------|---------|--------------|--------------|-------|"
            )
        lines.append(header)
        lines.append(sep)
        for _, row in winners.iterrows():
            match = manifest[manifest["variant_id"] == row["variant_id"]].iloc[0]
            if has_sf and has_fx:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | "
                    f"{match.get('soundfont_id', '')} | {match.get('fx_profile', '')} | "
                    f"{match.get('pool_id', '')} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )
            else:
                lines.append(
                    f"| {row['category']} | {row['variant_id']} | {match['pool_id']} | "
                    f"{row['mean_content']} | {row['mean_realism']} | |"
                )

    lines.extend(["", "## Config suggestions", ""])
    if sweep_type == "preset":
        lines.append("```yaml")
        lines.append(preset_config_suggestions(winners, catalog).strip())
        lines.append("```")
    else:
        lines.append("```")
        lines.append(patch_config_suggestions(winners, catalog).strip())
        lines.append("```")

    return "\n".join(lines) + "\n"


def run_aggregate(
    *,
    sweep_type: str,
    responses_path: Path,
    output_path: Path | None,
    sweep_dir: Path | None,
    content_threshold: float,
    content_mean_threshold: float,
) -> pd.DataFrame:
    responses = load_responses(responses_path)
    df = ratings_dataframe(responses)
    _, winners = aggregate_winners(
        df,
        content_threshold=content_threshold,
        content_mean_threshold=content_mean_threshold,
    )

    catalog = SweepCatalog(
        sweep_type,
        sweep_dir or default_sweep_dir(sweep_type),
        probe_stems_path=DEFAULT_PROBE_STEMS,
    )

    markdown = format_results_markdown(
        sweep_type=sweep_type,
        winners=winners,
        catalog=catalog,
        responses_path=responses_path,
    )

    print(markdown)
    if output_path is not None:
        output_path.write_text(markdown)

    return winners


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Aggregate sweep listening-test responses.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        choices=["preset", "patch"],
        help="Sweep type to aggregate.",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Exported responses JSON from listening test.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write results markdown (e.g. experiments/patch_sweep/results_notes.md).",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        type=Path,
        help="Sweep output root (default: experiments/*/output symlink).",
    )
    parser.add_argument(
        "--content-threshold",
        default=DEFAULT_CONTENT_THRESHOLD,
        type=float,
        help="Drop variants with min content below this on any stem.",
    )
    parser.add_argument(
        "--content-mean-threshold",
        default=DEFAULT_CONTENT_MEAN_THRESHOLD,
        type=float,
        help="Drop variants with mean content below this per category.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    run_aggregate(
        sweep_type=args.sweep,
        responses_path=args.responses,
        output_path=args.output,
        sweep_dir=args.sweep_dir,
        content_threshold=args.content_threshold,
        content_mean_threshold=args.content_mean_threshold,
    )


if __name__ == "__main__":
    main()
