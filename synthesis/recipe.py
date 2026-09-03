"""Per-category synthesis recipe for hybrid (final) rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from shared.config import DATA_DIR_NAME
from synthesis.audio import stem_is_valid, stem_path
from synthesis.ddsp.routing import (
    BACKEND_DDSP_PIANO,
    BACKEND_MIDI_DDSP,
    BACKEND_SOUNDFONT,
    StemRoute,
    route_stem,
)
from synthesis.listening.catalog import CONDITION_ORDER
from synthesis.patches import LISTENING_CATEGORY_GM_CLASSES, resolve_probe_category

DEFAULT_RECIPE_PATH = Path(__file__).resolve().parent / "recipe.yaml"

STEM_RECIPE_FILE_NAME = "stem_recipe.csv"
STEM_RECIPE_COLUMNS = [
    "path",
    "track",
    "category",
    "ablation",
    "method",
    "fallback",
    "backend",
    "realify",
    # Why this backend was chosen (e.g. midi_ddsp_eligible, soundfont_polyphonic).
    # Empty/NA on older rows written before this column existed.
    "reason",
]

METHOD_BASIC = "basic"
METHOD_SLAKH = "slakh"
METHOD_MIDI_DDSP = "midi-ddsp"
METHODS = (METHOD_BASIC, METHOD_SLAKH, METHOD_MIDI_DDSP)

FALLBACK_BASIC = "basic"
FALLBACK_SLAKH = "slakh"
FALLBACKS = (FALLBACK_BASIC, FALLBACK_SLAKH)

BACKEND_FLUIDSYNTH = "fluidsynth"

_ABLATION_IDS = frozenset(CONDITION_ORDER)
_METHOD_ALIASES = {
    "basic": METHOD_BASIC,
    "slakh": METHOD_SLAKH,
    "midi-ddsp": METHOD_MIDI_DDSP,
    "midi_ddsp": METHOD_MIDI_DDSP,
}


@dataclass(frozen=True)
class CategorySpec:
    """Resolved synthesis choice for one listening category."""

    method: str
    realify: bool
    fallback: str
    ablation: str | None = None


@dataclass(frozen=True)
class TrackPlan:
    """Per-track hybrid plan (category recipe + routing hints)."""

    category: str
    method: str
    realify: bool
    use_slakh: bool
    neural_ok: bool
    fallback: str
    ablation: str | None = None

    def sidecar_row(
        self,
        *,
        path: str,
        track: int,
        backend: str,
        realify: bool | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        applied = self.realify if realify is None else bool(realify)
        return {
            "path": path,
            "track": int(track),
            "category": self.category,
            "ablation": self.ablation,
            "method": self.method,
            "fallback": self.fallback,
            "backend": backend,
            "realify": applied,
            "reason": reason,
        }


@dataclass(frozen=True)
class RecipeConflict:
    path: str
    track: int
    category: str | None
    recorded: str
    desired: str


@dataclass(frozen=True)
class CategoryRecipe:
    """Category → spec mapping loaded from YAML."""

    specs: dict[str, CategorySpec]
    path: Path | None = None

    def spec_for_category(self, category: str) -> CategorySpec:
        try:
            return self.specs[category]
        except KeyError as exc:
            raise KeyError(f"No recipe spec for listening category {category!r}") from exc

    def plan_for_track(
        self,
        *,
        program: int,
        is_drum: bool,
        track_name: str | None = None,
    ) -> TrackPlan:
        category = resolve_probe_category(
            program=int(program),
            is_drum=bool(is_drum),
            track_name=track_name,
        )
        spec = self.spec_for_category(category)
        use_slakh = spec.method == METHOD_SLAKH or spec.fallback == FALLBACK_SLAKH
        return TrackPlan(
            category=category,
            method=spec.method,
            realify=bool(spec.realify),
            use_slakh=use_slakh,
            neural_ok=spec.method == METHOD_MIDI_DDSP,
            fallback=spec.fallback,
            ablation=spec.ablation,
        )

    def uses_ddsp(self) -> bool:
        return any(spec.method == METHOD_MIDI_DDSP for spec in self.specs.values())

    def uses_ddsp_piano(self) -> bool:
        """True when the piano listening category is MIDI-DDSP (DDSP-Piano engine)."""
        spec = self.specs.get("piano")
        return spec is not None and spec.method == METHOD_MIDI_DDSP

    def uses_realify(self) -> bool:
        return any(spec.realify for spec in self.specs.values())

    def uses_slakh(self) -> bool:
        return any(
            spec.method == METHOD_SLAKH or spec.fallback == FALLBACK_SLAKH
            for spec in self.specs.values()
        )

    def pass_categories(self) -> dict[str, tuple[str, ...]]:
        """Category grouping for method passes (primary method, not fallbacks)."""
        fluidsynth = tuple(
            c for c, spec in self.specs.items() if spec.method in (METHOD_BASIC, METHOD_SLAKH)
        )
        ddsp = tuple(c for c, spec in self.specs.items() if spec.method == METHOD_MIDI_DDSP)
        realify = tuple(c for c, spec in self.specs.items() if spec.realify)
        return {"fluidsynth": fluidsynth, "ddsp": ddsp, "realify": realify}

    def realify_categories(self) -> frozenset[str]:
        return frozenset(c for c, spec in self.specs.items() if spec.realify)


def hybrid_pass_for_track(
    recipe: CategoryRecipe,
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
) -> str:
    """Which hybrid pass owns this track (metadata only; no MIDI monophony check)."""
    plan = recipe.plan_for_track(
        program=int(program),
        is_drum=bool(is_drum),
        track_name=track_name,
    )
    if not plan.neural_ok:
        return "fluidsynth"
    route = route_stem(
        program=int(program),
        is_drum=bool(is_drum),
        track_name=track_name,
        check_monophony=False,
    )
    if route.backend == BACKEND_DDSP_PIANO:
        if recipe.uses_ddsp_piano():
            return "ddsp_piano"
        return "fluidsynth"
    if route.backend == BACKEND_MIDI_DDSP:
        return "midi_ddsp"
    return "fluidsynth"


def parse_ablation_id(ablation: str) -> CategorySpec:
    """Expand a listening-test condition id into method / realify / fallback."""
    name = str(ablation).strip()
    if name not in _ABLATION_IDS:
        raise ValueError(
            f"Unknown ablation id {name!r}. Expected one of: {', '.join(CONDITION_ORDER)}"
        )
    realify = name.endswith("_realify")
    raw = name[: -len("_realify")] if realify else name
    if raw == "basic":
        return CategorySpec(METHOD_BASIC, realify, FALLBACK_BASIC, name)
    if raw == "slakh":
        return CategorySpec(METHOD_SLAKH, realify, FALLBACK_SLAKH, name)
    if raw == "ddsp_basic":
        return CategorySpec(METHOD_MIDI_DDSP, realify, FALLBACK_BASIC, name)
    if raw == "ddsp_slakh":
        return CategorySpec(METHOD_MIDI_DDSP, realify, FALLBACK_SLAKH, name)
    raise ValueError(f"Cannot expand ablation id {name!r}")


def parse_category_spec(value: Any, *, category: str) -> CategorySpec:
    """Parse an ablation id string or expanded ``{method, realify, fallback}`` mapping."""
    if isinstance(value, str):
        return parse_ablation_id(value)
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Recipe for {category!r} must be an ablation id or mapping, got {type(value).__name__}"
        )
    raw_method = value.get("method")
    if raw_method is None:
        raise ValueError(f"Recipe for {category!r} is missing 'method'")
    method = _METHOD_ALIASES.get(str(raw_method).strip().lower())
    if method is None:
        raise ValueError(
            f"Recipe for {category!r} has unknown method {raw_method!r}. "
            f"Expected one of: {', '.join(METHODS)}"
        )
    realify = bool(value.get("realify", False))
    raw_fallback = value.get("fallback")
    if raw_fallback is None:
        fallback = FALLBACK_SLAKH if method == METHOD_SLAKH else FALLBACK_BASIC
    else:
        fallback = str(raw_fallback).strip().lower()
        if fallback not in FALLBACKS:
            raise ValueError(
                f"Recipe for {category!r} has unknown fallback {raw_fallback!r}. "
                f"Expected one of: {', '.join(FALLBACKS)}"
            )
    if method == METHOD_SLAKH:
        fallback = FALLBACK_SLAKH
    elif method == METHOD_BASIC:
        fallback = FALLBACK_BASIC
    ablation = value.get("ablation")
    return CategorySpec(method, realify, fallback, str(ablation) if ablation else None)


def _categories_mapping(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        raise TypeError(f"Recipe YAML must be a mapping, got {type(doc).__name__}")
    if "categories" in doc:
        inner = doc["categories"]
        if not isinstance(inner, Mapping):
            raise TypeError("'categories' must be a mapping of category → recipe")
        return dict(inner)
    return {k: v for k, v in doc.items() if not str(k).startswith("_")}


def load_recipe(source: str | Path | Mapping[str, Any] | None = None) -> CategoryRecipe:
    """Load and validate a per-category recipe from a YAML path or in-memory mapping."""
    path: Path | None = None
    if source is None:
        source = DEFAULT_RECIPE_PATH
    if isinstance(source, Mapping):
        raw = _categories_mapping(source)
    else:
        path = Path(source)
        with path.open() as f:
            loaded = yaml.safe_load(f)
        raw = _categories_mapping(loaded)

    required = tuple(LISTENING_CATEGORY_GM_CLASSES.keys())
    missing = [c for c in required if c not in raw]
    extra = sorted(c for c in raw if c not in LISTENING_CATEGORY_GM_CLASSES)
    if missing:
        raise ValueError(
            "Recipe YAML is missing listening categories: " + ", ".join(missing)
        )
    if extra:
        raise ValueError("Recipe YAML has unknown categories: " + ", ".join(extra))

    specs = {c: parse_category_spec(raw[c], category=c) for c in required}
    return CategoryRecipe(specs=specs, path=path)


def resolve_track_backend(plan: TrackPlan, route: StemRoute | None) -> str:
    """Return ``fluidsynth``, ``ddsp_piano``, or ``midi_ddsp`` for this track."""
    if plan.neural_ok and route is not None:
        if route.backend in (BACKEND_MIDI_DDSP, BACKEND_DDSP_PIANO):
            return route.backend
    return BACKEND_FLUIDSYNTH


def route_for_plan(
    plan: TrackPlan,
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
    track=None,
    ticks_per_beat: int = 480,
    check_monophony: bool = True,
) -> StemRoute | None:
    """Run DDSP routing only when the category recipe is neural."""
    if not plan.neural_ok:
        return StemRoute(BACKEND_SOUNDFONT, None, "category_soundfont")
    return route_stem(
        program=program,
        is_drum=is_drum,
        track_name=track_name,
        track=track,
        ticks_per_beat=ticks_per_beat,
        check_monophony=check_monophony,
    )


def listening_category_from_stem_row(row: Mapping[str, Any]) -> str:
    """Map stem metadata to a listening category (same as synthesis)."""
    program = row.get("program")
    try:
        program_i = int(program) if program is not None and str(program) != "nan" else 0
    except (TypeError, ValueError):
        program_i = 0
    is_drum = row.get("is_drum")
    if isinstance(is_drum, str):
        is_drum_b = is_drum.strip().lower() in {"1", "true", "yes"}
    else:
        is_drum_b = bool(is_drum)
    name = row.get("name")
    if name is not None and str(name) in {"", "nan", "None"}:
        name = None
    return resolve_probe_category(program=program_i, is_drum=is_drum_b, track_name=name)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def raw_fingerprint(method: str, fallback: str, backend: str) -> tuple[str, str, str]:
    return (str(method), str(fallback), str(backend))


def realify_fingerprint(
    method: str, fallback: str, backend: str, realify: bool,
) -> tuple[str, str, str, bool]:
    return (str(method), str(fallback), str(backend), bool(realify))


def recorded_raw_fingerprint(row: Mapping[str, Any] | None) -> tuple[str, str, str] | None:
    if row is None:
        return None
    return raw_fingerprint(row["method"], row["fallback"], row["backend"])


def recorded_realify_fingerprint(
    row: Mapping[str, Any] | None,
) -> tuple[str, str, str, bool] | None:
    if row is None:
        return None
    return realify_fingerprint(
        row["method"], row["fallback"], row["backend"], _as_bool(row.get("realify")),
    )


def format_raw_fingerprint(fp: tuple[str, str, str] | None) -> str:
    if fp is None:
        return "(none)"
    method, fallback, backend = fp
    return f"method={method} fallback={fallback} backend={backend}"


def format_realify_fingerprint(fp: tuple[str, str, str, bool] | None) -> str:
    if fp is None:
        return "(none)"
    method, fallback, backend, realify = fp
    return (
        f"method={method} fallback={fallback} backend={backend} realify={str(realify).lower()}"
    )


def load_stem_recipe_index(
    stems_dir: str | Path,
    filename: str = STEM_RECIPE_FILE_NAME,
) -> dict[tuple[str, int], dict]:
    """Map (song path, track) → sidecar row (last row wins)."""
    from shared.csv_tables import read_csv_flexible

    path = Path(stems_dir) / filename
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        df = read_csv_flexible(path, columns=STEM_RECIPE_COLUMNS)
    except OSError:
        return {}
    if df.empty or "path" not in df.columns or "track" not in df.columns:
        return {}
    index: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        index[(str(row["path"]), int(row["track"]))] = row.to_dict()
    return index


def desired_raw_fingerprint(spec: CategorySpec, backend: str) -> tuple[str, str, str]:
    return raw_fingerprint(spec.method, spec.fallback, backend)


def desired_realify_fingerprint(
    spec: CategorySpec, backend: str, *, applied: bool | None = None,
) -> tuple[str, str, str, bool]:
    flag = spec.realify if applied is None else bool(applied)
    return realify_fingerprint(spec.method, spec.fallback, backend, flag)


def scan_recipe_conflicts(
    stems_dir: str | Path,
    recipe: CategoryRecipe,
    *,
    audio_format: str,
    stage: str,
) -> list[RecipeConflict]:
    """Find on-disk stems whose sidecar does not match the current recipe.

    ``stage`` is ``raw`` (ignore realify flag) or ``realify`` (include it).
    Stems with no sidecar after a completed ``data.csv`` are treated as conflicts.
    Incomplete runs (stem files but no ``data.csv``) resume without prompting.
    """
    root = Path(stems_dir)
    data_csv = root / f"{DATA_DIR_NAME}.csv"
    if not data_csv.is_file():
        return []
    songs = pd.read_csv(data_csv)
    if songs.empty or "path" not in songs.columns:
        return []
    index = load_stem_recipe_index(root)
    stems_csv = root / "stems.csv"
    stems_by_key: dict[tuple[str, int], dict] = {}
    if stems_csv.is_file():
        stems = pd.read_csv(stems_csv)
        for _, row in stems.iterrows():
            stems_by_key[(str(row["path"]), int(row["track"]))] = row.to_dict()

    conflicts: list[RecipeConflict] = []
    for _, song in songs.iterrows():
        song_path = str(song["path"])
        n_tracks = int(song["n_tracks"])
        for track in range(n_tracks):
            if not stem_is_valid(stem_path(Path(song_path), track, audio_format)):
                continue
            rec = index.get((song_path, track))
            stem_meta = stems_by_key.get((song_path, track), rec or {})
            try:
                category = str(rec["category"]) if rec and rec.get("category") else (
                    listening_category_from_stem_row(stem_meta)
                    if stem_meta else None
                )
            except Exception:
                category = None
            if category is None or category not in recipe.specs:
                conflicts.append(RecipeConflict(
                    path=song_path,
                    track=track,
                    category=category,
                    recorded=format_raw_fingerprint(recorded_raw_fingerprint(rec)),
                    desired="(unknown category)",
                ))
                continue
            spec = recipe.spec_for_category(category)
            backend = str(rec["backend"]) if rec and rec.get("backend") else BACKEND_FLUIDSYNTH
            if stage == "realify":
                recorded = recorded_realify_fingerprint(rec)
                desired = desired_realify_fingerprint(spec, backend)
                if recorded != desired:
                    conflicts.append(RecipeConflict(
                        path=song_path,
                        track=track,
                        category=category,
                        recorded=format_realify_fingerprint(recorded),
                        desired=format_realify_fingerprint(desired),
                    ))
            else:
                recorded = recorded_raw_fingerprint(rec)
                desired = desired_raw_fingerprint(spec, backend)
                if recorded != desired:
                    conflicts.append(RecipeConflict(
                        path=song_path,
                        track=track,
                        category=category,
                        recorded=format_raw_fingerprint(recorded),
                        desired=format_raw_fingerprint(desired),
                    ))
    return conflicts


def confirm_recipe_conflicts(
    conflicts: list[RecipeConflict],
    *,
    yes: bool = False,
    input_fn=input,
) -> bool:
    """Ask whether to regenerate mismatched stems. Return True to proceed.

    With ``yes=True`` (``-y``), proceed without prompting. Empty ``conflicts``
    proceeds. A no / EOF aborts.
    """
    if not conflicts:
        return True
    n = len(conflicts)
    print(f"{n} existing stem(s) do not match the current category recipe:")
    preview = conflicts[:20]
    for item in preview:
        loc = f"{item.path} track={item.track}"
        cat = item.category or "?"
        print(f"  [{cat}] {loc}")
        print(f"    recorded: {item.recorded}")
        print(f"    desired:  {item.desired}")
    if n > len(preview):
        print(f"  … and {n - len(preview)} more")
    if yes:
        print("Proceeding with regeneration (-y).")
        return True
    try:
        reply = input_fn(
            "Regenerate these stems to match the recipe? [y/N] "
        ).strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def require_recipe_conflicts_ok(
    conflicts: list[RecipeConflict],
    *,
    yes: bool,
    input_fn=input,
) -> None:
    if confirm_recipe_conflicts(conflicts, yes=yes, input_fn=input_fn):
        return
    raise SystemExit(
        "Aborted: existing stems do not match the current recipe. "
        "Re-run with -y to regenerate them, or --reset to start over."
    )


def sync_realify_sidecar(
    source_dir: str | Path,
    dest_dir: str | Path,
    recipe: CategoryRecipe,
) -> None:
    """Rewrite dest ``stem_recipe.csv`` from the current recipe and raw sidecar backends."""
    from synthesis.paths import remap_path_prefix

    dest = Path(dest_dir)
    source = Path(source_dir)
    stems_csv = dest / "stems.csv"
    if not stems_csv.is_file():
        return
    stems = pd.read_csv(stems_csv)
    src_index = load_stem_recipe_index(source)
    rows: list[dict[str, Any]] = []
    for _, row in stems.iterrows():
        dest_path = str(row["path"])
        track = int(row["track"])
        src_path = remap_path_prefix(dest_path, dest, source)
        rec = src_index.get((src_path, track))
        if rec and rec.get("category"):
            category = str(rec["category"])
        else:
            category = listening_category_from_stem_row(row.to_dict())
        spec = recipe.spec_for_category(category)
        backend = str(rec["backend"]) if rec and rec.get("backend") else BACKEND_FLUIDSYNTH
        rows.append({
            "path": dest_path,
            "track": track,
            "category": category,
            "ablation": spec.ablation,
            "method": spec.method,
            "fallback": spec.fallback,
            "backend": backend,
            "realify": bool(spec.realify),
        })
    out = dest / STEM_RECIPE_FILE_NAME
    pd.DataFrame(rows, columns=STEM_RECIPE_COLUMNS).to_csv(out, index=False)
