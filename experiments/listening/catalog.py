"""Build JSON-serializable catalogs for patch sweep outputs.

Preset / SA3 catalog support was removed with ``experiments.preset_sweep``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments.paths import (
    DEFAULT_PROBE_STEMS,
    patch_sweep_output_root,
)
from experiments.patch_sweep.sweep import MANIFEST_FILENAME as PATCH_MANIFEST
from experiments.patch_sweep.sweep import VARIANTS_DIR_NAME as PATCH_VARIANTS_DIR
from experiments.listening_shared.clips import (
    PRESET_SWEEP_REMOVED,
    load_diverse_stems_manifest,
    resolve_sweep_clips_dir,
)
from shared.config import DATA_DIR_NAME, DEFAULT_AUDIO_FORMAT, FLAC_AUDIO_FORMAT, OUTPUT_DIR, STEMS_FILE_NAME
from shared.repo_symlinks import REPO_PATCH_SWEEP_OUTPUT_SYMLINK
from synthesis.audio import stem_filename
from synthesis.listening.catalog import default_ablations_dir, song_id_from_path
from synthesis.paths import ablation_raw_dir

CLIP_MANIFEST_CSV = "clip_manifest.csv"


def song_id_from_manifest_row(row) -> str:
    """Extract song id from manifest path or out_path."""
    try:
        return song_id_from_path(str(row["path"]))
    except ValueError:
        out_path = Path(str(row["out_path"]))
        parts = out_path.parts
        if DATA_DIR_NAME in parts:
            idx = parts.index(DATA_DIR_NAME)
            return "/".join(parts[idx + 1 : -1])
        raise


def default_source_dir(output_root: str = OUTPUT_DIR) -> Path:
    ablations = default_ablations_dir()
    basic = ablations / "basic"
    if basic.is_dir():
        return basic
    return Path(ablation_raw_dir(output_root, "basic"))


def default_sweep_dir(sweep_type: str, output_root: str = OUTPUT_DIR) -> Path:
    if sweep_type == "preset":
        raise RuntimeError(PRESET_SWEEP_REMOVED)
    if sweep_type == "patch":
        if REPO_PATCH_SWEEP_OUTPUT_SYMLINK.is_dir():
            return REPO_PATCH_SWEEP_OUTPUT_SYMLINK.resolve()
        return Path(patch_sweep_output_root(output_root))
    raise ValueError(f"Unknown sweep type: {sweep_type}")


def resolve_sweep_catalog_dir(
    sweep_type: str,
    sweep_dir: Path,
    *,
    prefer_verification_phase: bool = False,
) -> Path:
    """Pick a phased sweep output directory that actually has a manifest."""
    sweep_dir = sweep_dir.resolve()
    if sweep_type == "preset":
        raise RuntimeError(PRESET_SWEEP_REMOVED)
    manifest_name = PATCH_MANIFEST

    if prefer_verification_phase:
        try:
            from experiments.listening.final_verify import final_sweep_dir, readiness_errors

            if not readiness_errors(sweep_type):
                phase_dir = final_sweep_dir(sweep_type)
                if (phase_dir / manifest_name).is_file():
                    return phase_dir.resolve()
        except Exception:
            pass

    if (sweep_dir / manifest_name).is_file():
        return sweep_dir

    candidates = sorted(
        (
            path
            for path in sweep_dir.iterdir()
            if path.is_dir() and (path / manifest_name).is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return sweep_dir


class SweepCatalog:
    def __init__(
        self,
        sweep_type: str,
        sweep_dir: Path,
        source_dir: Path | None = None,
        probe_stems_path: Path = DEFAULT_PROBE_STEMS,
    ):
        if sweep_type == "preset":
            raise RuntimeError(PRESET_SWEEP_REMOVED)
        if sweep_type != "patch":
            raise ValueError(f"Unknown sweep type: {sweep_type}")
        self.sweep_type = sweep_type
        self.sweep_dir = sweep_dir.resolve()
        self.source_dir = (source_dir or default_source_dir()).resolve()
        clips_dir = resolve_sweep_clips_dir(self.sweep_dir)
        if clips_dir is not None:
            self.source_dir = clips_dir
        self.probe_stems_path = probe_stems_path
        self._manifest = self._load_manifest()
        self._clip_manifest = self._load_clip_manifest()
        self._probe_by_id = self._load_probe_index()
        self._audio_format = self._detect_audio_format()
        self._stem_groups, self._variant_out_paths = self._build_manifest_indexes(self._manifest)
        self._clip_stem_groups, self._clip_variant_out_paths = self._build_manifest_indexes(
            self._clip_manifest
        )

    def _build_manifest_indexes(
        self,
        manifest: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], Path]]:
        if manifest.empty:
            return {}, {}

        stem_groups = {
            str(stem_id): group
            for stem_id, group in manifest.groupby("stem_id", sort=False)
        }
        variant_out_paths: dict[tuple[str, str], Path] = {}
        for _, row in manifest.iterrows():
            variant_id = str(row["variant_id"])
            song_id = song_id_from_manifest_row(row)
            out_path = Path(str(row["out_path"]))
            filename = out_path.name
            variant_out_paths[(variant_id, song_id, filename)] = out_path
        return stem_groups, variant_out_paths

    @property
    def _variants_dir_name(self) -> str:
        return PATCH_VARIANTS_DIR

    @property
    def _manifest_filename(self) -> str:
        return PATCH_MANIFEST

    def manifest_id(self) -> str:
        path = self.sweep_dir / self._manifest_filename
        if not path.is_file():
            return "missing"
        stat = path.stat()
        return f"{stat.st_mtime_ns}_{stat.st_size}"

    def _load_manifest(self) -> pd.DataFrame:
        path = self.sweep_dir / self._manifest_filename
        if not path.is_file():
            return pd.DataFrame()
        df = pd.read_csv(path)
        return self._normalize_manifest(df)

    def _load_clip_manifest(self) -> pd.DataFrame:
        path = self.sweep_dir / CLIP_MANIFEST_CSV
        if not path.is_file():
            return pd.DataFrame()
        return self._normalize_manifest(pd.read_csv(path))

    def clip_manifest_id(self) -> str:
        path = self.sweep_dir / CLIP_MANIFEST_CSV
        if not path.is_file():
            return "missing"
        stat = path.stat()
        return f"{stat.st_mtime_ns}_{stat.st_size}"

    def swipe_available(self) -> bool:
        return not self._clip_manifest.empty

    def get_swipe_meta(
        self,
        *,
        category: str | None = None,
        order: str = "shuffle",
        seed: int = 42,
    ) -> dict:
        from experiments.listening.swipe import (
            build_swipe_cards,
            order_swipe_cards,
            swipe_session_id,
            swipe_storage_key,
        )

        cards = build_swipe_cards(self, category=category)
        cards = order_swipe_cards(cards, order=order, seed=seed)
        clip_manifest_id = self.clip_manifest_id()
        session_id = swipe_session_id(self.sweep_dir)
        categories = sorted({str(card["category"]) for card in cards if card.get("category")})
        return {
            "sweep_type": self.sweep_type,
            "mode": "swipe",
            "available": self.swipe_available(),
            "manifest_id": clip_manifest_id,
            "session_id": session_id,
            "storage_key": swipe_storage_key(self.sweep_type, session_id),
            "category": category,
            "categories": categories,
            "order": order,
            "seed": seed,
            "total_cards": len(cards),
            "cards": cards,
            "keyboard": {
                "strong_reject": "ArrowLeft",
                "strong_accept": "ArrowRight",
                "weak_reject": "ArrowDown",
                "weak_accept": "ArrowUp",
                "replay": "Space",
                "undo": "Backspace",
            },
        }

    def _normalize_manifest(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for col in (
            "phase",
            "variant_id",
            "soundfont_id",
            "soundfont_file",
            "fx_profile",
            "pool_id",
            "gm_class",
            "stem_id",
            "category",
            "path",
            "clip_id",
            "clip_index",
            "clip_start_seconds",
            "clip_seconds",
            "out_path",
            "prompt_variant",
            "prompt",
        ):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        if "program" in df.columns:
            df["program"] = pd.to_numeric(df["program"], errors="coerce")
        if "track" in df.columns:
            df["track"] = pd.to_numeric(df["track"], errors="coerce").astype("Int64")
        if "init_noise_level" in df.columns:
            df["init_noise_level"] = pd.to_numeric(df["init_noise_level"], errors="coerce")
        return df

    def _load_probe_index(self) -> dict[str, dict]:
        diverse_path = self.sweep_dir / "diverse_stems.yaml"
        if diverse_path.is_file():
            return {entry["id"]: entry for entry in load_diverse_stems_manifest(diverse_path)}
        with open(self.probe_stems_path) as f:
            cfg = yaml.safe_load(f)
        return {entry["id"]: entry for entry in cfg.get("stems", [])}

    def _detect_audio_format(self) -> str:
        if self._manifest.empty:
            return DEFAULT_AUDIO_FORMAT
        out_path = Path(self._manifest.iloc[0]["out_path"])
        suffix = out_path.suffix.lower().lstrip(".")
        if suffix in (DEFAULT_AUDIO_FORMAT, FLAC_AUDIO_FORMAT):
            return suffix
        return DEFAULT_AUDIO_FORMAT

    def available(self) -> bool:
        return (self.sweep_dir / self._manifest_filename).is_file()

    def variants(self) -> list[dict]:
        if self._manifest.empty:
            return []
        cols = ["variant_id"]
        for c in ("pool_id", "soundfont_id", "fx_profile", "phase"):
            if c in self._manifest.columns:
                cols.append(c)
        rows = (
            self._manifest[cols]
            .drop_duplicates(subset=["variant_id"])
            .sort_values("variant_id")
        )
        return rows.to_dict(orient="records")

    def list_stems(self) -> list[dict]:
        if self._manifest.empty:
            return []
        stems = []
        for stem_id, group in self._manifest.groupby("stem_id"):
            first = group.iloc[0]
            probe = self._probe_by_id.get(stem_id, {})
            song_id = song_id_from_path(first["path"])
            stems.append({
                "id": stem_id,
                "category": first.get("category") or probe.get("category"),
                "song_id": song_id,
                "track": int(first["track"]),
                "note": probe.get("note"),
            })
        stems.sort(key=lambda row: (row.get("category") or "", row["id"]))
        return stems

    def get_stem_test(
        self,
        stem_id: str,
        *,
        session_seed: int,
        blinded: bool = True,
    ) -> dict | None:
        if self._manifest.empty:
            return None
        group = self._stem_groups.get(stem_id)
        if group is None or group.empty:
            return None

        from experiments.listening.session import blinded_variant_order

        first = group.iloc[0]
        track = int(first["track"])
        song_id = song_id_from_path(first["path"])
        filename = self._reference_filename(stem_id, track)
        probe = self._probe_by_id.get(stem_id, {})

        variant_ids = group.sort_values("variant_id")["variant_id"].tolist()
        order = blinded_variant_order(
            variant_ids,
            stem_id=stem_id,
            session_seed=session_seed,
        )

        rows_by_variant = {str(row["variant_id"]): row for _, row in group.iterrows()}
        samples = []
        for blind_label, variant_id in order:
            row = rows_by_variant[str(variant_id)]
            sample = {
                "blind_label": blind_label,
                "audio": self._variant_cell_from_row(row),
            }
            if not blinded:
                sample["variant_id"] = variant_id
                if self.sweep_type == "preset":
                    sample["init_noise_level"] = float(row["init_noise_level"])
                    sample["prompt_variant"] = row["prompt_variant"]
                    sample["prompt"] = row.get("prompt")
                else:
                    if "pool_id" in row:
                        sample["pool_id"] = row["pool_id"]
                    if "soundfont_id" in row:
                        sample["soundfont_id"] = row["soundfont_id"]
                    if "fx_profile" in row:
                        sample["fx_profile"] = row["fx_profile"]
            else:
                sample["variant_id"] = variant_id
            samples.append(sample)

        return {
            "id": stem_id,
            "category": first.get("category") or probe.get("category"),
            "song_id": song_id,
            "track": track,
            "note": probe.get("note"),
            "filename": filename,
            "reference": self._reference_cell(stem_id, filename),
            "samples": samples,
        }

    def _reference_cell(self, stem_id: str, filename: str) -> dict:
        audio_path = self.resolve_reference_audio(stem_id, filename)
        available = audio_path is not None
        return {
            "available": available,
            "url": (
                f"/audio/{self.sweep_type}/reference/{stem_id}/{filename}"
                if available
                else None
            ),
        }

    def _variant_cell(self, variant_id: str, song_id: str, filename: str) -> dict:
        audio_path = self.resolve_variant_audio(variant_id, song_id, filename)
        available = audio_path is not None
        return {
            "available": available,
            "url": (
                f"/audio/{self.sweep_type}/variant/{variant_id}/{song_id}/{filename}"
                if available
                else None
            ),
        }

    def _variant_cell_from_row(self, row) -> dict:
        """Resolve variant audio from manifest out_path (robust to output-dir symlinks)."""
        variant_id = str(row["variant_id"])
        song_id = song_id_from_manifest_row(row)
        out_path = Path(str(row["out_path"]))
        filename = out_path.name
        return {
            "available": True,
            "url": f"/audio/{self.sweep_type}/variant/{variant_id}/{song_id}/{filename}",
        }

    def _reference_filename(self, stem_id: str, track: int) -> str:
        probe = self._probe_by_id.get(stem_id, {})
        audio_format = str(probe.get("audio_format") or self._audio_format)
        return stem_filename(track, audio_format)

    def _manifest_reference_path(self, stem_id: str) -> Path | None:
        if self._manifest.empty:
            return None
        group = self._manifest[self._manifest["stem_id"] == stem_id]
        if group.empty:
            return None

        row = group.iloc[0]
        track = int(row["track"])
        song_dir = Path(str(row["path"])).resolve()
        filename = self._reference_filename(stem_id, track)
        audio_path = (song_dir / filename).resolve()

        allowed_roots = {
            self.source_dir.resolve(),
            self.sweep_dir.resolve(),
        }
        clips_dir = resolve_sweep_clips_dir(self.sweep_dir)
        if clips_dir is not None:
            allowed_roots.add(clips_dir.resolve())

        if not any(str(audio_path).startswith(str(root)) for root in allowed_roots):
            return None
        return audio_path if audio_path.is_file() else None

    def resolve_reference_audio(self, stem_id: str, filename: str) -> Path | None:
        stem = self._probe_by_id.get(stem_id)
        if stem is None and self._manifest.empty:
            return None
        if "/" in filename or "\\" in filename or ".." in Path(filename).parts:
            return None

        manifest_path = self._manifest_reference_path(stem_id)
        if manifest_path is not None:
            return manifest_path

        if stem is None:
            return None
        song_id = stem["song_id"]
        audio_path = (self.source_dir / DATA_DIR_NAME / song_id / filename).resolve()
        if not str(audio_path).startswith(str(self.source_dir.resolve())):
            return None
        return audio_path if audio_path.is_file() else None

    def resolve_variant_audio(
        self,
        variant_id: str,
        song_id: str,
        filename: str,
    ) -> Path | None:
        if ".." in Path(song_id).parts or ".." in Path(filename).parts:
            return None
        if "/" in filename or "\\" in filename:
            return None

        indexed = self._variant_out_paths.get((variant_id, song_id, filename))
        if indexed is not None and indexed.is_file():
            return indexed.resolve()

        clip_indexed = self._clip_variant_out_paths.get((variant_id, song_id, filename))
        if clip_indexed is not None and clip_indexed.is_file():
            return clip_indexed.resolve()

        audio_path = (
            self.sweep_dir
            / self._variants_dir_name
            / variant_id
            / DATA_DIR_NAME
            / song_id
            / filename
        )
        if audio_path.is_file():
            return audio_path.resolve()
        return None

    def responses_dir(self) -> Path:
        path = self.sweep_dir / "responses"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_responses_path(self) -> Path:
        return self.responses_dir() / "responses_in_progress.json"

    def swipe_session_responses_path(self) -> Path:
        return self.responses_dir() / "swipe_in_progress.json"
