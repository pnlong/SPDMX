"""Shared configuration for the sPDMX pipeline."""

from pathlib import Path

from shared.env import load_project_env, path_from_env, repo_root

load_project_env()
_REPO_ROOT = repo_root()

# Machine-specific paths: copy .env.example → .env and edit (not this file).
PDMX_FILEPATH = path_from_env(
    "SPDMX_PDMX_FILEPATH",
    "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv",
)
OUTPUT_DIR = path_from_env(
    "SPDMX_OUTPUT_DIR",
    "/deepfreeze/pnlong/SPDMX",
)

# Slakh2100-redux (train/val/test FLAC stems) for MSS / SAO PoCs.
SLAKH_ROOT = path_from_env(
    "SPDMX_SLAKH_ROOT",
    "/deepfreeze/share/pnlong/slakh2100_flac_redux",
)

# Optional MUSDB18-HQ for cross-domain separation eval (Bass/Drums).
MUSDB_ROOT = path_from_env(
    "SPDMX_MUSDB_ROOT",
    "/deepfreeze/share/pnlong/musdb18hq",
)

# MedleyDB (V1/ + V2/ Artist_Track folders with MIX + STEMS).
MEDLEYDB_ROOT = path_from_env(
    "SPDMX_MEDLEYDB_ROOT",
    "/deepfreeze/share/pnlong/MedleyDB",
)

# MoisesDB (moisesdb_v0.1/<uuid>/ under this root; also accepts …/moisesdb/moisesdb_v0.1/).
MOISESDB_ROOT = path_from_env(
    "SPDMX_MOISESDB_ROOT",
    "/deepfreeze/share/pnlong/moisesdb",
)

# Local soundfont library (symlinked at repo root via shared.setup_symlinks).
SOUNDFONT_DIR = path_from_env(
    "SPDMX_SOUNDFONT_DIR",
    "/data3/pnlong/soundfonts",
)
_REPO_SOUNDFONTS_SYMLINK = _REPO_ROOT / "soundfonts"
if _REPO_SOUNDFONTS_SYMLINK.is_dir():
    SOUNDFONT_DIR = str(_REPO_SOUNDFONTS_SYMLINK)
SOUNDFONT_PATH = path_from_env(
    "SPDMX_SOUNDFONT_PATH",
    f"{SOUNDFONT_DIR}/SGM-V2.01.sf2",
)

CHUNK_SIZE = 1
NA_STRING = "NA"

DATA_DIR_NAME = "data"
STEMS_FILE_NAME = "stems"  # ablation / final pipeline tables (dev/…)
# Released + SPDMX_dev track map (stem-level). Was SPDMX.csv / track_map.csv.
SPDMX_FILE_NAME = "stems"
CAPTIONS_FILE_NAME = "captions"

# {OUTPUT_DIR}/dev/ — development artifacts (ablations, analysis, interim stems)
DEV_DIR_NAME = "dev"

# {OUTPUT_DIR}/dev/stems/ — full-scale stem synthesis (synthesize.py --full; ablations)
STEMS_DIR_NAME = "stems"

# {OUTPUT_DIR}/dev/ablations/{basic,slakh,ddsp_basic,ddsp_slakh}/ — listening test sample
ABLATIONS_DIR_NAME = "ablations"

# {OUTPUT_DIR}/dev/analysis/ — analysis outputs (song lengths, etc.)
ANALYSIS_DIR_NAME = "analysis"
SONG_LENGTHS_DIR_NAME = "song_lengths"
INSTRUMENTS_DIR_NAME = "instruments"
TRACK_NAMES_DIR_NAME = "track_names"

# {OUTPUT_DIR}/dev/mid_corrected/ — legacy dense MIDI tree (now {OUTPUT_DIR}/SPDMX_dev/mid/)
MID_CORRECTED_DIR_NAME = "mid_corrected"

# {OUTPUT_DIR}/dev/experiments/ — experiment outputs (patch sweep, etc.)
EXPERIMENTS_DIR_NAME = "experiments"
PATCH_SWEEP_DIR_NAME = "patch_sweep"

# {OUTPUT_DIR}/SPDMX_dev/ — flat production render (synthesis.final):
# Flat production render lives at {OUTPUT_DIR}/SPDMX_dev/ with LICENSE,
# README, stems.csv, raw/, audio/, mid/, mix/. Packaging never mutates this tree.
# {OUTPUT_DIR}/SPDMX/ — chunked distributable (build_spdmx): chunk_N/<song_id>/…
SPDMX_DEV_DIR_NAME = "SPDMX_dev"
SPDMX_DATASET_DIR_NAME = "SPDMX"
SPDMX_RAW_DIR_NAME = "raw"
SPDMX_AUDIO_DIR_NAME = "audio"
SPDMX_MID_DIR_NAME = "mid"
# Full-song mixes (ffmpeg stem sum); mirrors mid/ as mix/<song_id>.flac.
SPDMX_MIX_DIR_NAME = "mix"
# Filenames inside a flattened release song directory (chunk_N/<song_id>/).
SPDMX_RELEASE_MIX_AUDIO_NAME = "mix.flac"
SPDMX_RELEASE_MIX_MIDI_NAME = "mix.mid"

ABLATION_SUBSET_COLUMN = "subset:rated_deduplicated"
# sPDMX song-level subset (PDMX-style boolean on songs.csv): all four BDGP classes.
SPDMX_BDGP_SUBSET_COLUMN = "subset:bdgp"
# True when the song has ≥2 stems (False for single-track songs).
SPDMX_MULTITRACK_SUBSET_COLUMN = "subset:multitrack"
SPDMX_SONGS_FILE_NAME = "songs.csv"
# Safety cap for category-stratified ablation fill (not a fixed random N).
ABLATION_SAMPLE_SIZE = 400
ABLATION_SAMPLE_SEED = 43
ABLATION_MIN_STEMS_PER_CATEGORY = 50
LISTENING_SAMPLE_FILE_NAME = "listening_sample.yaml"

STEMS_TABLE_COLUMNS = [
    "path", "track", "original_track", "program", "is_drum", "name", "has_lyrics",
    "max_velocity", "velocity_scale",
    # True when the track has note_ons but every note is a same-tick on/off pair
    # (zero-duration *notes*). The track timeline can still be long; Fluidsynth
    # owns these — we never invent note lengths for MIDI-DDSP.
    "zero_duration_notes",
]

SONGS_TABLE_COLUMNS = [
    "path", "is_user_pro", "is_user_publisher", "is_user_staff",
    "has_paywall", "is_rated", "is_official", "is_original", "is_draft",
    "has_custom_audio", "has_custom_video", "n_comments", "n_favorites",
    "n_views", "n_ratings", "rating", "license", "license_url", "license_conflict",
    "genres", "groups", "tags", "song_name", "title", "subtitle", "artist_name",
    "composer_name", "publisher", "complexity", "n_tracks", "tracks",
    "song_length", "song_length.seconds", "song_length.bars", "song_length.beats",
    "n_notes", "notes_per_bar", "n_annotations", "has_annotations", "n_lyrics",
    "has_lyrics", "n_tokens", "pitch_class_entropy", "scale_consistency",
    "groove_consistency", "is_best_path", "is_best_arrangement",
    "is_best_unique_arrangement", "subset:all", "subset:rated",
    "subset:deduplicated", "subset:rated_deduplicated",
    "subset:no_license_conflict", "subset:valid_mxl_pdf",
]

CAPTION_MD_COLUMNS = [
    "genres", "groups", "tags", "song_name", "title", "subtitle",
    "artist_name", "composer_name", "publisher", "complexity", "license",
]

CAPTIONS_TABLE_COLUMNS = ["path", "track", "prompt"]

SAMPLE_RATE = 44100
# Channel layout for on-disk stems/mixtures and in-memory tensors (channels, samples).
# 1 = mono (downmix fluidsynth stereo). 2 = stereo (keep fluidsynth L/R).
STEM_CHANNELS = 1
GAIN = 1.0
STEM_FILE_PATTERN = "{track}.mp3"
MIXTURE_FILE_NAME = "mixture.mp3"
MIXTURE_PEAK_LIMIT = 1.0
FLAC_SUBTYPE = "PCM_16"  # on-disk stems/mixtures; processing uses float32 internally
DEFAULT_AUDIO_FORMAT = "mp3"
FLAC_AUDIO_FORMAT = "flac"
# Deprecated alias; use DEFAULT_AUDIO_FORMAT or FLAC_AUDIO_FORMAT.
PROTOTYPE_AUDIO_FORMAT = DEFAULT_AUDIO_FORMAT

TARGET_LOUDNESS_LUFS = -23.0

# Ablation listening viewer / make_clips: require ``{condition}_summable`` trees
# from ``synthesis.mix --no-overwrite`` (errors if any are missing; no raw fallback).
# Set False to audition raw stems.
LISTENING_PREFER_SUMMABLE = True

RENDER_MODE_BASIC = "basic"
RENDER_MODE_SLAKH = "slakh"
# Hybrid: MIDI-DDSP (mono URMP instruments) + DDSP-Piano (piano) + soundfont fallback.
RENDER_MODE_DDSP_BASIC = "ddsp_basic"
RENDER_MODE_DDSP_SLAKH = "ddsp_slakh"
# Deprecated alias kept for one release of import compatibility.
RENDER_MODE_SLAKH_DDSP = RENDER_MODE_DDSP_SLAKH
RENDER_MODES = (
    RENDER_MODE_BASIC,
    RENDER_MODE_SLAKH,
    RENDER_MODE_DDSP_BASIC,
    RENDER_MODE_DDSP_SLAKH,
)
# DDSP soundfont-fallback donors (raw ablation trees).
FALLBACK_DONOR = {
    RENDER_MODE_DDSP_BASIC: RENDER_MODE_BASIC,
    RENDER_MODE_DDSP_SLAKH: RENDER_MODE_SLAKH,
}

MAX_N_NOTES_IN_STEM = 50_000
MAX_STEM_DURATION = 30 * 60
MAX_N_SAMPLES_IN_STEM = int(MAX_STEM_DURATION * SAMPLE_RATE)
