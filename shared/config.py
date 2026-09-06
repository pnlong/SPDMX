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
STEMS_FILE_NAME = "stems"
SPDMX_FILE_NAME = "SPDMX"  # released table {SPDMX}/SPDMX.csv (was track_map.csv)
CAPTIONS_FILE_NAME = "captions"

# {OUTPUT_DIR}/dev/ — development artifacts (ablations, analysis, interim stems)
DEV_DIR_NAME = "dev"

# {OUTPUT_DIR}/dev/stems/ — full-scale stem synthesis (synthesize.py --full; ablations)
STEMS_DIR_NAME = "stems"
STEMS_REALIFY_DIR_NAME = "stems_realify"

# {OUTPUT_DIR}/dev/ablations/{basic,basic_realify,slakh,slakh_realify}/ — listening test sample
ABLATIONS_DIR_NAME = "ablations"

# {OUTPUT_DIR}/dev/analysis/ — analysis outputs (song lengths, etc.)
ANALYSIS_DIR_NAME = "analysis"
SONG_LENGTHS_DIR_NAME = "song_lengths"
INSTRUMENTS_DIR_NAME = "instruments"
TRACK_NAMES_DIR_NAME = "track_names"

# {OUTPUT_DIR}/dev/mid_corrected/ — legacy dense MIDI tree (now {OUTPUT_DIR}/SPDMX/mid/)
MID_CORRECTED_DIR_NAME = "mid_corrected"

# {OUTPUT_DIR}/dev/experiments/ — experiment outputs (preset sweep, etc.)
EXPERIMENTS_DIR_NAME = "experiments"
PRESET_SWEEP_DIR_NAME = "preset_sweep"
PATCH_SWEEP_DIR_NAME = "patch_sweep"

# {OUTPUT_DIR}/SPDMX/ — released dataset: LICENSE, README, SPDMX.csv, audio/, mid/
# Raw hybrid stems live under raw/; mix writes the released summable tree to audio/.
SPDMX_DATASET_DIR_NAME = "SPDMX"
SPDMX_RAW_DIR_NAME = "raw"
SPDMX_AUDIO_DIR_NAME = "audio"
SPDMX_MID_DIR_NAME = "mid"

ABLATION_SUBSET_COLUMN = "subset:rated_deduplicated"
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
# DDSP soundfont-fallback donors (raw / realify).
FALLBACK_DONOR = {
    RENDER_MODE_DDSP_BASIC: RENDER_MODE_BASIC,
    RENDER_MODE_DDSP_SLAKH: RENDER_MODE_SLAKH,
}

MAX_N_NOTES_IN_STEM = 50_000
MAX_STEM_DURATION = 30 * 60
MAX_N_SAMPLES_IN_STEM = int(MAX_STEM_DURATION * SAMPLE_RATE)

SA3_SMALL_MUSIC_MAX_DURATION = 120
SA3_MEDIUM_MAX_DURATION = 380

# Stable Audio 3 realify defaults (audio-to-audio timbre transfer)
REALIFY_INIT_NOISE_LEVEL = 0.45
REALIFY_STEPS = 8
REALIFY_CFG_SCALE = 1.0
REALIFY_DURATION_PADDING_SEC = 6.0  # SA3 model.generate default
REALIFY_CHUNK_OVERLAP_SEC = 2.0
# Skip GPUs with less free VRAM at realify startup (medium needs ~8–10 GiB per worker).
REALIFY_MIN_GPU_FREE_GB = 10.0
REALIFY_CHUNKED_DECODE = True
# SA3 stems per GPU forward pass. 0 = auto from free/total VRAM per GPU
# (see suggest_realify_batch_size). Positive ints override (--realify-batch-size N).
REALIFY_BATCH_SIZE = 0

# Post-SA3 silence enforcement (reference vs realified energy comparison)
REALIFY_SILENCE_ENFORCE = True
REALIFY_SILENCE_CHUNK_MS = 1000.0
REALIFY_SILENCE_OVERLAP_RATIO = 0.5
REALIFY_SILENCE_THRESHOLD_DB = -60.0
REALIFY_SILENCE_ACTIVE_MARGIN_MS = 1000.0
REALIFY_SILENCE_FADE_MS = 200.0

# Post-SA3 content fidelity gate (reference vs realified onset alignment)
REALIFY_CONTENT_FIDELITY_ENFORCE = False
REALIFY_CONTENT_FIDELITY_THRESHOLD = 0.85
REALIFY_CONTENT_FIDELITY_NOISE_STEP = 0.10
REALIFY_CONTENT_FIDELITY_MIN_NOISE = 0.25
REALIFY_CONTENT_FIDELITY_MAX_ATTEMPTS = 4
REALIFY_CONTENT_FIDELITY_ONSET_TOLERANCE_MS = 50.0
REALIFY_CONTENT_FIDELITY_ACTIVE_MARGIN_MS = 100.0

# Realify inference backend: "pytorch" (default) or "trt" (TensorRT eager audio-to-audio)
REALIFY_BACKEND = "pytorch"
