"""Map GM / Slakh instrument labels onto Bass, Drums, Guitar, Piano."""

from __future__ import annotations

from experiments.separation.paths import TARGETS

# GM program ranges (non-drum). Aligns with synthesis.patches._gm_class.
_GM_TARGET: dict[str, tuple[int, int]] = {
    "piano": (0, 8),  # [lo, hi)
    "guitar": (24, 32),
    "bass": (32, 40),
}

# Slakh metadata `inst_class` strings → target (case-insensitive substring ok).
_SLAKH_INST_CLASS: dict[str, str] = {
    "bass": "bass",
    "drums": "drums",
    "drum": "drums",
    "guitar": "guitar",
    "piano": "piano",
}


def gm_to_target(program: int, is_drum: bool) -> str | None:
    """Return bass/drums/guitar/piano or None if the stem is ignored."""
    if is_drum:
        return "drums"
    for target, (lo, hi) in _GM_TARGET.items():
        if lo <= int(program) < hi:
            return target
    return None


def slakh_inst_class_to_target(inst_class: str | None, *, is_drum: bool = False) -> str | None:
    """Map Slakh ``inst_class`` (or drum flag) onto a separation target."""
    if is_drum:
        return "drums"
    if not inst_class:
        return None
    key = inst_class.strip().lower()
    if key in _SLAKH_INST_CLASS:
        return _SLAKH_INST_CLASS[key]
    for needle, target in _SLAKH_INST_CLASS.items():
        if needle in key:
            return target
    return None


def required_targets() -> tuple[str, ...]:
    return TARGETS
