"""Route stems to MIDI-DDSP, DDSP-Piano, or slakh soundfont fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mido

from synthesis.patches import _gm_class, _normalize_name

BACKEND_MIDI_DDSP = "midi_ddsp"
BACKEND_DDSP_PIANO = "ddsp_piano"
BACKEND_SOUNDFONT = "soundfont"

REASON_PIANO = "piano"
REASON_MIDI_DDSP = "midi_ddsp_eligible"
REASON_POLYPHONIC = "soundfont_polyphonic"
REASON_UNSUPPORTED = "soundfont_unsupported"
REASON_DRUM = "soundfont_drum"
REASON_BASS_GUITAR = "soundfont_bass_guitar"
REASON_VOCAL = "soundfont_vocal"
REASON_GUITAR = "soundfont_guitar"
REASON_EMPTY = "empty_track"
# note_on + note_off at the same tick → PrettyMIDI drops the notes / end_time=0
REASON_ZERO_DURATION = "soundfont_zero_duration_notes"


def track_uses_drum_channel(track) -> bool:
    """True if any sounding note_on uses GM drum channel 9.

    Do not use the first channel-bearing message: meta/control on ch 0 often
    precedes channel-9 percussion notes, which mis-labeled drums as melodic and
    sent them to MIDI-DDSP (PrettyMIDI then raises ``Cannot synthesize drum``).
    """
    if track is None:
        return False
    for message in track:
        if (
            getattr(message, "type", None) == "note_on"
            and int(getattr(message, "velocity", 0) or 0) > 0
            and getattr(message, "channel", None) == 9
        ):
            return True
    return False


def midi_path_uses_drum_channel(midi_path: str | Path) -> bool:
    """True if any track in a MIDI file has sounding channel-9 notes."""
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    return any(track_uses_drum_channel(t) for t in midi.tracks)

# Canonical MIDI-DDSP / URMP names → GM program (0-indexed), matching magenta midi-ddsp.
MIDI_DDSP_NAME_TO_PROGRAM: dict[str, int] = {
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "double bass": 43,
    "flute": 73,
    "oboe": 68,
    "clarinet": 71,
    "saxophone": 66,
    "bassoon": 70,
    "trumpet": 56,
    "horn": 60,
    "trombone": 57,
    "tuba": 58,
}

MIDI_DDSP_PROGRAM_TO_NAME: dict[int, str] = {
    program: name for name, program in MIDI_DDSP_NAME_TO_PROGRAM.items()
}

# Track-name substrings → MIDI-DDSP instrument (checked before coarse GM class).
# Distinct neighbors (piccolo, pan flute, english horn, muted trumpet) are denied
# separately — do not fold them onto URMP models.
_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("contrabass", "double bass"),
    ("double bass", "double bass"),
    ("doublebass", "double bass"),
    ("violoncello", "cello"),
    ("cello", "cello"),
    ("viola", "viola"),
    ("violin", "violin"),
    ("fiddle", "violin"),
    ("flute", "flute"),
    ("oboe", "oboe"),
    ("clarinet", "clarinet"),
    ("bassoon", "bassoon"),
    ("saxophone", "saxophone"),
    ("sassofono", "saxophone"),  # Italian
    ("saxofon", "saxophone"),  # DE/SV/PL stem (saxofon/saxofón/…)
    ("szaxofon", "saxophone"),  # Hungarian
    ("saksafon", "saxophone"),  # TR/ID-ish spellings
    ("saksofon", "saxophone"),  # FI/NO/…
    ("alto sax", "saxophone"),
    ("tenor sax", "saxophone"),
    ("bari sax", "saxophone"),
    ("sax", "saxophone"),
    ("trumpet", "trumpet"),
    ("cornet", "trumpet"),
    ("trombone", "trombone"),
    ("tuba", "tuba"),
    ("french horn", "horn"),
    ("horn", "horn"),
)

# Name cues that must NOT map to a MIDI-DDSP instrument (timbre too different).
_MIDI_DDSP_NAME_DENY: tuple[str, ...] = (
    "piccolo",
    "english horn",
    "englishhorn",
    "cor anglais",
    "coranglais",
    "pan flute",
    "panflute",
    "recorder",
    "muted trumpet",
    "mute trumpet",
)

# GM muted trumpet — keep on soundfont even when the track is named "Trumpet".
MUTED_TRUMPET_PROGRAM = 59

# GM saxophone family (Soprano/Alto/Tenor/Bari) → one MIDI-DDSP sax model.
SAX_PROGRAMS = frozenset(range(64, 68))

# GM programs that look like "bass" but are bass guitar / synth bass — never double bass.
BASS_GUITAR_PROGRAMS = frozenset(range(32, 40))

# Full GM piano-class byte range (0–7). Not all go to DDSP-Piano.
PIANO_PROGRAMS = frozenset(range(0, 8))

# Acoustic hammer pianos only (MAESTRO). Exclude electric grand (2), e-pianos (4–5),
# harpsichord (6), clavinet (7).
DDSP_PIANO_PROGRAMS = frozenset({0, 1, 3})

# Name cues that are keyboard-family but not acoustic MAESTRO piano → soundfont.
_PIANO_DENY_NEEDLES: tuple[str, ...] = (
    "harpsichord",
    "cembalo",
    "clavinet",
    "clavi",
    "electric piano",
    "epiano",
    "e-piano",
    "e piano",
    "rhodes",
    "wurlitzer",
    "wurli",
    "dx7",
    "dx piano",
    "toy piano",
    "synth piano",
    "digital piano",
)

# SATB part names that imply choir/voice — skipped when an instrument keyword is present
# (e.g. "Alto Saxophone" must not count as vocal).
_VOCAL_PART_NEEDLES: tuple[str, ...] = (
    "soprano",
    "alto",
    "tenor",
    "baritone",
)
_VOCAL_ALWAYS_NEEDLES: tuple[str, ...] = (
    "voice",
    "vocal",
    "choir",
    "bass singer",
)
# If any of these appear, SATB part needles do not imply vocals.
_VOCAL_INSTRUMENT_EXCEPTIONS: tuple[str, ...] = (
    "sax",
    "saxophone",
    "sassofono",
    "saxofon",
    "szaxofon",
    "saksafon",
    "saksofon",
    "trumpet",
    "cornet",
    "trombone",
    "tuba",
    "clarinet",
    "oboe",
    "bassoon",
    "flute",
    "piccolo",
    "violin",
    "viola",
    "cello",
    "guitar",
    "piano",
    "organ",
    "harp",
    "french horn",
    "english horn",
    "horn",
    "recorder",
)

# Guitar / lute name cues. Use word boundaries so Dutch "dwarsfluit" (flute) is not
# matched by the "luit" substring.
_GUITAR_NAME_RE = re.compile(
    r"(guitar|gitar|gitarre|\blute\b|\bluit\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StemRoute:
    backend: str
    instrument_key: str | None
    reason: str


def _has_any(name: str, needles: tuple[str, ...]) -> bool:
    return any(needle in name for needle in needles)


def midi_ddsp_instrument_from_name(
    track_name: str | None,
    *,
    program: int | None = None,
) -> str | None:
    name = _normalize_name(track_name)
    if not name:
        return None
    if _has_any(name, _MIDI_DDSP_NAME_DENY):
        return None
    # Prefer longer / more specific aliases first (tuple order).
    for needle, instrument in _NAME_ALIASES:
        if needle not in name:
            continue
        # Muted trumpet (GM 59) stays on soundfont even if named "Trumpet".
        if instrument == "trumpet" and program is not None and int(program) == MUTED_TRUMPET_PROGRAM:
            return None
        return instrument
    return None


def midi_ddsp_instrument_from_program(program: int, is_drum: bool) -> str | None:
    if is_drum:
        return None
    if program in BASS_GUITAR_PROGRAMS:
        return None
    prog = int(program)
    if prog in SAX_PROGRAMS:
        return "saxophone"
    return MIDI_DDSP_PROGRAM_TO_NAME.get(prog)


def is_piano_stem(*, program: int, is_drum: bool, track_name: str | None) -> bool:
    """True only for acoustic hammer piano (DDSP-Piano / MAESTRO).

    Harpsichord, clavinet, and electric pianos are intentionally excluded even
    though GM places them in the 0–7 piano class.
    """
    if is_drum:
        return False
    name = _normalize_name(track_name)
    if _has_any(name, _PIANO_DENY_NEEDLES):
        return False
    prog = int(program)
    # Explicit non-acoustic GM piano-class programs stay on soundfont even when
    # the track is vaguely named "Piano".
    if prog in PIANO_PROGRAMS and prog not in DDSP_PIANO_PROGRAMS:
        return False
    if "piano" in name and "guitar" not in name:
        return True
    return prog in DDSP_PIANO_PROGRAMS and _gm_class(program, is_drum) == "piano"


def is_vocal_stem(*, program: int, is_drum: bool, track_name: str | None) -> bool:
    if is_drum:
        return False
    name = _normalize_name(track_name)
    prog = int(program)
    # GM saxophone family: never treat bare SATB names ("Alto", "Tenor") as choir.
    # Only explicit voice/choir cues win (rare mislabel).
    if prog in SAX_PROGRAMS:
        return _has_any(name, _VOCAL_ALWAYS_NEEDLES)
    if _has_any(name, _VOCAL_ALWAYS_NEEDLES):
        return True
    # SATB part names → vocal unless the track is clearly an instrument (Alto Sax, …).
    if _has_any(name, _VOCAL_PART_NEEDLES) and not _has_any(name, _VOCAL_INSTRUMENT_EXCEPTIONS):
        return True
    gm = _gm_class(program, is_drum)
    return gm == "ensemble" and prog in (52, 53, 54)  # Choir Aahs, Voice Oohs, Synth Voice


def is_monophonic_messages(messages, *, ticks_per_beat: int = 480) -> bool:
    """True if at most one note is sounding at any time (velocity>0 note_on overlaps)."""
    del ticks_per_beat  # absolute tick timeline from delta times is enough
    open_notes: set[int] = set()
    for message in messages:
        if message.type == "note_on" and message.velocity > 0:
            if open_notes:
                return False
            open_notes.add(message.note)
        elif message.type in ("note_off", "note_on") and (
            message.type == "note_off" or message.velocity == 0
        ):
            open_notes.discard(message.note)
    return True


def is_monophonic_midi(midi_path: str | Path) -> bool:
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    # Merge all tracks onto one timeline for the stem file (usually single-track).
    if len(midi.tracks) == 1:
        return is_monophonic_messages(midi.tracks[0], ticks_per_beat=midi.ticks_per_beat)
    merged = mido.merge_tracks(midi.tracks)
    return is_monophonic_messages(merged, ticks_per_beat=midi.ticks_per_beat)


def is_monophonic_track(track, *, ticks_per_beat: int = 480) -> bool:
    return is_monophonic_messages(track, ticks_per_beat=ticks_per_beat)


def _note_on_count(messages) -> int:
    return sum(
        1
        for message in messages
        if getattr(message, "type", None) == "note_on"
        and getattr(message, "velocity", 0) > 0
    )


def has_positive_duration_notes(messages) -> bool:
    """True if any note is closed at a later tick than its note_on.

    Some MIDI exports encode events as note_on + note_off (vel=0) at the *same*
    tick. Those are zero-duration *notes* (the track itself may still span many
    beats). PrettyMIDI drops them (``get_end_time() == 0``), so MIDI-DDSP would
    write empty audio — route to Fluidsynth instead; do not invent lengths.
    """
    open_ticks: dict[int, list[int]] = {}
    abs_tick = 0
    for message in messages:
        abs_tick += int(getattr(message, "time", 0) or 0)
        typ = getattr(message, "type", None)
        if typ == "note_on" and int(getattr(message, "velocity", 0) or 0) > 0:
            open_ticks.setdefault(int(message.note), []).append(abs_tick)
        elif typ == "note_off" or (
            typ == "note_on" and int(getattr(message, "velocity", 0) or 0) == 0
        ):
            stack = open_ticks.get(int(message.note))
            if not stack:
                continue
            start = stack.pop()
            if abs_tick > start:
                return True
    return False


def midi_path_has_positive_duration_notes(midi_path: str | Path) -> bool:
    """True if any track in ``midi_path`` has a positive-duration note."""
    midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
    return any(has_positive_duration_notes(track) for track in midi.tracks)


def route_stem(
    *,
    program: int,
    is_drum: bool,
    track_name: str | None = None,
    midi_path: str | Path | None = None,
    track=None,
    ticks_per_beat: int = 480,
    check_monophony: bool = True,
) -> StemRoute:
    """Decide neural vs soundfont backend for one stem.

    Monophony is only required for MIDI-DDSP. DDSP-Piano accepts polyphony.
    When ``check_monophony`` is True, pass ``midi_path`` and/or ``track``.
    """
    if is_drum or track_uses_drum_channel(track):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_DRUM)

    # Grand-staff / conductor stubs often have a name + program default 0 but no notes.
    # Sending those to DDSP-Piano yields a zero-length file that fails finalize.
    if track is not None and _note_on_count(track) == 0:
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_EMPTY)
    if track is not None and not has_positive_duration_notes(track):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_ZERO_DURATION)
    if track is None and midi_path is not None:
        midi = mido.MidiFile(filename=str(midi_path), charset="utf8")
        if sum(_note_on_count(t) for t in midi.tracks) == 0:
            return StemRoute(BACKEND_SOUNDFONT, None, REASON_EMPTY)
        if not any(has_positive_duration_notes(t) for t in midi.tracks):
            return StemRoute(BACKEND_SOUNDFONT, None, REASON_ZERO_DURATION)

    if is_vocal_stem(program=program, is_drum=is_drum, track_name=track_name):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_VOCAL)

    if is_piano_stem(program=program, is_drum=is_drum, track_name=track_name):
        return StemRoute(BACKEND_DDSP_PIANO, "piano", REASON_PIANO)

    name = _normalize_name(track_name)
    if name and _GUITAR_NAME_RE.search(name):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_GUITAR)
    if int(program) in BASS_GUITAR_PROGRAMS or (
        _gm_class(program, is_drum) == "bass" and int(program) in BASS_GUITAR_PROGRAMS
    ):
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_BASS_GUITAR)
    if _gm_class(program, is_drum) == "bass":
        # Acoustic/electric bass class without mapping to double bass.
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_BASS_GUITAR)
    if _gm_class(program, is_drum) == "guitar":
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_GUITAR)

    instrument = midi_ddsp_instrument_from_name(track_name, program=program)
    if instrument is None:
        instrument = midi_ddsp_instrument_from_program(program, is_drum)

    if instrument is None:
        return StemRoute(BACKEND_SOUNDFONT, None, REASON_UNSUPPORTED)

    if check_monophony:
        mono = True
        if track is not None:
            mono = is_monophonic_track(track, ticks_per_beat=ticks_per_beat)
        elif midi_path is not None:
            mono = is_monophonic_midi(midi_path)
        if not mono:
            return StemRoute(BACKEND_SOUNDFONT, instrument, REASON_POLYPHONIC)

    return StemRoute(BACKEND_MIDI_DDSP, instrument, REASON_MIDI_DDSP)
