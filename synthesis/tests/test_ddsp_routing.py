"""Unit tests for neural-DDSP stem routing and monophony checks."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from synthesis.ddsp.routing import (
    BACKEND_DDSP_PIANO,
    BACKEND_MIDI_DDSP,
    BACKEND_SOUNDFONT,
    DDSP_PIANO_PROGRAMS,
    REASON_BASS_GUITAR,
    REASON_DRUM,
    REASON_EMPTY,
    REASON_GUITAR,
    REASON_MIDI_DDSP,
    REASON_PIANO,
    REASON_POLYPHONIC,
    REASON_UNSUPPORTED,
    REASON_VOCAL,
    is_monophonic_messages,
    is_monophonic_midi,
    route_stem,
    track_uses_drum_channel,
)


def _track(*messages) -> mido.MidiTrack:
    track = mido.MidiTrack()
    for message in messages:
        track.append(message)
    return track


def test_route_piano_by_program():
    route = route_stem(program=0, is_drum=False, check_monophony=False)
    assert route.backend == BACKEND_DDSP_PIANO
    assert route.reason == REASON_PIANO


def test_empty_track_not_routed_to_piano():
    """Metadata-only grand-staff stubs default to program 0 but have no notes."""
    track = _track(
        mido.MetaMessage("track_name", name="Marimba (Grand Staff)", time=0),
        mido.MetaMessage("end_of_track", time=1),
    )
    route = route_stem(
        program=0,
        is_drum=False,
        track_name="Marimba (Grand Staff)",
        track=track,
        check_monophony=True,
    )
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_EMPTY


def test_route_piano_by_name():
    route = route_stem(program=48, is_drum=False, track_name="Grand Piano", check_monophony=False)
    assert route.backend == BACKEND_DDSP_PIANO


@pytest.mark.parametrize("program", sorted(DDSP_PIANO_PROGRAMS))
def test_ddsp_piano_allowlist_programs(program: int):
    route = route_stem(program=program, is_drum=False, check_monophony=False)
    assert route.backend == BACKEND_DDSP_PIANO


@pytest.mark.parametrize(
    "program,name",
    [
        (2, None),  # Electric Grand
        (4, None),  # Electric Piano 1
        (5, None),  # Electric Piano 2
        (6, None),  # Harpsichord
        (7, None),  # Clavinet
        (0, "Harpsichord"),
        (0, "Electric Piano"),
        (0, "Rhodes"),
        (6, "Harpsichord"),
        (7, "Clavinet"),
        (4, "E-Piano"),
        (4, "Piano"),  # GM e-piano wins over vague name
        (6, "Piano"),
    ],
)
def test_non_acoustic_piano_goes_soundfont(program: int, name: str | None):
    route = route_stem(program=program, is_drum=False, track_name=name, check_monophony=False)
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason != REASON_PIANO


def test_route_violin_mono():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_on", note=62, velocity=80, time=0),
        mido.Message("note_off", note=62, velocity=0, time=480),
    )
    route = route_stem(program=40, is_drum=False, track=track, check_monophony=True)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "violin"
    assert route.reason == REASON_MIDI_DDSP


def test_route_violin_polyphonic_falls_back():
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_on", note=64, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
        mido.Message("note_off", note=64, velocity=0, time=0),
    )
    route = route_stem(program=40, is_drum=False, track=track, check_monophony=True)
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_POLYPHONIC
    assert route.instrument_key == "violin"


def test_route_double_bass_not_bass_guitar():
    route = route_stem(program=43, is_drum=False, check_monophony=False)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "double bass"

    bass_guitar = route_stem(program=33, is_drum=False, check_monophony=False)
    assert bass_guitar.backend == BACKEND_SOUNDFONT
    assert bass_guitar.reason == REASON_BASS_GUITAR


def test_route_drums_and_vocals():
    assert route_stem(program=0, is_drum=True, check_monophony=False).reason == REASON_DRUM
    vocal = route_stem(program=52, is_drum=False, track_name="Choir", check_monophony=False)
    assert vocal.backend == BACKEND_SOUNDFONT
    assert vocal.reason == REASON_VOCAL


def test_drum_channel_after_non_drum_control_routes_to_soundfont():
    """First channel msg can be ch0; sounding notes on ch9 must still be drums."""
    track = mido.MidiTrack()
    track.append(mido.Message("control_change", channel=0, control=7, value=100, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("note_on", channel=9, note=36, velocity=80, time=0))
    track.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=100))
    assert track_uses_drum_channel(track)
    # is_drum=False mimics the old first-channel bug; track scan must still catch it.
    route = route_stem(
        program=0, is_drum=False, track_name="Kick", track=track, check_monophony=False,
    )
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_DRUM


def test_melodic_track_not_marked_drum():
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=100))
    assert not track_uses_drum_channel(track)


def test_alto_sax_not_vocal_goes_midi_ddsp():
    route = route_stem(
        program=65,
        is_drum=False,
        track_name="Alto Saxophone",
        check_monophony=False,
    )
    assert route.reason != REASON_VOCAL
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "saxophone"


@pytest.mark.parametrize(
    "name,program",
    [
        ("Alto", 65),
        ("Tenor", 66),
        ("Soprano", 64),
        ("Baritone", 67),
        ("Sassofono contralto", 65),
        ("Sassofono tenore", 66),
        ("Tenorszaxofon", 66),
        ("Alto Saksafon", 65),
    ],
)
def test_sax_program_or_multilingual_name_not_vocal(name: str, program: int):
    route = route_stem(program=program, is_drum=False, track_name=name, check_monophony=False)
    assert route.reason != REASON_VOCAL
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "saxophone"


def test_bare_alto_still_vocal_when_choir_program():
    route = route_stem(program=52, is_drum=False, track_name="Alto", check_monophony=False)
    assert route.reason == REASON_VOCAL


def test_all_gm_sax_programs_map_to_saxophone():
    for program in (64, 65, 66, 67):
        route = route_stem(program=program, is_drum=False, check_monophony=False)
        assert route.backend == BACKEND_MIDI_DDSP
        assert route.instrument_key == "saxophone"


def test_route_unsupported():
    route = route_stem(program=80, is_drum=False, check_monophony=False)  # synth lead
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason == REASON_UNSUPPORTED


@pytest.mark.parametrize(
    "name,program",
    [
        ("Piccolo", 72),
        ("Pan Flute", 75),
        ("English Horn", 69),
        ("Muted Trumpet", 59),
        ("Trumpet", 59),  # GM muted; name alone must not force open trumpet
    ],
)
def test_timbre_mismatches_not_midi_ddsp(name: str, program: int):
    route = route_stem(program=program, is_drum=False, track_name=name, check_monophony=False)
    assert route.backend == BACKEND_SOUNDFONT
    assert route.reason != REASON_MIDI_DDSP


def test_cornet_and_fiddle_still_map():
    cornet = route_stem(program=56, is_drum=False, track_name="Cornet", check_monophony=False)
    assert cornet.backend == BACKEND_MIDI_DDSP
    assert cornet.instrument_key == "trumpet"

    fiddle = route_stem(program=40, is_drum=False, track_name="Fiddle", check_monophony=False)
    assert fiddle.backend == BACKEND_MIDI_DDSP
    assert fiddle.instrument_key == "violin"


def test_open_trumpet_still_midi_ddsp():
    route = route_stem(program=56, is_drum=False, track_name="Trumpet", check_monophony=False)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "trumpet"


def test_french_horn_still_midi_ddsp():
    route = route_stem(program=60, is_drum=False, track_name="French Horn", check_monophony=False)
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "horn"


def test_dwarsfluit_not_guitar():
    """Dutch flute name contains 'luit' but must not hit the lute/guitar path."""
    route = route_stem(program=73, is_drum=False, track_name="Dwarsfluit", check_monophony=False)
    assert route.reason != REASON_GUITAR
    assert route.backend == BACKEND_MIDI_DDSP
    assert route.instrument_key == "flute"


def test_lute_still_guitar():
    route = route_stem(program=24, is_drum=False, track_name="Lute", check_monophony=False)
    assert route.reason == REASON_GUITAR


def test_is_monophonic_messages_and_file(tmp_path: Path):
    mono = _track(
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=100),
    )
    assert is_monophonic_messages(mono)

    poly = _track(
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_on", note=64, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=100),
        mido.Message("note_off", note=64, velocity=0, time=0),
    )
    assert not is_monophonic_messages(poly)

    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mono)
    path = tmp_path / "mono.mid"
    midi.save(str(path))
    assert is_monophonic_midi(path)


def test_synthesize_stem_neural_mocked(monkeypatch, tmp_path: Path):
    import torch

    from synthesis.ddsp.routing import StemRoute
    from synthesis.ddsp import synthesize as synth_mod

    midi = mido.MidiFile(ticks_per_beat=480)
    track = _track(
        mido.Message("program_change", program=40, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, velocity=0, time=480),
    )
    midi.tracks.append(track)
    mid_path = tmp_path / "v.mid"
    midi.save(str(mid_path))

    fake = torch.zeros(1, 1000)

    def fake_midi_ddsp(midi_path, instrument_name, **kwargs):
        assert instrument_name == "violin"
        return fake

    monkeypatch.setattr(synth_mod, "synthesize_stem_midi_ddsp", fake_midi_ddsp)
    out = synth_mod.synthesize_stem_neural(
        mid_path,
        StemRoute(BACKEND_MIDI_DDSP, "violin", REASON_MIDI_DDSP),
    )
    assert out is fake
