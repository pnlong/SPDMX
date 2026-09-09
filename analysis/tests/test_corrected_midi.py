"""Tests for dense corrected MIDI writing."""

from __future__ import annotations

from pathlib import Path

import mido
import pandas as pd

from analysis.corrected_midi import (
    TRACK_MAP_COLUMNS,
    load_track_map,
    note_on_count,
    resolve_corrected_midi_path,
    track_map_csv_path,
    write_corrected_midi,
    write_corrected_midis_from_register,
)
from synthesis.dense_midi import stem_original_track


def _song_with_empty_stub(tmp_path: Path) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    t0.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    t0.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(t0)
    t1 = mido.MidiTrack()
    t1.append(mido.MetaMessage("track_name", name="Marimba (Grand Staff)", time=0))
    t1.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(t1)
    t2 = mido.MidiTrack()
    t2.append(mido.MetaMessage("track_name", name="Marimba", time=0))
    t2.append(mido.Message("program_change", program=0, channel=0, time=0))
    t2.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
    t2.append(mido.Message("note_on", note=60, velocity=0, channel=0, time=480))
    t2.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(t2)
    path = tmp_path / "src.mid"
    mid.save(path)
    return path


def test_write_corrected_midi_drops_empty_and_applies_register(tmp_path: Path):
    src = _song_with_empty_stub(tmp_path)
    mid_dir = tmp_path / "SPDMX" / "mid"
    dest = mid_dir / "x.mid"
    rows = write_corrected_midi(
        src,
        dest,
        program_by_original_track={2: 12},
        mid_rel="./mid/x.mid",
    )
    assert dest.is_file()
    assert not dest.with_suffix(dest.suffix + ".track_map.csv").is_file()
    assert len(rows) == 1
    assert rows[0]["song_id"] == "x"
    assert rows[0]["path"] == "./audio/x"
    assert rows[0]["mid"] == "./mid/x.mid"
    assert rows[0]["track"] == 0
    assert rows[0]["original_track"] == 2
    assert rows[0]["program"] == 12

    out = mido.MidiFile(filename=str(dest), charset="utf8")
    assert len(out.tracks) == 1
    assert note_on_count(out.tracks[0]) > 0
    programs = [m.program for m in out.tracks[0] if m.type == "program_change"]
    assert programs and programs[0] == 12
    assert any(m.type == "set_tempo" for m in out.tracks[0])

    csv_path = track_map_csv_path(mid_dir)
    pd.DataFrame(rows, columns=TRACK_MAP_COLUMNS).to_csv(csv_path, index=False)
    assert csv_path == tmp_path / "SPDMX" / "stems.csv"
    assert load_track_map(dest, corrected_midi_dir=mid_dir)[0]["original_track"] == 2


def test_track_map_csv_accepts_null_byte_in_track_name(tmp_path: Path):
    src = _song_with_empty_stub(tmp_path)
    midi = mido.MidiFile(filename=str(src), charset="utf8")
    midi.tracks[-1][0] = mido.MetaMessage("track_name", name="Piano\x00 \"A\"", time=0)
    midi.save(src)
    mid_dir = tmp_path / "SPDMX" / "mid"
    dest = mid_dir / "x.mid"
    rows = write_corrected_midi(src, dest, mid_rel="./mid/x.mid")
    assert rows[0]["name"] == 'Piano "A"'
    csv_path = track_map_csv_path(mid_dir)
    pd.DataFrame(rows, columns=TRACK_MAP_COLUMNS).to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)
    assert reloaded.iloc[0]["name"] == 'Piano "A"'


def test_write_corrected_midis_writes_global_track_map(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    pd.DataFrame({
        "path": ["./data/8/44/Qm.json"],
        "mid": ["./mid/8/44/Qm.mid"],
    }).to_csv(pdmx_root / "PDMX.csv", index=False)
    src = pdmx_root / "mid" / "8" / "44" / "Qm.mid"
    src.parent.mkdir(parents=True)
    stub = _song_with_empty_stub(tmp_path)
    stub.replace(src)
    register = pd.DataFrame(
        [
            {
                "mid": "./mid/8/44/Qm.mid",
                "track": 2,
                "program_corrected": 12,
            }
        ]
    )
    corrected_root = tmp_path / "SPDMX" / "mid"
    ok, failed = write_corrected_midis_from_register(
        register,
        pdmx_root=pdmx_root,
        corrected_midi_dir=corrected_root,
    )
    assert (ok, failed) == (1, 0)
    csv_path = tmp_path / "SPDMX" / "stems.csv"
    assert csv_path.is_file()
    assert not (corrected_root / "track_map.csv").is_file()
    assert not (tmp_path / "SPDMX" / "track_map.csv").is_file()
    dest = corrected_root / "8" / "44" / "Qm.mid"
    assert dest.is_file()
    assert not dest.with_name(dest.name + ".track_map.csv").is_file()
    mapped = load_track_map(dest, corrected_midi_dir=corrected_root)
    assert mapped[0]["program"] == 12
    table = pd.read_csv(csv_path)
    assert list(table.columns[:4]) == ["song_id", "path", "mid", "track"]
    assert table.iloc[0]["song_id"] == "8/44/Qm"
    assert table.iloc[0]["path"] == "./audio/8/44/Qm"
    assert table.iloc[0]["mid"] == "./mid/8/44/Qm.mid"
    assert (tmp_path / "SPDMX" / "LICENSE").is_file()
    assert (tmp_path / "SPDMX" / "README.md").is_file()


def test_write_corrected_midi_skips_note_less_song(tmp_path: Path):
    src = tmp_path / "empty.mid"
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(src)

    dest = tmp_path / "SPDMX" / "mid" / "0" / "1" / "QmEmpty.mid"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stub")
    rows = write_corrected_midi(src, dest, mid_rel="./mid/0/1/QmEmpty.mid")
    assert rows == []
    assert not dest.is_file()


def test_write_corrected_midis_uses_jobs(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    pdmx_root.mkdir()
    rows = []
    for name in ("QmA", "QmB"):
        src = pdmx_root / "mid" / "0" / "0" / f"{name}.mid"
        src.parent.mkdir(parents=True, exist_ok=True)
        stub = _song_with_empty_stub(tmp_path)
        stub.replace(src)
        rows.append({"mid": f"./mid/0/0/{name}.mid", "track": 2, "program_corrected": 12})
    pd.DataFrame({"path": ["./data/0/0/QmA.json"], "mid": ["./mid/0/0/QmA.mid"]}).to_csv(
        pdmx_root / "PDMX.csv", index=False
    )
    ok, failed = write_corrected_midis_from_register(
        pd.DataFrame(rows),
        pdmx_root=pdmx_root,
        corrected_midi_dir=tmp_path / "SPDMX" / "mid",
        jobs=2,
    )
    assert (ok, failed) == (2, 0)
    assert (tmp_path / "SPDMX" / "mid" / "0" / "0" / "QmA.mid").is_file()
    assert (tmp_path / "SPDMX" / "mid" / "0" / "0" / "QmB.mid").is_file()


def test_song_id_from_mid():
    from analysis.corrected_midi import pdmx_path_from_mid, song_id_from_mid, song_id_from_pdmx_path

    assert song_id_from_mid("./mid/8/44/Qm.mid") == "8/44/Qm"
    assert song_id_from_mid("/deepfreeze/pnlong/PDMX/PDMX/mid/8/44/Qm.mid") == "8/44/Qm"
    assert song_id_from_pdmx_path("./data/8/44/Qm.json") == "8/44/Qm"
    assert pdmx_path_from_mid("./mid/8/44/Qm.mid") == "./data/8/44/Qm.json"


def test_resolve_corrected_midi_path_strips_mid_prefix(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    src = pdmx_root / "mid" / "8" / "44" / "Qm.mid"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"mthd")
    corrected_root = tmp_path / "SPDMX" / "mid"
    path = resolve_corrected_midi_path(
        src, pdmx_root=pdmx_root, corrected_midi_dir=corrected_root
    )
    assert path == corrected_root / "8" / "44" / "Qm.mid"


def test_resolve_corrected_midi_path_falls_back_to_legacy_dev_tree(tmp_path: Path):
    pdmx_root = tmp_path / "PDMX"
    src = pdmx_root / "mid" / "8" / "44" / "Qm.mid"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"mthd")
    output_root = tmp_path / "out"
    legacy = output_root / "dev" / "mid_corrected" / "8" / "44" / "Qm.mid"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    canonical = output_root / "SPDMX" / "mid"
    path = resolve_corrected_midi_path(
        src, pdmx_root=pdmx_root, corrected_midi_dir=canonical
    )
    assert path == legacy


def test_stem_original_track_fallback():
    assert stem_original_track({"track": 3}) == 3
    assert stem_original_track({"track": 1, "original_track": 5}) == 5
    assert stem_original_track(pd.Series({"track": 2, "original_track": float("nan")})) == 2
