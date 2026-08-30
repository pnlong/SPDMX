"""Tests for stable multi-machine song sharding."""

import pandas as pd
import pytest

from synthesis.final import parse_args
from synthesis.shard import (
    all_shard_song_ids,
    filter_work_indices_by_shard,
    partition_slices,
    shard_song_ids,
    shuffled_song_ids,
    song_id_from_synthesis_path,
    validate_shard_args,
)


def test_validate_shard_args_rejects_bad_index():
    with pytest.raises(ValueError, match="shard-index"):
        validate_shard_args(3, 3)
    with pytest.raises(ValueError, match="shard-count"):
        validate_shard_args(0, 0)


def test_partition_slices_even_and_remainder():
    assert partition_slices(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert partition_slices(9, 3) == [(0, 3), (3, 6), (6, 9)]
    assert partition_slices(1, 4) == [(0, 1), (1, 1), (1, 1), (1, 1)]


def test_shuffled_song_ids_stable_for_seed():
    songs = ["c/1/QmC", "a/1/QmA", "b/1/QmB"]
    first = shuffled_song_ids(songs, shard_count=3)
    second = shuffled_song_ids(songs, shard_count=3)
    assert first == second
    assert sorted(first) == sorted(songs)


def test_all_shards_disjoint_and_cover_universe():
    songs = [f"song/{i}" for i in range(17)]
    shards = all_shard_song_ids(songs, shard_count=4)
    assert len(shards) == 4
    union = set().union(*shards)
    assert union == set(songs)
    assert sum(len(s) for s in shards) == len(songs)
    for i, left in enumerate(shards):
        for j, right in enumerate(shards):
            if i != j:
                assert left.isdisjoint(right)


def test_shard_assignment_unchanged_when_subset_done():
    songs = [f"s/{i}" for i in range(12)]
    assigned = shard_song_ids(songs, shard_count=3, shard_index=1)
    dataset = pd.DataFrame({"song_id": songs})
    all_indices = list(range(len(songs)))
    filtered_all, _ = filter_work_indices_by_shard(
        all_indices, dataset, shard_count=3, shard_index=1,
    )
    assert {dataset.at[i, "song_id"] for i in filtered_all} == assigned

    # Simulate resume: only incomplete songs remain in work_indices.
    incomplete = [0, 2, 5, 11]
    filtered_inc, assigned_count = filter_work_indices_by_shard(
        incomplete, dataset, shard_count=3, shard_index=1,
    )
    assert assigned_count == len(assigned)
    assert {dataset.at[i, "song_id"] for i in filtered_inc} <= assigned


def test_filter_work_indices_noop_when_single_shard():
    dataset = pd.DataFrame({"song_id": ["a/1/QmA", "b/2/QmB"]})
    work = [0, 1]
    kept, assigned = filter_work_indices_by_shard(
        work, dataset, shard_count=1, shard_index=0,
    )
    assert kept == work
    assert assigned == 2


def test_song_id_from_synthesis_path_audio_and_data():
    assert song_id_from_synthesis_path(
        "/deepfreeze/share/SPDMX/SPDMX/raw/1/44/QmTest"
    ) == "1/44/QmTest"
    assert song_id_from_synthesis_path(
        "/tmp/dev/ablations/basic/data/7/19/QmAblation"
    ) == "7/19/QmAblation"


def test_parse_args_accepts_shard_flags():
    args = parse_args([
        "--only-pass", "midi_ddsp",
        "--shard-count", "3",
        "--shard-index", "2",
    ])
    assert args.shard_count == 3
    assert args.shard_index == 2


def test_parse_args_rejects_sharded_layout():
    with pytest.raises(SystemExit):
        from synthesis.final import main
        main(["--only-pass", "layout", "--shard-count", "2"])
