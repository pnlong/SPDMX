"""Tests for ablation paths and dataset sampling."""

from pathlib import Path

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SIZE,
    DEV_DIR_NAME,
    OUTPUT_DIR,
    SPDMX_AUDIO_DIR_NAME,
    SPDMX_DATASET_DIR_NAME,
    SPDMX_MID_DIR_NAME,
    STEMS_DIR_NAME,
    STEMS_REALIFY_DIR_NAME,
)
from synthesis.dataset import (
    prepare_ablation_dataset,
    prepare_full_dataset,
    stratified_song_sample,
    write_listening_sample,
)
from synthesis.paths import (
    ablation_dir,
    ablation_raw_dir,
    ablation_realify_dir,
    ablations_root,
    condition_name,
    dev_root,
    full_stems_dir,
    full_stems_realify_dir,
    song_lengths_dir,
    spdmx_audio_dir,
    spdmx_dataset_dir,
    spdmx_mid_dir,
)


def _fake_pdmx(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "path": [f"./data/{i}/song.json" for i in range(n)],
        "mid": [f"./data/{i}/song.mid" for i in range(n)],
        "subset:rated_deduplicated": [i % 2 == 0 for i in range(n)],
    })


def _fake_register_for_pdmx(dataset: pd.DataFrame) -> pd.DataFrame:
    """One piano stem and one drum stem per song mid."""
    rows = []
    for _, row in dataset.iterrows():
        if not row["subset:rated_deduplicated"]:
            continue
        mid = row["mid"]
        rows.append({
            "mid": mid,
            "track": 0,
            "name": "Piano",
            "is_drum": False,
            "program_corrected": 0,
        })
        rows.append({
            "mid": mid,
            "track": 1,
            "name": "Drums",
            "is_drum": True,
            "program_corrected": 0,
        })
    return pd.DataFrame(rows)


def test_dev_root():
    assert dev_root("/out") == f"/out/{DEV_DIR_NAME}"


def test_ablation_paths():
    assert ablations_root(OUTPUT_DIR).endswith(f"/{DEV_DIR_NAME}/ablations")
    assert ablation_dir(OUTPUT_DIR, "basic") == f"{OUTPUT_DIR}/{DEV_DIR_NAME}/ablations/basic"
    assert condition_name("basic", realify=True) == "basic_realify"


def test_ablation_output_dirs():
    assert ablation_raw_dir("/out", "basic") == f"/out/{DEV_DIR_NAME}/ablations/basic"
    assert ablation_realify_dir("/out", "slakh") == f"/out/{DEV_DIR_NAME}/ablations/slakh_realify"


def test_full_stem_dirs():
    assert full_stems_dir("/out") == f"/out/{DEV_DIR_NAME}/{STEMS_DIR_NAME}"
    assert full_stems_realify_dir("/out") == f"/out/{DEV_DIR_NAME}/{STEMS_REALIFY_DIR_NAME}"


def test_song_lengths_dir():
    assert song_lengths_dir("/out") == f"/out/{DEV_DIR_NAME}/analysis/song_lengths"


def test_spdmx_dataset_dir():
    from shared.config import SPDMX_DEV_DIR_NAME
    from synthesis.paths import mid_corrected_dir, production_tables_dir, spdmx_dev_dir

    assert spdmx_dataset_dir("/out") == f"/out/{SPDMX_DATASET_DIR_NAME}"
    assert spdmx_dev_dir("/out") == f"/out/{SPDMX_DEV_DIR_NAME}"
    assert spdmx_audio_dir("/out") == f"/out/{SPDMX_DEV_DIR_NAME}/{SPDMX_AUDIO_DIR_NAME}"
    assert spdmx_mid_dir("/out") == f"/out/{SPDMX_DEV_DIR_NAME}/{SPDMX_MID_DIR_NAME}"

    assert mid_corrected_dir("/out") == spdmx_mid_dir("/out")
    assert production_tables_dir("/out") == f"/out/{DEV_DIR_NAME}/final"


def test_song_output_dir_swaps_data_for_audio():
    from synthesis.synthesize import song_output_dir

    json_path = "/pdmx/data/1/11/QmSong.json"
    assert song_output_dir("/out", "/pdmx", json_path) == "/out/data/1/11/QmSong"
    assert song_output_dir(
        "/out", "/pdmx", json_path, tree_dir_name=SPDMX_AUDIO_DIR_NAME,
    ) == "/out/audio/1/11/QmSong"


def test_ablation_sample_size_and_seed():
    df = prepare_ablation_dataset(_fake_pdmx(), sample_size=10, sample_seed=0)
    assert len(df) == 10
    df2 = prepare_ablation_dataset(_fake_pdmx(), sample_size=10, sample_seed=0)
    assert list(df["path"]) == list(df2["path"])


def test_ablation_filters_subset():
    df = prepare_ablation_dataset(_fake_pdmx(), sample_size=ABLATION_SAMPLE_SIZE)
    assert df["subset:rated_deduplicated"].all()


def test_full_dataset_keeps_all_rows():
    df = prepare_full_dataset(_fake_pdmx(50))
    assert len(df) == 50


def test_stratified_song_sample_fills_categories():
    dataset = _fake_pdmx(80)
    register = _fake_register_for_pdmx(dataset)
    selected, inventory = stratified_song_sample(
        dataset[dataset["subset:rated_deduplicated"]],
        register,
        min_stems_per_category=3,
        max_songs=40,
        sample_seed=43,
    )
    assert len(selected) > 0
    assert len(inventory) >= len(selected)
    piano = sum(1 for s in inventory if s["category"] == "piano")
    drums = sum(1 for s in inventory if s["category"] == "drums")
    assert piano >= 3
    assert drums >= 3


def test_listening_sample_roundtrip(tmp_path: Path):
    dataset = _fake_pdmx(40)
    register = _fake_register_for_pdmx(dataset)
    rated = dataset[dataset["subset:rated_deduplicated"]].reset_index(drop=True)
    selected, inventory = stratified_song_sample(
        rated,
        register,
        min_stems_per_category=2,
        max_songs=20,
        sample_seed=43,
    )
    sample_file = tmp_path / "listening_sample.yaml"
    write_listening_sample(
        sample_file,
        selected,
        inventory,
        sample_seed=43,
        min_stems_per_category=2,
        max_songs=20,
    )
    reloaded = prepare_ablation_dataset(
        dataset,
        listening_sample_file=sample_file,
        persist_sample=False,
    )
    assert list(reloaded["path"]) == list(selected["path"])
