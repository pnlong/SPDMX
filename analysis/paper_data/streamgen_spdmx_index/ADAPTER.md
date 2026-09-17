# SPDMX adapter for stream-music-gen

1. `git clone https://github.com/lukewys/stream-music-gen`
2. Copy or symlink `spdmx_multitrack.jsonl` into the upstream data root.
3. Implement `stream_music_gen/dataset/spdmx.py` mirroring `slakh2100.py`:
   - group by `song_id`
   - load stem FLACs from `stems[].audio_path`
   - map `program` / `is_drum` → upstream instrument class ids
4. Register `spdmx` in extract_causal_dac_32k / extract_rms / dump_audio_mixdown.
5. Train: `scripts/train_dec_online.py` with `future_visibility: 0`.
6. Eval: `scripts/gen_pred/gen_and_evaluate.py` on Slakh test.
