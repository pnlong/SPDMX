# SPDMX on Mozilla Data Collective

## Scale

| Metric | Value |
|--------|------|
| Songs | 253,982 |
| Mono stems | 490,159 |
| Mixture hours | 5,469 h @ 44.1 kHz |
| Packaged media | **1.595 TB** archive (1.451 TiB; 1,594,920,765,289 bytes) |
| Layout chunks | 64 × ~23.556 GiB (spread under 1 MB) |
| MDC parts | 2 × ~0.80 TB (even / odd chunk ids), under the 1 TB cap |

MDC accepts one `.tar.gz` per dataset (plus an optional smaller sample `.tar.gz`)
and the full archive is over the 1 TB upload cap. Publish **two datasets**.
Internal `chunk_N/` folders stay inside each part so `stems.csv` paths match the
release layout. Part 1 is even chunk ids (`0, 2, …, 62`). Part 2 is odd chunk
ids (`1, 3, …, 63`). That split does not depend on file size.

## Stage archives (lab)

```bash
# Run from the repo so `synthesis` is importable:
cd /home/pnlong/spdmx

# Defaults: dataset=/deepfreeze/share/SPDMX/SPDMX
#           stage=/deepfreeze/share/SPDMX/SPDMX_mdc
uv run python -m synthesis.distribute_spdmx -o /deepfreeze/share/SPDMX

# Sample only (fast), keep unpacked tree for inspection
uv run python -m synthesis.distribute_spdmx -o /deepfreeze/share/SPDMX \
  --skip-full --keep-sample-dir

# Full only
uv run python -m synthesis.distribute_spdmx -o /deepfreeze/share/SPDMX --skip-sample
```

Produces:

```
SPDMX_mdc/
├── spdmx-part1.tar.gz    # even chunk ids + full CSV index
├── spdmx-part2.tar.gz    # odd chunk ids + the same CSV index
├── spdmx-sample.tar.gz   # 16 multitrack + 16 single-track, seed 42
├── PARTS.md
└── SHA256SUMS
```

`--parts 1` still writes the legacy single `spdmx.tar.gz` (too large for MDC).
The command refuses a part whose raw media plus metadata exceeds 950 GB, and it
does not move frozen songs to get under that cap.

Upload each part as its own submission with the MDC web UI or the
`datacollective` Python SDK (resumable multipart; `part_size ≈ 200 MiB` is about
4,000 parts per ~0.80 TB archive, under the 10,000-part cap). Attach
`spdmx-sample.tar.gz` to part 1. Reuse that same sample on part 2 only if that
form requires one. Opt out of exclusivity if you also host elsewhere.

## Datasheet copy/paste

Short description, long description, intended usage, and ethical review are
**plain text**. Restrictions / forbidden / additional / other information are
Markdown-ready.

### Short description (≤160 characters, plain text)

This is the only card field that differs between the two submissions.
Suggested listing titles, separate from this field: `SPDMX (part 1 of 2)` and
`SPDMX (part 2 of 2)`.

Part 1 (126 characters):

```
Public-domain FLAC stems + dense MIDI for 253,982 PDMX songs. Mixes provided as well (see below). Part 1 of 2 (download both).
```

Part 2 (126 characters):

```
Public-domain FLAC stems + dense MIDI for 253,982 PDMX songs. Mixes provided as well (see below). Part 2 of 2 (download both).
```

### Long description (plain text)

```
SPDMX (Synthesized PDMX) renders the public-domain PDMX symbolic corpus into mono FLAC stems, optional linear-sum mixes, and dense corrected MIDI at 44.1 kHz (253,982 songs; ~5,469 mixture hours). Join to PDMX on song_id. Layout: chunk_N/<song_id>/{k.flac, mix.mid, mix.flac?}.

Mozilla Data Collective hosts this corpus as two archives, part 1 of 2 and part 2 of 2. Download both and extract them into the same directory. Part 1 contains even chunk ids (chunk_0, chunk_2, ...) and part 2 contains odd chunk ids. The CSV indexes in each archive describe the full corpus. Then optionally run ./link_single_track_mixes.sh.

Single-track songs (~65%) do not ship mix.flac (it would duplicate 0.flac). The mix CSV column still points at …/mix.flac. Resolve by (1) loading 0.flac when subset:multitrack is false / n_tracks == 1, or (2) running ./link_single_track_mixes.sh after extract to create mix.flac → 0.flac symlinks. Multitrack songs already include mix.flac. See README.md in the archive.

Synthesis uses open FluidSynth soundfonts plus selective MIDI-DDSP / DDSP-Piano. License: CC BY 4.0. Demo: https://pnlong.github.io/SPDMX/
```

### Restrictions (user agreement, Markdown)

```markdown
**License and attribution.** Use is subject to [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

- Provide appropriate attribution to **SPDMX**.
- When using underlying scores or PDMX-derived metadata, also attribute **PDMX** (Long et al., ICASSP 2025).

**Nature of the audio.** The audio is **machine-synthesized** from symbolic scores. It is **not** a collection of human studio recordings. Do not present SPDMX stems or mixes as live or commercial studio performances of identifiable musicians.

**Public-domain provenance.** Underlying scores and public-domain designations originate in **PDMX**. We rely on PDMX’s public-domain labeling for those symbolic sources; SPDMX does **not** independently re-clear copyright for every MuseScore/PDMX item. Users remain responsible for complying with applicable law in their jurisdiction when redistributing or commercially exploiting derived works.
```

### Forbidden usages (user agreement, Markdown)

```markdown
**Misrepresentation.**

- Do **not** misrepresent SPDMX audio as human-recorded performances.
- Do **not** claim the dataset contains voice or instrumental recordings of identifiable people.

**Attribution.** Do **not** remove or obscure required attribution to SPDMX / PDMX when sharing the data or models trained on it, beyond what CC BY 4.0 already requires.

**Warranty of title.** Do **not** assert that SPDMX’s authors warrant that every underlying score is public domain in every jurisdiction. Public-domain status is **inherited from PDMX’s curation and labeling**.
```

### Additional information (user agreement, Markdown)

```markdown
**Voices.** Any pitched “voice”- or choir-like stems are synthesized from MIDI via open soundfonts / open models — **not** recordings of people.

**Single-track mixes.** Single-track songs **omit** `mix.flac` (it would be identical to `0.flac`).

- Load `0.flac` when `subset:multitrack` is false / `n_tracks == 1`, **or**
- Run `./link_single_track_mixes.sh` after extract to create `mix.flac → 0.flac` symlinks.

Full schema and joins are in `README.md` inside the archive.
```

### Other information (datasheet / technical, Markdown)

```markdown
**Provenance.** SPDMX is a listening-guided rendering of [PDMX](https://zenodo.org/records/13763756), a large public-domain MusicXML/MIDI collection.

- We take song identity (`song_id`) and PDMX’s public-domain-oriented metadata **as given**.
- We then synthesize mono FLAC stems and dense corrected MIDI with **open tools only**:
  - FluidSynth + redistributable soundfonts
  - Selective MIDI-DDSP / DDSP-Piano (**Apache 2.0** weights; research training corpora such as URMP / MAESTRO — SPDMX does not redistribute those recordings)

See third-party / ethics notes in the code repository for synthesizer provenance.

**Preprocessing and layout.**

- GM program correction from track-name aliases; empty tracks dropped
- Loudness- / velocity-aware mixing so multitrack stems remain summable
- Release layout uses `chunk_N/` directories for sharding. On Mozilla Data Collective the corpus is two `.tar.gz` archives (part 1 of 2 and part 2 of 2). Extract both into one directory. Even chunk ids are in part 1; odd chunk ids are in part 2. `chunk_assignment.csv` keeps each song in the same chunk across later releases.

**Scale.** 253,982 songs; 490,159 mono stems; ~5,469 mixture hours @ 44.1 kHz. Demo and paper materials: [pnlong.github.io/SPDMX](https://pnlong.github.io/SPDMX/)

**Disclaimer.** Provided **AS IS** without warranty of title, non-infringement, or fitness for a particular purpose, to the extent permitted by law. CC BY 4.0 terms apply to the SPDMX release files themselves.
```

### Intended usage (plain text)

```
This dataset is intended for machine learning and MIR research and development, including source separation, multi-instrument transcription, generative music, stem-aware retrieval and data augmentation, and related audio/symbolic modeling. It is synthesized training/evaluation audio paired with MIDI, not a singer-identity corpus.
```

### Ethical review (plain text)

```
No human-subjects protocol or IRB review was required for SPDMX audio creation: there were no participant recordings or personally identifying performance captures. Audio is synthesized from symbolic scores that PDMX curated and labeled as public domain. We do not collect demographic data about composers or performers as research subjects. Users should not treat synthesized timbres as biometric or identity-bearing human data. For concerns about a specific underlying score’s status, refer to PDMX’s documentation and applicable public-domain law; SPDMX inherits PDMX’s public-domain designations rather than conducting a new rights clearance for each item. Rendering uses redistributable soundfonts and open neural synthesizer weights (Apache 2.0; MIDI-DDSP / DDSP-Piano), not closed commercial VST sample libraries. Neural models were trained upstream on purpose-recorded research corpora (e.g. URMP, MAESTRO); SPDMX emits new synthesized audio conditioned on public-domain MIDI.
```
### Other fields

- **License:** CC BY 4.0
- **Format:** FLAC, MIDI, CSV
- **Locale:** `zxx` (no linguistic content)
- **Task:** `ML` or `OTH` (no MIR option; avoid ASR/TTS)
- **Visibility:** Public

### Compliance badges

Leave **all** compliance dropdowns as **Unspecified**.

| Badge | Recommendation | Why |
|-------|----------------|-----|
| Contributors gave explicit consent | **Unspecified** (do **not** select Yes) | No human contributors / voice donors; audio is synthesized from PDMX PD scores. “Yes” would imply a consent pipeline you don’t have. “No” can read as “collected people without consent.” |
| Contributors were compensated fairly | **Unspecified** (do **not** select Yes) | Same — there were no paid data contributors for SPDMX audio. Don’t claim fair pay or imply unpaid people. |

For any other Unspecified rows on the form (if unlabeled in the UI): keep **Unspecified** unless the label clearly fits synthetic / public-domain symbolic data.

**Show compliance attributes as badges on the public page:** prefer **unchecked** so you don’t publish a row of vague Unspecified badges. Your ethical-review / restrictions text already covers provenance.
