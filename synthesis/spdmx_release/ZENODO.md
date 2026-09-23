# SPDMX on Zenodo — logistics and deposit text

## Scale (why this is hard)

| Metric | Value |
|--------|------|
| Songs | 253,982 |
| Mono stems | 490,159 |
| Mixture hours | 5,469 h @ 44.1 kHz |
| Packaged media | **~1.7 TiB** (64 chunks × ~27 GiB each) |
| Metadata only | ~few MB (`stems.csv`, `songs.csv`, `chunks.csv`, docs) |

For comparison, [PDMX on Zenodo](https://zenodo.org/records/13763756) is **1.6 GB** (symbolic JSON only). SPDMX is roughly **1,000× larger** because it ships rendered FLAC stems, mixes, and dense MIDI.

The repo already targets this layout via `synthesis.build_spdmx` and `synthesis.distribute_spdmx` (default **64** roughly equal-sized chunk zips, `store` compression — FLAC is already compressed).

## Zenodo limits (as of 2026)

Per [Zenodo file limits](https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo):

| Limit | Value |
|-------|------|
| Default per record | **50 GB**, max **100 files** |
| Account extra quota | **+150 GB** total, assignable across records (→ up to **200 GB** on one record) |
| Fair usage | Splitting *only* to dodge the 50 GB cap is discouraged; **contact support first** for multi‑TB deposits |
| After publish | Metadata editable anytime; **files editable ~45 days**, then versioning required |
| Upload methods | Web UI or [REST API](https://developers.zenodo.org/) (`PUT` to record bucket; use long HTTP timeouts) |
| Resumable uploads | **Not reliably available** on Zenodo today; large uploads over unstable links are risky |

**Bottom line:** a single Zenodo record cannot hold ~1.7 TiB. Each ~27 GiB chunk zip *does* fit under the 50 GB per-record cap, but you still need **~65 records** (1 metadata + 64 media) or an external bulk host.

## Recommended deposit structure

### Option A — Zenodo metadata record + 64 chunk records (preferred if Zenodo agrees)

Best aligned with `chunks.csv` and partial download.

1. **Record 0 — Metadata (small, cite this DOI in the paper)**
   - Files: `README.md`, `LICENSE`, `link_single_track_mixes.sh`, `stems.csv`, `songs.csv`, `chunks.csv`, `SHA256SUMS`
   - Description: full dataset documentation (below)
   - `Related identifiers`: link to Records 1–64 (`HasPart` / `IsMetadataFor`)

2. **Records 1–64 — Media chunks**
   - One file each: `chunk_N.zip` (~27 GiB)
   - Minimal title: `SPDMX chunk N (of 64)`
   - `Related identifiers`: `IsPartOf` → Record 0 DOI
   - Same license, same creators

**Before uploading:** email [Zenodo support](https://zenodo.org/support) with:
- Total size (~1.7 TiB), 64 × ~27 GiB zips
- Chunks are independently downloadable (users filter `chunks.csv` / `stems.csv`)
- You are **not** splitting arbitrarily — chunk size matches the documented release format
- Ask whether they consider this fair usage and if they recommend a Zenodo **community** for the collection

### Option B — Zenodo metadata DOI + bulk mirror (pragmatic for upload pain)

- Zenodo Record 0 only (metadata + `chunks.csv` with **mirror URLs** and SHA-256)
- Bulk zips on Hugging Face Datasets, Internet Archive, or UCSD/institutional storage
- Zenodo `Related identifiers`: `IsSupplementedBy` → mirror landing page

Useful if upload bandwidth to CERN is the bottleneck (27 GB × 64 sequential uploads ≈ many days).

### Option C — EU Open Research Repository (only if EU-funded)

Some EU project communities allow **200 GB per record** — still far below one full copy; would not remove the need for chunking or an external mirror.

## Upload workflow (lab)

```bash
# 1. Package chunked release (once render is complete)
uv run python -m synthesis.final --only-pass mix -j 8
uv run python -m synthesis.build_spdmx -o "$SPDMX_OUTPUT_DIR"

# 2. Stage Zenodo-ready tree (metadata + chunk zips + SHA256SUMS)
uv run python -m synthesis.distribute_spdmx -o "$SPDMX_OUTPUT_DIR" \
  --stage-dir /path/to/zenodo_stage

# 3. Upload from a stable server (not a laptop on Wi‑Fi)
#    - Prefer machine with good connectivity to CERN/Geneva
#    - Verify each zip against SHA256SUMS after upload
#    - Script uploads via Zenodo REST API; set client timeout >> 27GB/upload
```

Staging produces:

```
zenodo_stage/
├── LICENSE
├── README.md
├── stems.csv
├── songs.csv          # if built (synthesis.build_songs_table / build_spdmx)
├── chunks.csv
├── SHA256SUMS
├── chunk_0.zip        # ~27 GiB each
├── chunk_1.zip
└── …
```

**Upload order:** publish metadata record first (get DOI), then chunk records with cross-links.

**Integrity:** `SHA256SUMS` covers all staged files; `chunks.csv` lists per-chunk `sha256` after staging.

## Zenodo metadata record — copy/paste fields

### Title

```
SPDMX: Public-Domain Multitrack Stem Audio at Scale (metadata)
```

(Chunk records: `SPDMX: chunk 0 of 64 (multitrack stem audio)`.)

### Resource type

`Dataset`

### Authors

Phillip Long, Eduardo Escoto, Julian McAuley, Zachary Novack — University of California San Diego

### License

`Creative Commons Attribution 4.0 International (CC BY 4.0)` — match `LICENSE` in the release tree.

### Keywords

```
multitrack stems, source separation, music information retrieval, symbolic-to-audio,
General MIDI, FluidSynth, public domain, PDMX, synthetic dataset, audio dataset
```

### Description (main text)

```
SPDMX (Synthesized PDMX) is an open, redistributable multitrack stem audio corpus
derived from the public-domain PDMX symbolic collection (Long et al., ICASSP 2025).
Each PDMX song is identified by song_id and rendered into mono FLAC stems, a linear
full-mix FLAC, and dense corrected MIDI at 44.1 kHz.

This record contains metadata and documentation only. Audio is split into 64 chunk
archives (chunk_0.zip … chunk_63.zip, ~27 GiB each, ~1.7 TiB total). Download
chunks.csv to plan partial downloads; join to stems.csv and songs.csv on song_id.

Corpus statistics (corrected GM register, non-empty tracks):
  • 253,982 songs
  • 490,159 mono stems
  • 5,469 hours of mixture audio
  • Song-aligned with PDMX for symbolic metadata and scores

Synthesis pipeline:
  1. GM program correction from track-name aliases; empty tracks removed.
  2. Listening study selects an open render recipe (FluidSynth soundfont variety +
     selective MIDI-DDSP for sustained monophonic winds/brass/strings).
  3. Loudness- and velocity-aware mixing; stems stored as mono FLAC.

Non-silent hours (reported in the paper) sum 0.25 s windows with RMS ≥ 0.01
(≈ −40 dBFS), excluding near-silent gaps between notes.

Please cite the SPDMX paper (ICASSP 2026) and PDMX if you use this dataset.
Demo and code: https://pnlong.github.io/SPDMX/
```

### Technical info

```
Files in this metadata record
-----------------------------
README.md                    — layout, CSV schemas, download instructions
LICENSE                      — CC BY 4.0 (+ notes on PDMX public-domain scores)
link_single_track_mixes.sh   — optional: symlink mix.flac → 0.flac for singles
stems.csv                    — one row per stem (song_id, track, program, chunk, paths, …)
songs.csv                    — one row per song (subsets, programs, song_length, …)
chunks.csv                   — per-chunk song/stem counts, bytes, archive name, sha256
SHA256SUMS                   — checksums for all release files including chunk zips

Media archives (separate downloads / related records)
---------------------------------------------------
chunk_N.zip     — unpack beside the CSVs; paths are ./chunk_N/<song_id>/…

Per song directory:
  {track}.flac   — mono stem
  mix.flac       — linear sum of stems (mono); omitted for single-track songs
  mix.mid        — dense corrected MIDI

Single-track songs omit mix.flac (identical to 0.flac). Optionally run
./link_single_track_mixes.sh after unzip to create mix.flac → 0.flac.

Join to PDMX
------------
song_id matches PDMX.csv path with ./data/ prefix and .json suffix removed.

Example (Python):
  songs = pd.read_csv("songs.csv")
  stems = pd.read_csv("stems.csv")
  bdgp = stems[stems["song_id"].isin(songs.loc[songs["subset:bdgp"], "song_id"])]
  multi = stems[stems["song_id"].isin(songs.loc[songs["subset:multitrack"], "song_id"])]
```

### Notes (download / verify)

```
1. Download this metadata record (CSVs, README, LICENSE, linker script, SHA256SUMS).
2. Choose chunks from chunks.csv (or filter stems.csv by chunk).
3. Download the corresponding chunk_N.zip files from the related chunk records.
4. Unzip each archive next to the CSVs so paths like ./chunk_0/<song_id>/… resolve.
5. Optionally: ./link_single_track_mixes.sh  # mix.flac → 0.flac for singles
6. Verify: sha256sum -c SHA256SUMS

Partial download example:
  stems[stems["chunk"] == 0]   # only songs whose media is in chunk_0
```

### Related identifiers (fill after upload)

| This record | Relation | Target |
|-------------|----------|--------|
| Metadata | `HasPart` | DOI of each chunk record |
| Chunk N | `IsPartOf` | DOI of metadata record |
| Metadata | `IsDerivedFrom` | PDMX Zenodo DOI `10.5281/zenodo.13763756` |
| Metadata | `IsSupplementTo` | SPDMX ICASSP 2026 paper DOI (when available) |

## Checklist before publishing

- [ ] Contact Zenodo support about ~1.7 TiB / 64-record plan
- [ ] Run `distribute_spdmx` on final `SPDMX/` tree; verify `SHA256SUMS`
- [ ] Spot-check unzip layout against `stems.csv` paths
- [ ] Upload metadata record; mint DOI; add to paper / demo site
- [ ] Upload chunk records (automate via API); cross-link Related identifiers
- [ ] Publish code separately (GitHub + optional Zenodo software archive)
