# Third-party dependencies (ethics / provenance)

Components used in the sPDMX synthesis pipeline beyond the core MIT-licensed
repository code. Training-data provenance is recorded for the paper ethics table.

## Fluidsynth + soundfonts

- **Fluidsynth** — soundfont MIDI renderer (system / conda package).
- **Soundfont banks** — catalogued in `experiments/patch_sweep/soundfonts.yaml`
  and `archive_soundfonts.yaml` with archive.org / schristiancollins sources.
  Prefer public-domain / redistributable banks for shipped ablations.

## MIDI-DDSP (neural orchestral stems, ablation B3)

- **Software:** [magenta/midi-ddsp](https://github.com/magenta/midi-ddsp) —
  **Apache License 2.0**.
- **Paper:** Wu, Y., et al. (2022). MIDI-DDSP: Detailed Control of Musical
  Performance via Hierarchical Modeling. ICLR.
- **Training data:** [URMP](https://www2.ece.rochester.edu/projects/air/projects/URMP.html)
  (University of Rochester Multi-Modal Music Performance) — purpose-recorded
  monophonic orchestral performances for research (not scraped web audio).
- **Instruments:** violin, viola, cello, double bass, flute, oboe, clarinet,
  saxophone, bassoon, trumpet, horn, trombone, tuba (monophonic stems only).
- **Weights:** URMP checkpoint via `midi_ddsp_download_model_weights` or
  `https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip`.

## DDSP-Piano (neural piano stems, ablation B3)

- **Software:** [lrenault/ddsp-piano](https://github.com/lrenault/ddsp-piano) —
  **Apache License 2.0** (single-maintainer academic repo; spot-listen before
  large runs).
- **Papers** (upstream README BibTeX):
  - Renault, L., Mignot, R., & Roebel, A. (2022). Differentiable Piano Model for
    MIDI-to-Audio Performance Synthesis. Proc. DAFx.
  - Renault, L., Mignot, R., & Roebel, A. (2023). DDSP-Piano: A Neural Sound
    Synthesizer Informed by Instrument Knowledge. *Journal of the Audio
    Engineering Society*, 71(9), 552–565. https://doi.org/10.17743/jaes.2022.0102
- **Training data:** [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
  — live International Piano-e-Competition performances on Yamaha Disklaviers,
  purpose-recorded for research.
- **Scope:** GM piano family; polyphony-native. Default checkpoint: `dafx22`
  @ 16 kHz (resampled to 44.1 kHz in-pipeline). Not Magenta Wave2Midi2Wave
  (Midi2Wave checkpoint was never released).
- **Inference note:** upstream `synthesize_midi_file.py` builds on a 1s dummy
  batch which breaks variable-length inference under TF 2.15; our worker builds
  with the actual sequence duration instead.

## Explicitly out of scope (provenance)

- Lyric-conditioned singing-voice synthesis (DiffSinger, NNSVS, etc.): no
  open-weight English SVS checkpoint met the same training-data provenance bar
  as URMP/MAESTRO at the time of this work. Vocal stems use the soundfont path.
