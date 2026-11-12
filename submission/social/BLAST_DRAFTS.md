# SPDMX Twitter / LinkedIn Blast Drafts

Images: `headline.png` (tweet 1 + LinkedIn; add later) · `datasets_table.png` (tweet 2) · `stems_per_song.png` (3) · `gm_programs.png` (4) · `ablation_listening.png` (5)

Generate table + fig PNGs:

```bash
uv run python -m submission.social.render_datasets_table
uv run python -m submission.social.export_fig_pngs
```

---

# Twitter Blast Draft

---

>250k songs. ~490k open stems. Paired MIDI.

Presenting SPDMX: open stem audio from public-domain PDMX (~5.5k song-hours, CC BY 4.0) 🎧

w/ @eduardoescoto_, @zacknovack, @McAuleyLabUCSD
🧵

![][headline.png]

---

Copyright blocks redistributing training audio, especially for tasks that need individual stems. Recorded multitrack datasets are tiny or restricted. Synthetic ones lean on proprietary VSTs, or dry GM renders. 🙁
🧵2/5

![][datasets_table.png]

---

To alleviate this, SPDMX ships 253,982 songs / 490,159 mono stems / 5,469 song-hours at 44.1 kHz.

88,403 multitrack songs (2,734 h; 34.8%): largest openly-licensed synthetic multitrack dataset by song-hours.
🧵3/5

![][stems_per_song.png]

---

We correct GM programs from track names and pick open FluidSynth / MIDI-DDSP recipes via a listening study. SPDMX easily links to the original PDMX corpus through shared song identifiers.
🧵4/5

![][gm_programs.png]

---

We demo SPDMX for source separation and transcription under fixed budgets to show that it is usable alongside Slakh2100 for synthetic training.

Huge thanks to @eduardoescoto_, @zacknovack, & @McAuleyLabUCSD.

📄: arxiv.org/abs/2611.09140
🎧: pnlong.github.io/SPDMX
🧵5/5

![][ablation_listening.png]

---

# LinkedIn Post Draft

---

>250k songs. ~490k open stems. Paired MIDI.

Excited to share SPDMX: an openly-licensed stem corpus rendered from public-domain PDMX. 253,982 songs, 490,159 mono stems, and 5,469 song-hours at 44.1 kHz under CC BY 4.0, with paired MIDI and identifiers linking to PDMX metadata. The 88,403 multitrack songs (2,734 song-hours, 34.8%) are, to our knowledge, the largest openly-licensed synthetic multitrack subset by song-hours. We demo SPDMX for source separation and transcription, and release the open creation pipeline for reuse on other symbolic collections.

w/ Eduardo Escoto, Julian McAuley, and Zachary Novack

📄: https://arxiv.org/abs/2611.09140
🎧: https://pnlong.github.io/SPDMX/

Go listen on the demo page!

Abstract: Large public datasets have driven recent progress in machine learning, from text and vision to speech and general audio. Music remains an outlier, as copyright blocks redistribution of training audio, in particular for modern tasks that need individual stems rather than mixtures. Existing multitrack resources trade off realism, scale, and licensing, as synthetic alternatives rely on proprietary VSTs or naive open rendering that yields weak stems. We introduce SPDMX, an openly-licensed stem corpus rendered from the public-domain PDMX dataset. SPDMX comprises 5,469 song-hours (2,734 song-hours, or 34.8%, of which are multitrack scores) at 44.1 kHz under CC BY 4.0, with paired MIDI and identifiers that link our dataset to PDMX's original metadata. To our knowledge, SPDMX's multitrack subset constitutes the largest openly-licensed synthetic multitrack subset by song-hours. We then demo SPDMX as a dataset for both source separation and transcription, showing how its increased scale can enable better training of models on synthetic data. We release SPDMX, as well as the open creation pipeline for reuse on other symbolic music datasets.

![][headline.png]

---
