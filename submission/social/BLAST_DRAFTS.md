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

Presenting SPDMX: a listening-guided render of PDMX into CC BY 4.0 stem audio (~5.5k song-hours) 🎧

w/ @eduardoescoto_, @zacknovack, @McAuleyLabUCSD
🧵

![][headline.png]

<!-- ~195 chars -->

---

Copyright blocks redistributing training audio. Recorded multitracks are tiny or restricted. Synthetic ones lean on proprietary VSTs, or dry GM renders. 🙁
🧵2/5

![][datasets_table.png]

<!-- ~159 chars -->

---

To alleviate this, SPDMX ships 253,982 songs / 490,159 mono stems / 5,469 song-hours at 44.1 kHz.

88,403 multitrack songs (2,734 h): largest openly-licensed synthetic multitrack by song-hours.
🧵3/5

![][stems_per_song.png]

<!-- ~196 chars -->

---

We correct GM programs from track names. Stems concentrate in piano, voice, drums, and orchestral families. Every file joins PDMX metadata on song ID.
🧵4/5

![][gm_programs.png]

<!-- ~155 chars -->

---

We run a listening test to pick the most realistic open soundfonts per instrument, and use MIDI-DDSP when possible. No closed VSTs.

Huge thanks to @eduardoescoto_, @zacknovack, & @McAuleyLabUCSD.

📄: arxiv.org/abs/XXXX.XXXXX
🎧: pnlong.github.io/SPDMX
🧵5/5

![][ablation_listening.png]

<!-- ~256 chars -->

---

# LinkedIn Post Draft

---

>250k songs. ~490k open stems. Paired MIDI.

Excited to share SPDMX: a listening-guided rendering of public-domain PDMX scores into an openly-licensed stem corpus with paired MIDI. We use a listening test to select the most realistic open soundfonts per instrument category, and MIDI-DDSP when possible. 253,982 songs, 490,159 mono stems, and 5,469 song-hours at 44.1 kHz under CC BY 4.0. The 88,403 multitrack songs (2,734 song-hours) are, to our knowledge, the largest openly-licensed synthetic multitrack subset by song-hours.

w/ Eduardo Escoto, Julian McAuley, and Zachary Novack

📄: https://arxiv.org/abs/XXXX.XXXXX
🎧: https://pnlong.github.io/SPDMX/

Go listen on the demo page!

Abstract: Large public datasets have driven recent progress in machine learning, from text and vision to speech and general audio. Music remains an outlier, as copyright blocks redistribution of training audio, while many modern tasks need individual stems rather than mixtures. Existing multitrack resources trade off realism, scale, and licensing, while synthetic alternatives rely on proprietary VSTs or naive open rendering that yields weak stems. We introduce SPDMX, a listening-guided rendering of public-domain PDMX scores into an openly-licensed stem corpus with aligned scores and an open render pipeline: 253,982 songs, 490,159 mono stems, and 5,469 song-hours at 44.1 kHz under CC BY 4.0, with paired MIDI and identifiers linking to PDMX metadata. 88,403 songs (34.8%; 2,734 song-hours) provide multitrack supervision. To our knowledge, this is the largest openly-licensed synthetic multitrack subset by song-hours, enabling source separation, stem-conditioned synthesis, MIDI-to-audio, transcription, and joint symbolic–audio modeling. We release the corpus, metadata, and open pipeline for reuse on other symbolic collections. Demos and experimental notes are on the project page.

![][headline.png]

---
