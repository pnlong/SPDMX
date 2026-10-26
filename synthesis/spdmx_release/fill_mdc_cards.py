"""Write both SPDMX Mozilla Data Collective draft cards.

Markdown fields are triple-quoted source text, so paragraph breaks stay as
newline characters in the JSON body. The SDK only strips leading and trailing
whitespace.

Does not submit the drafts for review. Requires ``MDC_API_KEY``,
``MDC_CONTACT_NAME``, and ``MDC_CONTACT_EMAIL``.

    uv run --no-project --python 3.11 --with 'datacollective>=0.4.5' \\
        python synthesis/spdmx_release/fill_mdc_cards.py
"""

from __future__ import annotations

import os
import textwrap

PART1_ID = "cmvn87mrb02vy07o5v8ljmnx8"
PART2_ID = "cmvn87nbr02vz07o5lzjm5b07"

SHORT = {
    PART1_ID: (
        "Public-domain FLAC stems + dense MIDI for 253,982 PDMX songs. "
        "Mixes provided as well (see below). Part 1 of 2 (download both)."
    ),
    PART2_ID: (
        "Public-domain FLAC stems + dense MIDI for 253,982 PDMX songs. "
        "Mixes provided as well (see below). Part 2 of 2 (download both)."
    ),
}
TITLES = {
    PART1_ID: "SPDMX (part 1 of 2)",
    PART2_ID: "SPDMX (part 2 of 2)",
}

LONG_DESCRIPTION = textwrap.dedent(
    """\
    SPDMX (Synthesized PDMX) renders the public-domain PDMX symbolic corpus into mono FLAC stems, optional linear-sum mixes, and dense corrected MIDI at 44.1 kHz (253,982 songs; ~5,469 mixture hours). Join to PDMX on song_id. Layout: chunk_N/<song_id>/{k.flac, mix.mid, mix.flac?}.

    Mozilla Data Collective hosts this corpus as two archives, parts 1 and 2. Download both and extract them into the same directory. Part 1 contains even chunk ids (chunk_0, chunk_2, ...) and part 2 contains odd chunk ids. The CSV indexes in each archive describe the full corpus. Then optionally run ./link_single_track_mixes.sh.

    Single-track songs (~65%) do not ship mix.flac (it would duplicate 0.flac). The mix CSV column still points at …/mix.flac. Resolve by (1) loading 0.flac when subset:multitrack is false / n_tracks == 1, or (2) running ./link_single_track_mixes.sh after extract to create mix.flac → 0.flac symlinks. Multitrack songs already include mix.flac. See README.md in the archive.

    Synthesis uses open FluidSynth soundfonts plus selective MIDI-DDSP / DDSP-Piano. License: CC BY 4.0. Demo: https://pnlong.github.io/SPDMX/
    """
).strip()

RESTRICTIONS = textwrap.dedent(
    """\
    **License and attribution.** Use is subject to [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

    - Provide appropriate attribution to **SPDMX**.
    - When using underlying scores or PDMX-derived metadata, also attribute **PDMX** (Long et al., ICASSP 2025).

    **Nature of the audio.** The audio is **machine-synthesized** from symbolic scores. It is **not** a collection of human studio recordings. Do not present SPDMX stems or mixes as live or commercial studio performances of identifiable musicians.

    **Public-domain provenance.** Underlying scores and public-domain designations originate in **PDMX**. We rely on PDMX’s public-domain labeling for those symbolic sources; SPDMX does **not** independently re-clear copyright for every MuseScore/PDMX item. Users remain responsible for complying with applicable law in their jurisdiction when redistributing or commercially exploiting derived works.
    """
).strip()

FORBIDDEN = textwrap.dedent(
    """\
    **Misrepresentation.**

    - Do **not** misrepresent SPDMX audio as human-recorded performances.
    - Do **not** claim the dataset contains voice or instrumental recordings of identifiable people.

    **Attribution.** Do **not** remove or obscure required attribution to SPDMX / PDMX when sharing the data or models trained on it, beyond what CC BY 4.0 already requires.

    **Warranty of title.** Do **not** assert that SPDMX’s authors warrant that every underlying score is public domain in every jurisdiction. Public-domain status is **inherited from PDMX’s curation and labeling**.
    """
).strip()

ADDITIONAL = textwrap.dedent(
    """\
    **Voices.** Any pitched “voice”- or choir-like stems are synthesized from MIDI via open soundfonts / open models — **not** recordings of people.

    **Single-track mixes.** Single-track songs **omit** `mix.flac` (it would be identical to `0.flac`).

    - Load `0.flac` when `subset:multitrack` is false / `n_tracks == 1`, **or**
    - Run `./link_single_track_mixes.sh` after extract to create `mix.flac → 0.flac` symlinks.

    Full schema and joins are in `README.md` inside the archive.
    """
).strip()

OTHER = textwrap.dedent(
    """\
    **Provenance.** SPDMX is a listening-guided rendering of [PDMX](https://zenodo.org/records/13763756), a large public-domain MusicXML/MIDI collection.

    - We take song identity (`song_id`) and PDMX’s public-domain-oriented metadata **as given**.
    - We then synthesize mono FLAC stems and dense corrected MIDI with **open tools only**:
      - FluidSynth + redistributable soundfonts
      - Selective MIDI-DDSP / DDSP-Piano (**Apache 2.0** weights; research training corpora such as URMP / MAESTRO — SPDMX does not redistribute those recordings)

    See third-party / ethics notes in the code repository for synthesizer provenance.

    **Preprocessing and layout.**

    - GM program correction from track-name aliases; empty tracks dropped
    - Loudness- / velocity-aware mixing so multitrack stems remain summable
    - Release layout uses `chunk_N/` directories for sharding. On Mozilla Data Collective the corpus is two `.tar.gz` archives (parts 1 and 2). Extract both into one directory. Even chunk ids are in part 1; odd chunk ids are in part 2. `chunk_assignment.csv` keeps each song in the same chunk across later releases.

    **Scale.** 253,982 songs; 490,159 mono stems; ~5,469 mixture hours @ 44.1 kHz. Demo and paper materials: [pnlong.github.io/SPDMX](https://pnlong.github.io/SPDMX/)

    **Disclaimer.** Provided **AS IS** without warranty of title, non-infringement, or fitness for a particular purpose, to the extent permitted by law. CC BY 4.0 terms apply to the SPDMX release files themselves.
    """
).strip()

INTENDED = (
    "This dataset is intended for machine learning and MIR research and development, "
    "including source separation, multi-instrument transcription, generative music, "
    "stem-aware retrieval and data augmentation, and related audio/symbolic modeling. "
    "It is synthesized training/evaluation audio paired with MIDI, not a singer-identity corpus."
)

ETHICAL = (
    "No human-subjects protocol or IRB review was required for SPDMX audio creation: "
    "there were no participant recordings or personally identifying performance captures. "
    "Audio is synthesized from symbolic scores that PDMX curated and labeled as public domain. "
    "We do not collect demographic data about composers or performers as research subjects. "
    "Users should not treat synthesized timbres as biometric or identity-bearing human data. "
    "For concerns about a specific underlying score’s status, refer to PDMX’s documentation "
    "and applicable public-domain law; SPDMX inherits PDMX’s public-domain designations rather "
    "than conducting a new rights clearance for each item. Rendering uses redistributable "
    "soundfonts and open neural synthesizer weights (Apache 2.0; MIDI-DDSP / DDSP-Piano), not "
    "closed commercial VST sample libraries. Neural models were trained upstream on "
    "purpose-recorded research corpora (e.g. URMP, MAESTRO); SPDMX emits new synthesized audio "
    "conditioned on public-domain MIDI."
)

MARKDOWN_FIELDS = {
    "restrictions": RESTRICTIONS,
    "forbiddenUsage": FORBIDDEN,
    "additionalConditions": ADDITIONAL,
    "other": OTHER,
}


def _require_breaks(label: str, value: str) -> None:
    if "\n\n" not in value:
        raise SystemExit(f"{label} is missing paragraph breaks")
    if any(line.startswith("    ") for line in value.splitlines()):
        raise SystemExit(f"{label} has stray indentation")


def main() -> None:
    contact_name = os.environ["MDC_CONTACT_NAME"]
    contact_email = os.environ["MDC_CONTACT_EMAIL"]
    for label, value in MARKDOWN_FIELDS.items():
        _require_breaks(label, value)
    _require_breaks("longDescription", LONG_DESCRIPTION)
    if "parts 1 and 2" not in LONG_DESCRIPTION or "parts 1 and 2" not in OTHER:
        raise SystemExit("expected 'parts 1 and 2' in the card text")

    from datacollective import (
        DatasetSubmission,
        License,
        Task,
        Visibility,
        update_submission,
    )

    shared = dict(
        longDescription=LONG_DESCRIPTION,
        restrictions=RESTRICTIONS,
        forbiddenUsage=FORBIDDEN,
        additionalConditions=ADDITIONAL,
        other=OTHER,
        intendedUsage=INTENDED,
        ethicalReviewProcess=ETHICAL,
        task=Task.ML,
        locale="zxx",
        format="FLAC, MIDI, CSV",
        licenseAbbreviation=License.CC_BY_4_0,
        visibility=Visibility.PUBLIC,
        exclusivityOptOut=True,
        showContactInfo=False,
        showComplianceAttributes=False,
        pointOfContactFullName=contact_name,
        pointOfContactEmail=contact_email,
    )
    for submission_id, title in TITLES.items():
        update_submission(
            submission_id,
            DatasetSubmission(
                name=title,
                shortDescription=SHORT[submission_id],
                **shared,
            ),
        )
        print(
            f"updated {title} {submission_id} "
            f"(restrictions newlines={RESTRICTIONS.count(chr(10))})"
        )


if __name__ == "__main__":
    main()
