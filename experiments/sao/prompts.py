"""Fixed prompt set for SAO generative evaluation."""

from __future__ import annotations

DEFAULT_PROMPTS: list[str] = [
    "instrumental music, piano drums bass guitar",
    "instrumental jazz trio, piano bass drums",
    "upbeat rock instrumental, electric guitar bass drums",
    "classical piano and strings, no vocals",
    "lofi hip hop instrumental, soft drums bass piano",
    "funk groove, slap bass drums electric guitar",
    "ambient instrumental pads and soft piano",
    "metal instrumental, distorted guitars drums bass",
    "folk acoustic guitar and soft piano",
    "orchestral brass and strings fanfare",
    "electronic dance instrumental, synth bass drums",
    "blues shuffle, electric guitar bass drums piano",
    "bossa nova, nylon guitar soft drums bass",
    "cinematic trailer instrumental, drums brass",
    "reggae instrumental, guitar bass drums",
    "country instrumental, acoustic guitar bass drums",
    "soul instrumental, electric piano bass drums",
    "punk rock instrumental, guitar bass drums",
    "synthwave instrumental, analog synths drums bass",
    "chamber music, piano violin cello",
    "marching band percussion and brass",
    "gospel instrumental, organ piano drums bass",
    "latin percussion, bass piano guitar",
    "indie rock instrumental, clean guitar bass drums",
    "trap instrumental, 808 bass hi-hats piano",
    "waltz piano and strings",
    "heavy drums and distorted bass riff",
    "soft ballad instrumental, piano acoustic guitar",
    "progressive rock instrumental, complex drums guitar bass",
    "minimal techno instrumental, kick bass synth",
    "bluegrass, banjo acoustic guitar bass",
    "ska instrumental, guitar bass drums brass",
    "r&b instrumental, electric piano bass drums",
    "baroque style, harpsichord and strings",
    "film noir jazz, muted trumpet piano bass drums",
    "power pop instrumental, bright guitar bass drums",
    "dub instrumental, bass drums guitar echoes",
    "new age instrumental, soft pads piano",
    "afrobeat instrumental, percussion guitar bass",
    "polka, accordion drums bass",
    "surf rock instrumental, reverb guitar bass drums",
    "chiptune style instrumental, square leads drums bass",
    "doom metal instrumental, slow drums bass guitar",
    "swing big band instrumental, brass rhythm section",
    "acoustic singer-songwriter backing, guitar piano (no vocals)",
    "house music instrumental, four-on-floor drums bass",
    "math rock instrumental, odd-meter drums guitar bass",
    "celtic instrumental, fiddle acoustic guitar",
    "smooth jazz, saxophone soft drums bass electric piano",
    "epic orchestral percussion and low brass",
]


def select_prompts(n: int) -> list[str]:
    if n <= len(DEFAULT_PROMPTS):
        return DEFAULT_PROMPTS[:n]
    # Cycle if more requested.
    out = []
    i = 0
    while len(out) < n:
        out.append(DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)])
        i += 1
    return out
