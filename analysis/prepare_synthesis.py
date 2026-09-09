"""Step-0 synthesis setup: GM register + dense corrected MIDI copies.

Preferred entrypoint (broader than ``analyze_gm_register``)::

    uv run python -m analysis.prepare_synthesis --subset all_valid -j 8

Writes:
  - ``{OUTPUT_DIR}/dev/analysis/instruments/<subset>/register.csv``
  - ``{OUTPUT_DIR}/SPDMX_dev/``: ``mid/``, ``stems.csv``, ``LICENSE``, ``README.md`` (disable MIDI with ``--no-write-corrected-midi``)

``analyze_gm_register`` remains a thin alias for the same CLI.
"""

from __future__ import annotations

from analysis.analyze_gm_register import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
