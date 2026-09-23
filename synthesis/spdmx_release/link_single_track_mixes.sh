#!/usr/bin/env bash
# Optional: for single-track songs, create mix.flac → 0.flac (release omits the mix).
# Run from the unpacked dataset root (where songs.csv lives):
#   ./link_single_track_mixes.sh
set -euo pipefail
ROOT="${1:-.}"
CSV="$ROOT/songs.csv"
if [[ ! -f "$CSV" ]]; then
  echo "error: missing $CSV" >&2
  exit 1
fi
python3 - "$ROOT" "$CSV" <<'PY'
import csv, os, sys
root, path = sys.argv[1], sys.argv[2]
n = 0
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        multi = row.get("subset:multitrack", "")
        if multi != "":
            if multi.strip().lower() in ("true", "1"):
                continue
        elif int(float(row.get("n_tracks") or 0)) >= 2:
            continue
        song = os.path.join(root, row["path"].lstrip("./"))
        if not os.path.isdir(song):
            continue
        os.symlink("0.flac", os.path.join(song, "mix.flac"))
        n += 1
print(f"linked mix.flac → 0.flac for {n} single-track song(s)")
PY
