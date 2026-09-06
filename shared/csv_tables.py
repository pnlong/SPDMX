"""Helpers for writing PDMX metadata CSV tables."""

from __future__ import annotations

import csv
import fcntl
from contextlib import contextmanager
from os.path import exists
from pathlib import Path

import pandas as pd

from shared.config import NA_STRING


def sanitize_track_name(name: str | None) -> str | None:
    """Remove characters that break CSV export (e.g. null bytes in PDMX MIDI track names)."""
    if name is None:
        return None
    cleaned = name.replace("\x00", "").replace(",", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or None


@contextmanager
def _csv_exclusive_lock(csv_path: str):
    """Serialize read-modify-write so Fluidsynth and DDSP can upsert in parallel."""
    lock_path = Path(str(csv_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def read_csv_flexible(
    csv_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV, tolerating schema drift (extra/missing trailing fields).

    When writers append a new column (e.g. ``reason``) without rewriting the
    header, pandas raises ``ParserError``. This reader extends the header with
    any missing ``columns``, then pads/truncates each data row to match.
    """
    path = Path(csv_path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns) if columns else [])
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=list(columns) if columns else [])
        header = [h.strip() for h in header]
        if not header or not any(header):
            return pd.DataFrame(columns=list(columns) if columns else [])
        if columns:
            for col in columns:
                if col not in header:
                    header.append(col)
        rows: list[dict] = []
        width = len(header)
        for raw in reader:
            if not raw or all(not c for c in raw):
                continue
            if len(raw) < width:
                raw = raw + [""] * (width - len(raw))
            elif len(raw) > width:
                raw = raw[:width]
            rows.append({header[i]: raw[i] for i in range(width)})
    df = pd.DataFrame(rows, columns=header)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[list(columns)]
    return df


def _csv_cell(col: str, row: dict) -> object:
    """Serialize one cell; bool stem flags default to False when missing."""
    value = row.get(col)
    if col == "zero_duration_notes":
        if value is None or value == "" or (
            isinstance(value, float) and value != value
        ):
            return False
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("", "nan", "none", "na", "<na>"):
                return False
            if text in ("true", "1", "yes"):
                return True
            if text in ("false", "0", "no"):
                return False
        return bool(value)
    if value is None or value == "":
        return NA_STRING
    return value


def _rewrite_csv_with_columns(
    path: Path,
    columns: list[str],
    existing_rows: list[dict],
    new_rows: list[dict],
) -> None:
    """Atomically rewrite ``path`` with a unified header and all rows."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in existing_rows + new_rows:
            writer.writerow({col: _csv_cell(col, row) for col in columns})
    tmp.replace(path)


def append_rows(csv_path: str, columns: list[str], new_rows: list[dict]) -> None:
    """Append rows without reading the existing file (O(rows added)).

    Creates a header when the file is missing or empty. If the on-disk header
    does not match ``columns`` (schema drift), rewrites the file once so old
    and new rows share one header. Duplicate keys are allowed; callers that
    need a unique table should merge/dedup later.
    """
    if not new_rows:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_exclusive_lock(str(path)):
        new_file = not path.is_file() or path.stat().st_size == 0
        if not new_file:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
            if list(header) != list(columns):
                existing = read_csv_flexible(path, columns=columns)
                _rewrite_csv_with_columns(
                    path,
                    columns,
                    existing.to_dict("records"),
                    new_rows,
                )
                return
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            if new_file:
                writer.writeheader()
            for row in new_rows:
                writer.writerow({col: _csv_cell(col, row) for col in columns})


def append_rows_deduped(
    csv_path: str,
    columns: list[str],
    new_rows: list[dict],
    *,
    key_col: str = "path",
    key_cols: list[str] | None = None,
) -> None:
    """Append rows to a CSV, replacing any existing rows with the same key value(s)."""
    if not new_rows:
        return

    cols = list(key_cols) if key_cols else [key_col]
    with _csv_exclusive_lock(csv_path):
        new_df = pd.DataFrame(new_rows, columns=columns)
        if exists(csv_path) and Path(csv_path).stat().st_size > 0:
            existing = read_csv_flexible(csv_path, columns=columns)
            if len(existing) > 0:
                new_keys = _row_key_set(new_df, cols)
                keep = [
                    _row_key(row, cols) not in new_keys
                    for row in existing.to_dict("records")
                ]
                existing = existing[keep]
            if len(existing) > 0:
                # Avoid pd.concat: all-NA columns (e.g. unnamed stems) warn on pandas 2.2+.
                new_df = pd.DataFrame(
                    existing.to_dict("records") + new_df.to_dict("records"),
                    columns=columns,
                )

        if "zero_duration_notes" in new_df.columns:
            new_df["zero_duration_notes"] = [
                _csv_cell("zero_duration_notes", {"zero_duration_notes": v})
                for v in new_df["zero_duration_notes"].tolist()
            ]

        new_df.to_csv(
            csv_path,
            sep=",",
            na_rep=NA_STRING,
            header=True,
            index=False,
            mode="w",
        )


def _row_key(row: dict, cols: list[str]) -> tuple:
    values = []
    for col in cols:
        value = row[col]
        if col == "track":
            values.append(int(value))
        else:
            values.append(str(value))
    return tuple(values)


def _row_key_set(df: pd.DataFrame, cols: list[str]) -> set[tuple]:
    return {_row_key(row, cols) for row in df.to_dict("records")}
