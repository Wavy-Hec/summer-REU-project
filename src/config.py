"""Locate the FRA Highway-Rail Grade Crossing Accident dataset.

Every script in this repository reads the same CSV. Rather than hardcoding a
path in each one, they all call :func:`dataset_path`, which resolves the file
in this order:

1. ``$REU_DATA_PATH`` if it is set (point it anywhere you like).
2. ``data/Highway-Rail_Grade_Crossing_Accident_Data.csv`` in the repo root.

The CSV is not committed -- it is a large public dataset. See ``data/README.md``
for the download link.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "Highway-Rail_Grade_Crossing_Accident_Data.csv"


def dataset_path() -> Path:
    """Return the path to the accident CSV, or explain how to get it."""
    override = os.environ.get("REU_DATA_PATH")
    path = Path(override).expanduser() if override else DEFAULT_CSV

    if not path.is_file():
        raise FileNotFoundError(
            f"Accident dataset not found at: {path}\n\n"
            "Download 'Highway-Rail Grade Crossing Accident Data' (FRA Form 57) from\n"
            "  https://data.transportation.gov/Railroads/"
            "Highway-Rail-Grade-Crossing-Accident-Data/7wn6-i5b9\n"
            f"and save it to {DEFAULT_CSV}, or set REU_DATA_PATH to point at it.\n"
            "See data/README.md for details."
        )
    return path
