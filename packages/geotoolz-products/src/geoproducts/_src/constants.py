"""Shared lazy loaders for packaged sensor calibration data."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from typing import Any


CsvRows = tuple[dict[str, str], ...]


@cache
def load_csv(package: str, resource: str) -> CsvRows:
    """Load a packaged CSV resource and cache the parsed rows.

    Lines starting with ``#`` are treated as citation/license headers and
    ignored by the CSV parser.
    """
    with files(package).joinpath(resource).open(encoding="utf-8") as f:
        filtered = [row for row in f if not row.startswith("#")]
        return tuple(csv.DictReader(filtered))


@cache
def load_json(package: str, resource: str) -> Any:
    """Load a packaged JSON resource and cache the parsed object."""
    with files(package).joinpath(resource).open(encoding="utf-8") as f:
        return json.load(f)


def band_values(
    rows: CsvRows, bands: Sequence[str], column: str, *, key: str = "name"
) -> tuple[float, ...]:
    """One number per band, in ``bands`` order, from a packaged band table.

    For readers whose band metadata lives in a ``BANDS`` table (centre
    wavelengths, FWHM) rather than in the file they read.

    Args:
        rows: The table (``load_csv`` rows), one per band.
        bands: Band names, in the order the reads hold them.
        column: The column to read, e.g. ``"wavelength_nm"``.
        key: The column holding the band name.

    Raises:
        KeyError: A band is not in the table, or the table has no ``column``.

    Example:
        >>> rows = ({"name": "red", "wavelength_nm": "645"},
        ...         {"name": "nir", "wavelength_nm": "858"})
        >>> band_values(rows, ("nir", "red"), "wavelength_nm")
        (858.0, 645.0)
    """
    table = {row[key]: row for row in rows}
    return tuple(float(table[band][column]) for band in bands)
