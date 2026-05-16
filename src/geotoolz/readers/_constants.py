"""Shared lazy loaders for packaged sensor calibration data."""

from __future__ import annotations

import csv
import json
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
        return tuple(csv.DictReader(row for row in f if not row.startswith("#")))


@cache
def load_json(package: str, resource: str) -> Any:
    """Load a packaged JSON resource and cache the parsed object."""
    with files(package).joinpath(resource).open(encoding="utf-8") as f:
        return json.load(f)
