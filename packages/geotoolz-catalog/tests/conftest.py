"""Shared fixtures for catalog tests — synthetic raster/vector files on disk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from hypothesis import settings
from rasterio.transform import from_bounds

from geocatalog._src.memory import InMemoryGeoCatalog


# Hypothesis profiles. Two are registered:
#
# - ``dev`` (loaded by default locally): random examples, no deadline,
#   ``print_blob=True`` so failure repro snippets show up in pytest output.
# - ``ci`` (selected by ``HYPOTHESIS_PROFILE=ci`` in `.github/workflows/ci.yml`):
#   ``derandomize=True`` so the same examples run on every build,
#   making regressions reproducible.
#
# The active profile is picked from ``HYPOTHESIS_PROFILE`` and falls
# back to ``dev``; neither call into Hypothesis's built-in ``default``.
settings.register_profile("ci", settings(derandomize=True, deadline=None))
settings.register_profile(
    "dev",
    settings(deadline=None, print_blob=True),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def catalog_from_rows(
    rows: list[dict[str, Any]],
    *,
    crs: str,
) -> InMemoryGeoCatalog:
    """Build an `InMemoryGeoCatalog` from a list of row dicts.

    Each row dict must carry ``geometry``, ``start_time`` and
    ``end_time`` (popped into the catalog's IntervalIndex) plus any
    extras columns the test needs (``filepath``, ``assets``, ...).

    Args:
        rows: Row dicts, one per catalog entry.
        crs: CRS assigned to the built GeoDataFrame.

    Returns:
        A raster-backend `InMemoryGeoCatalog` over the rows.
    """
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.index = pd.IntervalIndex.from_arrays(
        gdf.pop("start_time"),
        gdf.pop("end_time"),
        closed="both",
        name="datetime",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


@pytest.fixture
def utm29_tile_factory(tmp_path: Path):
    """Build a small GeoTIFF with given bounds (UTM zone 29N) at 10 m res.

    Returns a `(bounds, date_str) -> path` callable so each test can
    populate ``tmp_path`` with the catalog files it needs.
    """

    def _make(
        bounds: tuple[float, float, float, float],
        date_str: str,
        *,
        n_bands: int = 3,
        value: int = 1,
        shape: tuple[int, int] = (32, 32),
    ) -> Path:
        xmin, ymin, xmax, ymax = bounds
        height, width = shape
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        path = tmp_path / f"S2_T29SND_{date_str}_{int(xmin)}_{int(ymin)}.tif"
        data = np.full((n_bands, height, width), value, dtype=np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_bands,
            dtype="uint16",
            crs="EPSG:32629",
            transform=transform,
        ) as dst:
            dst.write(data)
        return path

    return _make
