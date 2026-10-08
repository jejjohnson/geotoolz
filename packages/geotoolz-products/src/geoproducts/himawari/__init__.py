"""Himawari-8 / -9 AHI readers, bucket helpers and recipes.

- ``Reader`` — one band of one scan from its Himawari Standard Data (HSD)
  segments (``.DAT`` or ``.DAT.bz2``; full disk, Japan or target area) on
  its ``+proj=geos`` fixed grid, calibrated to counts, radiance,
  reflectance or brightness temperature with the files' own coefficients.
  Standard library + numpy: no extra needed.
- ``L2Reader`` — NOAA's L2 full-disk cloud products (cloud mask, cloud-top
  height, cloud phase) on the same 2 km grid (``[himawari]`` extra, h5py).
- ``HSDHeader`` / ``read_header`` — the decoded segment header.
- ``aws`` — list and download files from NOAA's public buckets
  (standard library only, no credentials).
- ``recipes`` — the standard RGB recipes (true colour, natural colour,
  day cloud phase, fire temperature) on AHI bands, as data.
- ``BANDS`` / ``constants`` — the 16-band table, fixed grids, cloud-mask
  codes and the 140.7°E slot; ``presets`` — geotoolz operators bound to
  AHI band names and recipes (``[operators]`` extra).
"""

from __future__ import annotations

from typing import Any

from geoproducts.himawari import aws, constants, presets, recipes
from geoproducts.himawari._src.hsd import HSDHeader, read_header
from geoproducts.himawari.l2 import L2Reader
from geoproducts.himawari.presets import (
    NDVI,
    DayCloudPhase,
    FireTemperature,
    HybridGreen,
    MaskClouds,
    NaturalColor,
    ParallaxCorrect,
    TrueColor,
)
from geoproducts.himawari.reader import Calibration, Reader


def __getattr__(name: str) -> Any:
    if name == "BANDS":
        return constants.BANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BANDS",
    "NDVI",
    "Calibration",
    "DayCloudPhase",
    "FireTemperature",
    "HSDHeader",
    "HybridGreen",
    "L2Reader",
    "MaskClouds",
    "NaturalColor",
    "ParallaxCorrect",
    "Reader",
    "TrueColor",
    "aws",
    "constants",
    "presets",
    "read_header",
    "recipes",
]
