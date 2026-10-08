"""GOES-R ABI (GOES-16 … GOES-19) readers, bucket helpers and recipes.

- ``Reader`` — one L1b radiance file (one channel, one scan) on its
  ``+proj=geos`` fixed grid, calibrated to counts, radiance, reflectance
  or brightness temperature with the file's own coefficients.
- ``L2Reader`` — any L2 product (clear sky mask, cloud-top height, LST,
  fire, stability indices, MCMIP with all 16 channels, …) on the same grid.
- ``QualityReader`` — the ``DQF`` quality flags (``reader.quality``).
- ``aws`` — list and download files from NOAA's public buckets
  (unsigned, through ``geocloud.files``; geotoolz-cloud ships with the
  extra).
- ``recipes`` — the standard ABI RGB recipes (true colour, natural
  colour, day cloud phase, fire temperature) as data.
- ``BANDS`` / ``constants`` — the 16-channel table, flag codes and
  satellite positions; ``presets`` — geotoolz operators bound to ABI
  channel names and recipes (``[operators]`` extra).

The readers need the ``[goes]`` extra (``h5py``).
"""

from __future__ import annotations

from typing import Any

from geoproducts.goes import aws, constants, presets, recipes
from geoproducts.goes.l1b import Calibration, QualityReader, Reader
from geoproducts.goes.l2 import L2Reader
from geoproducts.goes.presets import (
    NDVI,
    DayCloudPhase,
    FireTemperature,
    MaskClouds,
    NaturalColor,
    ParallaxCorrect,
    SyntheticGreen,
    TrueColor,
)


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
    "L2Reader",
    "MaskClouds",
    "NaturalColor",
    "ParallaxCorrect",
    "QualityReader",
    "Reader",
    "SyntheticGreen",
    "TrueColor",
    "aws",
    "constants",
    "presets",
    "recipes",
]
