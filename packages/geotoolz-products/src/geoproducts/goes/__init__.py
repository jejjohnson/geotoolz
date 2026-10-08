"""GOES-R ABI (GOES-16 … GOES-19) L1b radiance reader.

- ``Reader`` — one ABI L1b file (one channel, one scan) on its
  ``+proj=geos`` fixed grid, calibrated to counts, radiance, reflectance
  or brightness temperature with the file's own coefficients; needs the
  ``[goes]`` extra (``h5py``).
- ``QualityReader`` — the file's ``DQF`` quality flags (``Reader.quality``).
- ``aws`` — list and download files from NOAA's public buckets
  (standard library only, no credentials).
- ``BANDS`` / ``constants`` — the 16-channel table, ``DQF`` meanings and
  satellite positions; ``presets`` — geotoolz operators bound to ABI
  channel names (``[operators]`` extra).
"""

from __future__ import annotations

from typing import Any

from geoproducts.goes import aws, constants, presets
from geoproducts.goes.presets import NDVI, ParallaxCorrect, SyntheticGreen
from geoproducts.goes.reader import Calibration, QualityReader, Reader


def __getattr__(name: str) -> Any:
    if name == "BANDS":
        return constants.BANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BANDS",
    "NDVI",
    "Calibration",
    "ParallaxCorrect",
    "QualityReader",
    "Reader",
    "SyntheticGreen",
    "aws",
    "constants",
    "presets",
]
