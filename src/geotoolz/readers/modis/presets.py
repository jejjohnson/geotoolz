"""Zero-argument MODIS-aware operator presets."""

from __future__ import annotations

from pipekit import Operator
from geotoolz.indices import NDVI as _NDVI
from geotoolz.readers.modis import constants


def NDVI() -> Operator:
    """Return a MODIS NDVI operator configured with MODIS band names."""
    return _NDVI(red=constants.BAND_RED, nir=constants.BAND_NIR)


__all__ = ["NDVI"]
