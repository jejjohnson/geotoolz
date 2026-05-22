"""Zero-argument toy-sensor-aware operator presets."""

from __future__ import annotations

from pipekit import Operator
from geotoolz.indices import NDVI as _NDVI
from geotoolz.readers.toy_sensor import constants


def NDVI() -> Operator:
    """Return a toy-sensor NDVI operator configured with the reference band names."""
    return _NDVI(red=constants.BAND_RED, nir=constants.BAND_NIR)


__all__ = ["NDVI"]
