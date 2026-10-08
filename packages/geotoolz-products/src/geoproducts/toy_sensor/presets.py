"""Zero-argument toy-sensor-aware operator presets.

Presets bind geotoolz operators to this sensor's band names. geotoolz is
an optional dependency (``pip install 'geotoolz-products[operators]'``),
imported when a preset is called, so the reader itself installs without
the operator library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geoproducts._src.extras import missing_extra
from geoproducts.toy_sensor import constants


if TYPE_CHECKING:
    from pipekit import Operator


def NDVI() -> Operator:
    """Return a toy-sensor NDVI operator configured with the reference band names.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    try:
        from geotoolz.indices import NDVI as _NDVI
    except ImportError as exc:
        raise missing_extra("toy_sensor.presets.NDVI", "operators") from exc
    return _NDVI(red=constants.BAND_RED, nir=constants.BAND_NIR)


__all__ = ["NDVI"]
