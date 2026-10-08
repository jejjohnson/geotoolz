"""GOES-R ABI-aware geotoolz operator presets.

Presets bind geotoolz operators to ABI channel names (``"C01"`` …
``"C16"``, the ``band_names`` every :class:`geoproducts.goes.Reader`
output carries). geotoolz is an optional dependency
(``pip install 'geotoolz-products[operators]'``), imported when a preset
is called, so the reader itself installs without the operator library.

The multi-channel presets expect the channels already stacked on one grid
(e.g. ``gz.spectral.StackBands()`` after resampling C02's 0.5 km grid to
the 1 km C01 / C03 grid): ABI ships one channel per file, each at its own
native resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geoproducts._src.extras import missing_extra
from geoproducts.goes import constants


if TYPE_CHECKING:
    from pipekit import Operator


def NDVI() -> Operator:
    """NDVI from ABI reflectance: ``red="C02"``, ``nir="C03"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    try:
        from geotoolz.indices import NDVI as _NDVI
    except ImportError as exc:
        raise missing_extra("goes.presets.NDVI", "operators") from exc
    return _NDVI(red="C02", nir="C03")


def SyntheticGreen() -> Operator:
    """The CIMSS synthetic green band, ``0.45 C02 + 0.10 C03 + 0.45 C01``.

    ABI has no green channel; this linear mix of red, "veggie" near-IR and
    blue reflectance stands in for it in true-colour imagery. Output is one
    band, ``(1, H, W)``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    try:
        from geotoolz.spectral import BandMath
    except ImportError as exc:
        raise missing_extra("goes.presets.SyntheticGreen", "operators") from exc
    expression = " + ".join(
        f"{weight} * {channel}"
        for channel, weight in constants.SYNTHETIC_GREEN_WEIGHTS.items()
    )
    return BandMath(expression=expression)


def ParallaxCorrect(
    *,
    satellite_lon_deg: float = constants.GOES_EAST_LON_DEG,
    target_height_m: float = 0.0,
    method: str = "bilinear",
) -> Operator:
    """Geostationary parallax correction from a GOES-R viewpoint.

    Binds ``gz.geom.GeostationaryParallaxCorrect`` to the GOES geostationary
    height; the input must already be on an EPSG:4326 grid. Pass
    ``reader.satellite_lon_deg`` (or ``constants.GOES_WEST_LON_DEG``) for a
    satellite other than GOES-East.

    Args:
        satellite_lon_deg: Sub-satellite longitude. Default GOES-East.
        target_height_m: Height of the observed feature (e.g. cloud top).
        method: ``"nearest"`` or ``"bilinear"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    try:
        from geotoolz.geom import GeostationaryParallaxCorrect
    except ImportError as exc:
        raise missing_extra("goes.presets.ParallaxCorrect", "operators") from exc
    return GeostationaryParallaxCorrect(
        satellite_lon_deg=satellite_lon_deg,
        satellite_height_m=constants.SATELLITE_HEIGHT_M,
        target_height_m=target_height_m,
        method=method,
    )


__all__ = ["NDVI", "ParallaxCorrect", "SyntheticGreen"]
