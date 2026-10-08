"""Himawari AHI band table, fixed grids, cloud-mask codes and orbit constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoproducts._src.constants import load_csv


# AHI band names, as they appear in the HSD file names (``..._B13_FLDK_...``).
CHANNELS: tuple[str, ...] = tuple(f"B{n:02d}" for n in range(1, 17))
REFLECTIVE_CHANNELS: tuple[str, ...] = CHANNELS[:6]
EMISSIVE_CHANNELS: tuple[str, ...] = CHANNELS[6:]

# Himawari-8 and -9 share the 140.7°E slot. Each HSD file carries its own
# projection (``Reader.satellite_lon_deg``); prefer it when a file is at hand.
HIMAWARI_LON_DEG = 140.7
# Satellite height above the equator: 42164 km from the Earth's centre
# minus the 6378.137 km equatorial radius (HSD block 3).
SATELLITE_HEIGHT_M = 35_785_863.0
EARTH_SEMI_MAJOR_M = 6_378_137.0
EARTH_SEMI_MINOR_M = 6_356_752.3

# Full-disk fixed grids by resolution (km): ``(size, CFAC = LFAC,
# COFF = LOFF)``. The NOAA L2 products sit on the 2 km grid.
FULL_DISK_GRIDS: dict[float, tuple[int, int, float]] = {
    0.5: (22_000, 81_865_099, 11_000.5),
    1.0: (11_000, 40_932_549, 5_500.5),
    2.0: (5_500, 20_466_275, 2_750.5),
}

# NOAA enterprise cloud mask (``AHI-CMSK``): ``CloudMask`` (four-level) and
# ``CloudMaskBinary`` codes.
CLOUD_MASK_CLEAR = 0
CLOUD_MASK_PROBABLY_CLEAR = 1
CLOUD_MASK_PROBABLY_CLOUDY = 2
CLOUD_MASK_CLOUDY = 3
CLOUD_MASK_BINARY_CLEAR = 0
CLOUD_MASK_BINARY_CLOUDY = 1

# Default variables of the L2 products, by product code; every other product
# defaults to all its data variables on the grid.
L2_DEFAULT_VARIABLES: dict[str, tuple[str, ...]] = {
    "CMSK": ("CloudMaskBinary", "CloudMask"),
}

# Hybrid green for true colour: AHI's 0.51 µm band sits below the
# vegetation reflectance peak, so a small share of the 0.86 µm band is
# mixed in (Miller et al., 2016).
HYBRID_GREEN_NIR_FRACTION = 0.07

_CACHE: dict[str, Any] = {}

if TYPE_CHECKING:
    BANDS: tuple[dict[str, str], ...]


def __getattr__(name: str) -> Any:
    if name == "BANDS":
        if name not in _CACHE:
            _CACHE[name] = load_csv(__package__, "data/bands.csv")
        return _CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BANDS",
    "CHANNELS",
    "CLOUD_MASK_BINARY_CLEAR",
    "CLOUD_MASK_BINARY_CLOUDY",
    "CLOUD_MASK_CLEAR",
    "CLOUD_MASK_CLOUDY",
    "CLOUD_MASK_PROBABLY_CLEAR",
    "CLOUD_MASK_PROBABLY_CLOUDY",
    "EARTH_SEMI_MAJOR_M",
    "EARTH_SEMI_MINOR_M",
    "EMISSIVE_CHANNELS",
    "FULL_DISK_GRIDS",
    "HIMAWARI_LON_DEG",
    "HYBRID_GREEN_NIR_FRACTION",
    "L2_DEFAULT_VARIABLES",
    "REFLECTIVE_CHANNELS",
    "SATELLITE_HEIGHT_M",
]
