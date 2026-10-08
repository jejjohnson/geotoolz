"""NOAA AHI L2 cloud products: cloud mask, cloud height, cloud phase.

NOAA's enterprise algorithms run on Himawari full-disk scans and publish
NetCDF-4 files (``AHI-<code>_v1r1_h09_s<start>_e<end>_c<created>.nc``;
``CMSK`` cloud mask, ``CHGT`` cloud-top height, ``CPHS`` cloud phase). The
variables sit on the AHI 2 km full-disk fixed grid, but the files carry
per-pixel latitude / longitude rather than a usable grid mapping, so the
reader places them on the nominal grid (identical, pixel for pixel, to the
L1b 2 km full disk).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from affine import Affine
from rasterio.crs import CRS

from geoproducts._src.base import Track
from geoproducts._src.geostationary import FixedGrid
from geoproducts._src.hdf import (
    PackedGridReader,
    Source,
    attr,
    grid_variables,
    time_attr,
)
from geoproducts.himawari import constants


__all__ = ["L2Reader", "full_disk_grid", "product_code"]

_PRODUCT_RE = re.compile(r"AHI-(?P<code>[A-Z0-9]+)_v\d+r\d+_h\d{2}_s\d+")
# Geolocation and the duplicated AWIPS copies are not data bands.
_NOT_DATA = ("Latitude", "Longitude")


def product_code(name: str) -> str:
    """The product code of an AHI L2 file name: ``"CMSK"``, ``"CHGT"``, ….

    Returns ``""`` when the name is not a NOAA AHI L2 file name.
    """
    match = _PRODUCT_RE.search(name)
    return match["code"] if match else ""


def full_disk_grid(resolution_km: float = 2.0) -> FixedGrid:
    """The nominal AHI full-disk fixed grid at ``resolution_km``.

    Args:
        resolution_km: ``0.5``, ``1.0`` or ``2.0``.

    Raises:
        KeyError: Not an AHI resolution.

    Examples:
        >>> grid = full_disk_grid(2.0)
        >>> grid.height, grid.width, round(grid.transform.a)
        (5500, 5500, 2000)
    """
    size, factor, offset = constants.FULL_DISK_GRIDS[resolution_km]
    return FixedGrid.from_cgms(
        columns=size,
        lines=size,
        cfac=factor,
        lfac=factor,
        coff=offset,
        loff=offset,
        lon_0=constants.HIMAWARI_LON_DEG,
        height_m=constants.SATELLITE_HEIGHT_M,
        semi_major_m=constants.EARTH_SEMI_MAJOR_M,
        semi_minor_m=constants.EARTH_SEMI_MINOR_M,
    )


class L2Reader(PackedGridReader):
    """NOAA AHI L2 full-disk cloud product reader: chosen variables as bands.

    Each band is one 2-D variable of the file on the 2 km full disk.
    Integer masks (``CloudMask``, ``CloudMaskBinary``, ``Dust_Mask``, …)
    stay ``int8`` with the file's fill (``-128``); anything floating
    (cloud probability, cloud-top height) is ``float32`` with ``NaN``.

    Args:
        source: Path to an L2 ``.nc`` file, or a binary file-like object.
        variables: Variable name(s) to read. Default: ``CloudMaskBinary`` +
            ``CloudMask`` for ``CMSK``, otherwise every data variable on
            the grid. ``available_variables`` lists them all.

    Raises:
        ImportError: ``h5py`` is not installed (the ``[himawari]`` extra).
        ValueError: A variable is missing or not on the 2 km full disk.

    Examples:
        The binary and four-level cloud masks of one full-disk scan::

            mask = himawari.L2Reader(cmsk_path)    # ('CloudMaskBinary', 'CloudMask')
            mask.shape                             # (2, 5500, 5500) int8
            mask.flags("CloudMask")  # {0: 'clear', 1: 'probably_clear', ...}
    """

    extra = "himawari"
    feature = "geoproducts.himawari.L2Reader"

    def __init__(
        self, source: Source, *, variables: str | Sequence[str] | None = None
    ) -> None:
        self._source = source
        self._grid = full_disk_grid(2.0)
        with self._open() as f:
            self._product = product_code(str(attr(f, "Metadata_Link", ""))) or str(
                attr(f, "title", "")
            ).removeprefix("AHI-")
            self._satellite = str(attr(f, "satellite_name", ""))
            self._start = time_attr(f, "time_coverage_start")
            self._end = time_attr(f, "time_coverage_end")
            self._available = tuple(
                n
                for n in grid_variables(f, self._grid_shape)
                if n not in _NOT_DATA and not n.endswith("AWIPS")
            )
            self._load_variables(f, self._resolve(variables))

    def _resolve(self, variables: str | Sequence[str] | None) -> list[str]:
        if isinstance(variables, str):
            return [variables]
        if variables is not None:
            names = list(variables)
            if not names:
                raise ValueError("variables must name at least one variable.")
            return names
        default = constants.L2_DEFAULT_VARIABLES.get(self._product)
        names = list(default) if default is not None else list(self._available)
        if not names:
            raise ValueError(f"{self._describe()} has no data variables on its grid.")
        return names

    def __repr__(self) -> str:
        return (
            f"himawari.L2Reader({self._describe()!r}, product={self.product!r}, "
            f"variables={self.variables}, shape={self.shape})"
        )

    # -- georeferencing ---------------------------------------------------

    @property
    def _grid_shape(self) -> tuple[int, int]:
        return (self._grid.height, self._grid.width)

    @property
    def _crs(self) -> CRS:
        return self._grid.crs

    @property
    def _transform(self) -> Affine:
        return self._grid.transform

    @property
    def _track(self) -> Track:
        return "A"

    @property
    def _bands(self) -> tuple[str, ...]:
        return self.variables

    # -- file metadata ----------------------------------------------------

    @property
    def product(self) -> str:
        """Product code (``"CMSK"``, ``"CHGT"``, ``"CPHS"``)."""
        return self._product

    @property
    def satellite(self) -> str:
        """``"Himawari-9"`` / ``"Himawari-8"``."""
        return self._satellite

    @property
    def available_variables(self) -> tuple[str, ...]:
        """Every data variable of the file on the grid."""
        return self._available

    @property
    def start_time(self) -> datetime | None:
        """Scan start (UTC)."""
        return self._start

    @property
    def end_time(self) -> datetime | None:
        """Scan end (UTC)."""
        return self._end

    @property
    def satellite_lon_deg(self) -> float:
        """Sub-satellite longitude of the grid (degrees east)."""
        return self._grid.lon_0
