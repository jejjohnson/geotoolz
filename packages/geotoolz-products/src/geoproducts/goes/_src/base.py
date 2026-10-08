"""The shared reader for one ABI file: the fixed grid plus ABI metadata.

Everything generic — windowed HDF5 reads of CF-packed variables, mask-vs-
value decoding, flag tables — lives in :class:`geoproducts._src.hdf.PackedGridReader`
and the grid in :class:`geoproducts._src.geostationary.FixedGrid`; this
layer only adds what is ABI-specific: the ``goes_imager_projection`` grid
mapping and the file's global metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from affine import Affine
from rasterio.crs import CRS

from geoproducts._src.base import Track
from geoproducts._src.geostationary import FixedGrid
from geoproducts._src.hdf import (
    PackedGridReader,
    Source,
    grid_variables as _grid_variables,
)
from geoproducts.goes._src.metadata import FileInfo


__all__ = ["ABIFile", "Source", "grid_variables"]

#: The CF grid-mapping variable of every ABI file.
GRID_MAPPING = "goes_imager_projection"


class ABIFile(PackedGridReader):
    """One ABI product file read as a ``(bands, H, W)`` ``GeoTensor``.

    Each band is one 2-D variable of the file, all on the file's fixed
    grid. Subclasses choose the variables and, where needed, how raw
    values decode (``_decode``).
    """

    extra = "goes"
    feature = "geoproducts.goes"

    _grid: FixedGrid
    _info: FileInfo

    def _init_file(self, source: Source, variables: Sequence[str]) -> None:
        self._source = source
        with self._open() as f:
            self._grid = FixedGrid.from_cf(f, GRID_MAPPING)
            self._info = FileInfo.from_file(f)
            self._load_variables(f, variables)

    def _share(self, other: ABIFile, variables: Sequence[str]) -> None:
        """Point at ``other``'s file, reusing its parsed grid and metadata."""
        self._source = other._source
        self._grid = other._grid
        self._info = other._info
        with self._open() as f:
            self._load_variables(f, variables)

    def _describe(self) -> str:
        return self._info.dataset_name or super()._describe()

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

    # -- file metadata ----------------------------------------------------

    @property
    def dataset_name(self) -> str:
        """The file's ``dataset_name`` (its NOAA file name)."""
        return self._info.dataset_name

    @property
    def platform(self) -> str:
        """Platform ID (``"G16"`` … ``"G19"``)."""
        return self._info.platform

    @property
    def scene(self) -> str:
        """Scan sector: ``"Full Disk"``, ``"CONUS"`` or ``"Mesoscale"``."""
        return self._info.scene

    @property
    def start_time(self) -> datetime | None:
        """Scan start time (UTC)."""
        return self._info.start_time

    @property
    def end_time(self) -> datetime | None:
        """Scan end time (UTC)."""
        return self._info.end_time

    @property
    def satellite_lon_deg(self) -> float:
        """Nominal sub-satellite longitude (degrees east)."""
        return self._info.satellite_lon_deg

    @property
    def satellite_height_m(self) -> float:
        """Nominal satellite height above the equator (metres)."""
        return self._info.satellite_height_m


def grid_variables(f: Any) -> tuple[str, ...]:
    """Names of an open ABI file's 2-D variables on its fixed grid."""
    return _grid_variables(f, (len(f["y"]), len(f["x"])))
