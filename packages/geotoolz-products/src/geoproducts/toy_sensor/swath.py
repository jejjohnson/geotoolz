"""A toy sensor-geometry reader: the toy bands on a swath (Track B).

Reads are a `geoarray.GeoSwath`: data on ``(band, row, col)`` with a 2-D
latitude / longitude per pixel, the shape of a level-1 swath before it is
gridded. Downstream code and tests use it as the stack's swath fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from geoarray import GeoSwath, PixelWindow
from geoarray.coords import SpectralAxis

from geoproducts._src.base import resolve_fill_value
from geoproducts._src.carrier import reject_georeader
from geoproducts._src.constants import band_values
from geoproducts.toy_sensor import constants


__all__ = ["SwathReader"]

_BANDS = (
    constants.BAND_BLUE,
    constants.BAND_GREEN,
    constants.BAND_RED,
    constants.BAND_NIR,
)


class SwathReader:
    """The toy sensor's four bands in sensor geometry, read as a `GeoSwath`.

    Args:
        path: Scene path recorded for auditability.
        data: ``(band, row, col)`` values, 4 bands (blue, green, red, nir).
        lat: ``(row, col)`` pixel-centre latitudes, degrees.
        lon: ``(row, col)`` pixel-centre longitudes, degrees.
        fill_value: Nodata marker; ``None`` picks ``NaN`` for float data and
            ``0`` for integer / boolean data.

    Raises:
        TypeError: ``data`` is a georeader ``GeoTensor``.
        ValueError: ``data`` is not ``(4, row, col)``, or ``lat`` / ``lon``
            are not ``(row, col)``.

    Example:
        >>> import numpy as np
        >>> rows, cols = np.linspace(40, 41, 3), np.linspace(-1, 0, 5)
        >>> lat, lon = np.meshgrid(rows, cols, indexing="ij")
        >>> reader = SwathReader("granule", data=np.zeros((4, 3, 5)), lat=lat, lon=lon)
        >>> swath = reader.load()
        >>> swath.dims, swath.lat.shape
        (('band', 'row', 'col'), (3, 5))
    """

    dims = ("band", "row", "col")

    def __init__(
        self,
        path: str | Path,
        *,
        data: Any,
        lat: Any,
        lon: Any,
        fill_value: float | None = None,
    ) -> None:
        reject_georeader(data, "toy_sensor.SwathReader(data=...)")
        self.path = Path(path)
        self._data = np.asarray(data)
        self._lat = np.asarray(lat, dtype=np.float64)
        self._lon = np.asarray(lon, dtype=np.float64)
        if self._data.ndim != 3 or self._data.shape[0] != len(_BANDS):
            raise ValueError(
                "toy_sensor.SwathReader expects data (4, row, col) "
                f"(blue, green, red, nir); got shape {self._data.shape}."
            )
        if (
            self._lat.shape != self._data.shape[1:]
            or self._lon.shape != self._lat.shape
        ):
            raise ValueError(
                f"lat {self._lat.shape} and lon {self._lon.shape} must be "
                f"(row, col) = {self._data.shape[1:]}."
            )
        self._fill = resolve_fill_value(
            fill_value, self._data.dtype, owner="toy_sensor.SwathReader"
        )

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names in array order."""
        return _BANDS

    @property
    def shape(self) -> tuple[int, ...]:
        """``(band, row, col)``."""
        return tuple(self._data.shape)

    @property
    def coords(self) -> tuple[SpectralAxis, ...]:
        """The band names and centre wavelengths (nm)."""
        wavelengths = band_values(constants.BANDS, _BANDS, "wavelength_nm")
        return (SpectralAxis(_BANDS, wavelengths),)

    def load(self) -> GeoSwath:
        """The whole granule."""
        return self._swath(slice(None), slice(None))

    def read_window(self, window: PixelWindow) -> GeoSwath:
        """The rows and columns of ``window``, clipped to the granule.

        Raises:
            ValueError: ``window`` misses the granule.
        """
        rows, cols = self._data.shape[1:]
        whole = PixelWindow(col_off=0, row_off=0, width=cols, height=rows)
        inside = window.intersection(whole)
        if inside is None:
            raise ValueError(f"window {window} misses the granule {(rows, cols)}")
        return self._swath(*inside.to_slices())

    def _swath(self, rows: slice, cols: slice) -> GeoSwath:
        return GeoSwath(
            self._data[:, rows, cols],
            self._lat[rows, cols],
            self._lon[rows, cols],
            dims=self.dims,
            coords=self.coords,
            fill_value=self._fill,
            attrs={"path": str(self.path)},
        )
