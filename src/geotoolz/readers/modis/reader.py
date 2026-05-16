"""Minimal MODIS reader skeleton used by the readers framework tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from rasterio.windows import Window

from geotoolz.readers._base import SensorReader, Track, as_path
from geotoolz.readers.modis import constants


_DEFAULT_NUM_BANDS = 4


class Reader(SensorReader):
    """Synthetic MODIS reader implementing the sensor-reader contract.

    Args:
        path: Scene path recorded for auditability.
        data: Optional in-memory array in ``(band, y, x)`` order.
        transform: Affine transform for the synthetic grid.
        crs: Coordinate reference system.
        fill_value_default: Fill value used for boundless reads.

    Examples:
        >>> import numpy as np
        >>> from geotoolz.readers.modis import Reader
        >>> reader = Reader("scene", data=np.zeros((4, 8, 8), dtype="float32"))
        >>> reader.load().shape
        (4, 8, 8)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        data: np.ndarray | None = None,
        transform: Affine | None = None,
        crs: Any = "EPSG:4326",
        fill_value_default: float = np.nan,
    ) -> None:
        self.path = as_path(path)
        self._data = (
            np.zeros((_DEFAULT_NUM_BANDS, 1, 1), dtype=np.float32)
            if data is None
            else np.asarray(data)
        )
        self._reader_transform = Affine.identity() if transform is None else transform
        self._reader_crs = crs
        self._fill_value_default = fill_value_default

    @property
    def _crs(self) -> Any:
        return self._reader_crs

    @property
    def _transform(self) -> Affine:
        return self._reader_transform

    @property
    def _shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def _dtype(self) -> Any:
        return self._data.dtype

    @property
    def _bands(self) -> tuple[str, ...]:
        return (
            constants.BAND_BLUE,
            constants.BAND_GREEN,
            constants.BAND_RED,
            constants.BAND_NIR,
        )

    @property
    def _fill_value(self) -> float:
        return self._fill_value_default

    @property
    def _track(self) -> Track:
        return "A"

    def _read_window(self, window: Window) -> np.ndarray:
        col_start = int(window.col_off)
        row_start = int(window.row_off)
        width = int(window.width)
        height = int(window.height)
        out = np.full(
            (*self._data.shape[:-2], height, width),
            self._fill_value_default,
            dtype=self._data.dtype,
        )

        src_col_start = max(col_start, 0)
        src_row_start = max(row_start, 0)
        src_col_stop = min(col_start + width, self.width)
        src_row_stop = min(row_start + height, self.height)
        if src_col_start >= src_col_stop or src_row_start >= src_row_stop:
            return out

        dst_col_start = src_col_start - col_start
        dst_row_start = src_row_start - row_start
        dst_col_stop = dst_col_start + (src_col_stop - src_col_start)
        dst_row_stop = dst_row_start + (src_row_stop - src_row_start)
        out[..., dst_row_start:dst_row_stop, dst_col_start:dst_col_stop] = self._data[
            ..., src_row_start:src_row_stop, src_col_start:src_col_stop
        ]
        return out
