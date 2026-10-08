"""The shared ``ProductReader`` for one ABI file: grid, metadata, windowed I/O."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.crs import CRS
from rasterio.windows import Window

from geoproducts._src.base import ProductReader, Track
from geoproducts.goes._src.grid import FileInfo, FixedGrid
from geoproducts.goes._src.hdf import PackedVariable, attr, h5py_module


Source = str | os.PathLike[str] | IO[bytes]


class ABIFile(ProductReader):
    """One ABI product file read as a ``(bands, H, W)`` ``GeoTensor``.

    Each band is one 2-D variable of the file, all on the file's fixed
    grid. Subclasses choose the variables and how raw values decode
    (``_decode``); this class owns the grid, the metadata, and the windowed
    HDF5 reads (only the chunks under a window are decompressed).
    """

    _source: Source
    _grid: FixedGrid
    _info: FileInfo
    _vars: tuple[PackedVariable, ...]

    def _init_file(self, source: Source, variables: Sequence[str]) -> None:
        self._source = source
        with self._open() as f:
            self._grid = FixedGrid.from_file(f)
            self._info = FileInfo.from_file(f)
            self._vars = tuple(self._variable(f, name) for name in variables)

    def _variable(self, f: Any, name: str) -> PackedVariable:
        if name not in f:
            raise ValueError(
                f"{self._info.dataset_name or 'file'} has no variable {name!r}; "
                f"2-D variables: {list(grid_variables(f))}."
            )
        shape = tuple(f[name].shape)
        if shape != (self._grid.height, self._grid.width):
            raise ValueError(
                f"variable {name!r} has shape {shape}, not the fixed grid "
                f"{(self._grid.height, self._grid.width)}."
            )
        return PackedVariable.from_file(f, name)

    def _share(self, other: ABIFile, variables: Sequence[str]) -> None:
        """Point at ``other``'s file, reusing its parsed grid and metadata."""
        self._source = other._source
        self._grid = other._grid
        self._info = other._info
        with self._open() as f:
            self._vars = tuple(self._variable(f, name) for name in variables)

    @contextmanager
    def _open(self) -> Iterator[Any]:
        # Re-opened per read: no handle outlives a call, so path-backed
        # readers pickle (process pools) and never leak file descriptors.
        with h5py_module().File(self._source, "r") as f:
            yield f

    # -- georeferencing ---------------------------------------------------

    @property
    def _crs(self) -> CRS:
        return self._grid.crs

    @property
    def _transform(self) -> Affine:
        return self._grid.transform

    @property
    def _shape(self) -> tuple[int, ...]:
        return (len(self._vars), self._grid.height, self._grid.width)

    @property
    def _track(self) -> Track:
        return "A"

    # -- file metadata ----------------------------------------------------

    @property
    def path(self) -> Path | None:
        """The file path, or ``None`` for a file-like source."""
        if isinstance(self._source, str | os.PathLike):
            return Path(self._source)
        return None

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

    @property
    def variables(self) -> tuple[str, ...]:
        """The file variables read, one per band, in band order."""
        return tuple(v.name for v in self._vars)

    def flags(self, variable: str | None = None) -> dict[int, str]:
        """A flag variable's ``{value: meaning}`` table, from its CF attributes.

        Args:
            variable: Any variable of the file (default: the first band's).

        Returns:
            ``flag_values`` zipped with ``flag_meanings``; empty when the
            variable declares none.

        Raises:
            ValueError: The variable pairs its values and meanings unevenly
                (bit fields declared through ``flag_masks``).
        """
        name = variable or self._vars[0].name
        with self._open() as f:
            var = f[name]
            values = attr(var, "flag_values")
            meanings = attr(var, "flag_meanings")
        if values is None or meanings is None:
            return {}
        values = np.atleast_1d(values).tolist()
        names = str(meanings).split()
        if len(values) != len(names):
            raise ValueError(
                f"{name!r} declares {len(values)} flag values but {len(names)} "
                "meanings (a bit field?); decode it from its flag_masks."
            )
        return dict(zip((int(v) for v in values), names, strict=True))

    # -- windowed I/O -----------------------------------------------------

    @property
    def _categorical(self) -> bool:
        """Whether every band is an unpacked integer sharing dtype and fill."""
        return all(v.is_categorical for v in self._vars) and (
            len({(v.storage, v.fill) for v in self._vars}) == 1
        )

    @property
    def _dtype(self) -> Any:
        return self._vars[0].storage if self._categorical else np.dtype(np.float32)

    @property
    def _fill_value(self) -> Any:
        if not self._categorical:
            return np.nan
        fill = self._vars[0].fill
        return np.iinfo(self._vars[0].storage).max if fill is None else fill

    def _decode(self, raw: list[np.ndarray]) -> np.ndarray:
        """Decode one window's raw variable slices into ``(bands, h, w)``."""
        if self._categorical:
            return np.stack([v.raw(r) for v, r in zip(self._vars, raw, strict=True)])
        return np.stack([v.decode(r) for v, r in zip(self._vars, raw, strict=True)])

    def _band_attrs(self) -> dict[str, Any]:
        """Extra per-read ``attrs`` (``band_names`` is set by the base class)."""
        return {"units": tuple(v.units for v in self._vars)}

    def _read_window(self, window: Window) -> np.ndarray:
        row0, col0 = int(window.row_off), int(window.col_off)
        n_rows, n_cols = int(window.height), int(window.width)
        out = np.full(
            (len(self._vars), n_rows, n_cols), self._fill_value, dtype=self._dtype
        )
        r_start, r_stop = max(row0, 0), min(row0 + n_rows, self._grid.height)
        c_start, c_stop = max(col0, 0), min(col0 + n_cols, self._grid.width)
        if r_start >= r_stop or c_start >= c_stop:
            return out
        with self._open() as f:
            raw = [f[v.name][r_start:r_stop, c_start:c_stop] for v in self._vars]
        out[:, r_start - row0 : r_stop - row0, c_start - col0 : c_stop - col0] = (
            self._decode(raw)
        )
        return out

    def read_from_window(self, window: Window, boundless: bool = True) -> GeoTensor:
        """Read a pixel window as a ``(bands, h, w)`` ``GeoTensor``.

        Besides ``band_names``, the output's ``attrs`` carry per-band
        ``units`` (and, for L1b, ``wavelengths`` in nm and ``calibration``).
        """
        out = super().read_from_window(window, boundless=boundless)
        out.attrs = {**(out.attrs or {}), **self._band_attrs()}
        return out

    def _repr_name(self) -> str:
        return self._info.dataset_name or type(self._source).__name__


def grid_variables(f: Any) -> tuple[str, ...]:
    """Names of the file's 2-D variables on the fixed grid."""
    h5py = h5py_module()
    n_rows, n_cols = len(f["y"]), len(f["x"])
    return tuple(
        name
        for name, var in f.items()
        if isinstance(var, h5py.Dataset) and var.shape == (n_rows, n_cols)
    )
