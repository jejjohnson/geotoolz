"""HDF5 / NetCDF-4 access and CF-packed decoding for file-format readers.

Most EO products ship as NetCDF-4 (GOES ABI, MTG FCI, TROPOMI, VIIRS,
Sentinel-3) — HDF5 files following the CF conventions. h5py reads them
directly but leaves the conventions to the caller: integers flagged
``_Unsigned``, a ``_FillValue`` and the ``scale_factor`` / ``add_offset``
packing. This module handles them once:

- :class:`PackedVariable` captures one variable's packing and decodes any
  slice of it (``float32`` with ``NaN`` fill, or raw integers for masks).
- :class:`PackedGridReader` is a ``ProductReader`` whose bands are 2-D
  variables of one file on one grid — windowed chunk reads, mask-vs-value
  decoding, CF flag tables. A sensor subclass supplies the grid (CRS,
  transform) and chooses the variables.

h5py is imported at use time; each reader names the extra that provides
it (``[goes]``, …).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

import numpy as np
from rasterio.windows import Window

from geoproducts._src.base import ProductReader
from geoproducts._src.extras import require


__all__ = [
    "PackedGridReader",
    "PackedVariable",
    "Source",
    "attr",
    "grid_variables",
    "h5py_module",
    "scalar",
    "unpacked",
]

#: A file path or an open binary file object (e.g. from ``fsspec``).
Source = str | os.PathLike[str] | IO[bytes]


def h5py_module(feature: str, extra: str) -> Any:
    """Import h5py, or raise an ``ImportError`` naming ``extra``."""
    return require("h5py", feature, extra)


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """An HDF5 attribute as a plain Python scalar / ``str`` (arrays kept)."""
    if name not in obj.attrs:
        return default
    value = obj.attrs[name]
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value
    if isinstance(value, bytes | np.bytes_):
        value = value.decode("utf-8")
    elif isinstance(value, np.generic):
        value = value.item()
    return value


def scalar(f: Any, name: str, *, fill: float | None = None) -> float:
    """A scalar variable as ``float``; ``NaN`` when absent or filled.

    The fill is the variable's own ``_FillValue``, or ``fill`` when the
    product documents one the file does not declare.
    """
    if name not in f:
        return float("nan")
    var = f[name]
    value = float(np.asarray(var[()]).reshape(-1)[0])
    declared = attr(var, "_FillValue")
    fills = {v for v in (declared, fill) if v is not None}
    return float("nan") if value in fills else value


def unpacked(f: Any, name: str) -> np.ndarray:
    """A packed variable (typically a coordinate) decoded to ``float64``."""
    var = f[name]
    raw = np.asarray(var[()], dtype=np.float64)
    scale = float(attr(var, "scale_factor", 1.0))
    return raw * scale + float(attr(var, "add_offset", 0.0))


@dataclass(frozen=True)
class PackedVariable:
    """How to decode one CF-packed 2-D variable.

    Attributes:
        name: Variable name in the file.
        storage: The integer / float dtype pixels are *interpreted* as
            (the unsigned twin of the stored dtype when ``_Unsigned``).
        fill: ``_FillValue`` in ``storage`` terms, or ``None``.
        scale: ``scale_factor`` (``None`` when unpacked).
        offset: ``add_offset`` (``None`` when unpacked).
        units: CF ``units`` (``""`` when absent).
        long_name: CF ``long_name`` (``""`` when absent).
    """

    name: str
    storage: np.dtype
    fill: int | float | None
    scale: float | None
    offset: float | None
    units: str
    long_name: str

    @classmethod
    def from_file(cls, f: Any, name: str) -> PackedVariable:
        """Read the packing attributes of ``name``."""
        var = f[name]
        storage = np.dtype(var.dtype)
        if attr(var, "_Unsigned") == "true" and storage.kind == "i":
            storage = np.dtype(f"u{storage.itemsize}")
        fill = None
        if "_FillValue" in var.attrs:
            raw_fill = np.asarray(var.attrs["_FillValue"]).reshape(-1)[:1]
            fill = raw_fill.astype(var.dtype).view(storage).item()
        scale = attr(var, "scale_factor")
        offset = attr(var, "add_offset")
        return cls(
            name=name,
            storage=storage,
            fill=fill,
            scale=None if scale is None else float(scale),
            offset=None if offset is None else float(offset),
            units=str(attr(var, "units", "")),
            long_name=str(attr(var, "long_name", "")),
        )

    @property
    def is_packed(self) -> bool:
        """Whether decoding changes values (scale / offset applied)."""
        return self.scale is not None or self.offset is not None

    @property
    def is_categorical(self) -> bool:
        """An unpacked integer variable: masks, flags, class codes."""
        return not self.is_packed and self.storage.kind in "iub"

    def raw(self, values: np.ndarray) -> np.ndarray:
        """Stored values reinterpreted in ``storage`` (``_Unsigned`` view)."""
        values = np.asarray(values)
        if values.dtype == self.storage:
            return values
        return np.ascontiguousarray(values).view(self.storage)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Physical values as ``float32``; fill pixels become ``NaN``."""
        raw = self.raw(values)
        out = raw.astype(np.float32)
        if self.scale is not None:
            out *= np.float32(self.scale)
        if self.offset is not None:
            out += np.float32(self.offset)
        if self.fill is not None:
            out[raw == self.fill] = np.nan
        return out


def grid_variables(f: Any, shape: tuple[int, int]) -> tuple[str, ...]:
    """Names of the file's top-level variables of shape ``shape``."""
    # Groups carry no ``shape``; datasets do.
    return tuple(
        name
        for name, var in f.items()
        if tuple(getattr(var, "shape", ()) or ()) == shape
    )


class PackedGridReader(ProductReader):
    """A ``ProductReader`` whose bands are CF-packed 2-D variables of one file.

    The file is re-opened per read: no handle outlives a call, so
    path-backed readers pickle (process pools) and never leak file
    descriptors; only the HDF5 chunks under a window are decompressed.
    Bands decode to ``float32`` / ``NaN`` unless every band is an unpacked
    integer sharing one dtype and fill (masks, class codes), which stay
    integer.

    Subclasses set :attr:`extra` / :attr:`feature` (for the missing-h5py
    message), call :meth:`_open_source` and :meth:`_load_variables`, and
    provide the georeferencing (``_crs``, ``_transform``, ``_track``,
    ``_bands``) plus :attr:`_grid_shape`.
    """

    #: The ``geotoolz-products`` extra that installs h5py for this reader.
    extra: ClassVar[str] = "goes"
    #: The feature named in the missing-extra message.
    feature: ClassVar[str] = "geoproducts"

    _source: Source
    _vars: tuple[PackedVariable, ...] = ()

    @property
    def _grid_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the grid every band lives on."""
        raise NotImplementedError

    @contextmanager
    def _open(self) -> Iterator[Any]:
        with h5py_module(self.feature, self.extra).File(self._source, "r") as f:
            yield f

    def _describe(self) -> str:
        """How messages and ``repr`` name the file."""
        path = self.path
        return path.name if path is not None else type(self._source).__name__

    def _load_variables(self, f: Any, names: Sequence[str]) -> None:
        """Validate ``names`` against the grid and record their packing."""
        loaded = []
        for name in names:
            if name not in f:
                raise ValueError(
                    f"{self._describe()} has no variable {name!r}; 2-D "
                    f"variables: {list(grid_variables(f, self._grid_shape))}."
                )
            shape = tuple(f[name].shape)
            if shape != self._grid_shape:
                raise ValueError(
                    f"variable {name!r} has shape {shape}, not the grid "
                    f"{self._grid_shape}."
                )
            loaded.append(PackedVariable.from_file(f, name))
        self._vars = tuple(loaded)

    @property
    def path(self) -> Path | None:
        """The file path, or ``None`` for a file-like source."""
        if isinstance(self._source, str | os.PathLike):
            return Path(self._source)
        return None

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

    @property
    def _shape(self) -> tuple[int, ...]:
        return (len(self._vars), *self._grid_shape)

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
        return {"units": tuple(v.units for v in self._vars)}

    def _read_window(self, window: Window) -> np.ndarray:
        def read(rows: slice, cols: slice) -> np.ndarray:
            with self._open() as f:
                return self._decode([f[v.name][rows, cols] for v in self._vars])

        return self._read_boundless(window, read)
