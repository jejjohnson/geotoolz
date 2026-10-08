"""HDF5 access and CF-packed variable decoding for ABI NetCDF-4 files.

ABI files (L1b and L2 alike) are NetCDF-4, i.e. HDF5. h5py reads them
directly but leaves the CF conventions to the caller: integers flagged
``_Unsigned``, a ``_FillValue`` and the ``scale_factor`` / ``add_offset``
packing. :class:`PackedVariable` captures those once per variable and
decodes any slice of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geoproducts._src.extras import missing_extra


# Scalar calibration / metadata variables use -999 as their ``_FillValue``.
SCALAR_FILL = -999.0


def h5py_module() -> Any:
    """Import h5py, or raise an ``ImportError`` naming the ``[goes]`` extra."""
    try:
        import h5py
    except ImportError as exc:
        raise missing_extra("geoproducts.goes", "goes") from exc
    return h5py


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


def scalar(f: Any, name: str) -> float:
    """A scalar variable as ``float``; ``NaN`` when absent or filled."""
    if name not in f:
        return float("nan")
    value = float(np.asarray(f[name][()]).reshape(-1)[0])
    return float("nan") if value == SCALAR_FILL else value


def decoded_coordinate(f: Any, name: str) -> np.ndarray:
    """A packed ``x`` / ``y`` scan-angle coordinate, in radians (float64)."""
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
