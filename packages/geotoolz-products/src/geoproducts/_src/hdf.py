"""CF-packed NetCDF-4 / HDF5 product readers, on geotoolz-cloud's decoders.

The decoding itself — ``_Unsigned``, ``_FillValue``, ``scale_factor`` /
``add_offset`` (`PackedVariable`), attribute helpers (`attr`, `scalar`,
`unpacked`, `time_attr`, `grid_variables`) and opening a file from any
location with ranged reads (`open_hdf5`) — lives in `geocloud.hdf`. This
module adds `PackedGridReader`, the ``ProductReader`` whose bands are the
2-D variables of one file on one grid (windowed chunk reads, mask-vs-value
decoding, CF flag tables); a sensor subclass supplies the grid (CRS,
transform) and chooses the variables.

geotoolz-cloud (and h5py) come with each reader's extra (``[goes]``,
``[himawari]``). The `geocloud.hdf` names are resolved on first use
(``hdf.attr(...)``), so this module — and every reader — imports on a
base install and fails only when a file is read, naming the extra.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import IO, TYPE_CHECKING, Any, ClassVar

import numpy as np
from rasterio.windows import Window

from geoproducts._src.base import ProductReader
from geoproducts._src.extras import missing_extra, require


if TYPE_CHECKING:
    from geocloud.hdf import (
        PackedVariable,
    )


__all__ = ["PackedGridReader", "Source"]

#: A URI (any location geotoolz-cloud reads), a local path, or an open
#: binary file object.
Source = str | os.PathLike[str] | IO[bytes]

#: The `geocloud.hdf` names this module hands out, resolved on first use.
_CLOUD_NAMES = frozenset(
    {
        "PackedVariable",
        "attr",
        "grid_variables",
        "open_hdf5",
        "scalar",
        "time_attr",
        "unpacked",
    }
)


def _cloud_hdf(
    feature: str = "geoproducts HDF readers", extra: str = "goes"
) -> ModuleType:
    """`geocloud.hdf`, or an ``ImportError`` naming the reader's extra."""
    return require("geocloud.hdf", feature, extra)


def __getattr__(name: str) -> Any:
    if name in _CLOUD_NAMES:
        return getattr(_cloud_hdf(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        hdf = _cloud_hdf(self.feature, self.extra)
        with contextlib.ExitStack() as stack:
            try:
                f = stack.enter_context(hdf.open_hdf5(self._source))
            except ImportError as exc:  # geotoolz-cloud is there, h5py is not
                raise missing_extra(self.feature, self.extra) from exc
            yield f

    def _describe(self) -> str:
        """How messages and ``repr`` name the file."""
        if isinstance(self._source, str | os.PathLike):
            return os.fspath(self._source).rstrip("/").rsplit("/", 1)[-1]
        return type(self._source).__name__

    def _load_variables(self, f: Any, names: Sequence[str]) -> None:
        """Validate ``names`` against the grid and record their packing."""
        loaded = []
        for name in names:
            if name not in f:
                raise ValueError(
                    f"{self._describe()} has no variable {name!r}; 2-D variables: "
                    f"{list(_cloud_hdf().grid_variables(f, self._grid_shape))}."
                )
            shape = tuple(f[name].shape)
            if shape != self._grid_shape:
                raise ValueError(
                    f"variable {name!r} has shape {shape}, not the grid "
                    f"{self._grid_shape}."
                )
            loaded.append(_cloud_hdf().PackedVariable.from_file(f, name))
        self._vars = tuple(loaded)

    @property
    def path(self) -> Path | None:
        """The local file path; ``None`` for a remote URI or a file-like source."""
        if isinstance(self._source, os.PathLike) or (
            isinstance(self._source, str) and "://" not in self._source
        ):
            return Path(self._source)
        if isinstance(self._source, str) and self._source.startswith("file://"):
            return Path(self._source[len("file://") :])
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
        hdf = _cloud_hdf(self.feature, self.extra)
        with self._open() as f:
            var = f[name]
            values = hdf.attr(var, "flag_values")
            meanings = hdf.attr(var, "flag_meanings")
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
