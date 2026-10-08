"""Raster `Field` adapter — bridges `georeader.GeoData` to the Patcher.

`RasterioReader`, `AsyncGeoTIFFReader`, and `GeoTensor` all satisfy the
`GeoData` / `AsyncGeoData` Protocols already. The only thing missing is
the method-name rename (`select` vs `read_from_window`). This adapter is
a one-attribute dataclass that does exactly that, with no behavior of
its own.

Users wrap once at the boundary::

    reader = RasterioReader("scene.tif")
    field  = RasterField(reader)
    for patch in patcher.split(field):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor


try:
    from georeader.abstract_reader import AsyncGeoData
except ImportError:  # pragma: no cover
    # Older georeader-spaceml releases (<= 2.2) don't ship the async
    # mirror yet. Define a minimal Protocol-shaped stub so the rest of
    # this module imports cleanly. `AsyncRasterField` will still raise at
    # construction time when used against the unreleased async surface.
    AsyncGeoData = Any  # type: ignore[assignment, misc]


@dataclass(eq=False)
class RasterField:
    """Wrap a sync `GeoData` (a `RasterioReader`, a `GeoTensor`, …) as a `Field`.

    The reader itself doubles as the `RasterDomain` — the `GeoDataBase`
    Protocol already carries ``crs`` / ``transform`` / ``shape`` /
    ``bounds`` / ``res``.

    Args:
        reader: Any object satisfying `georeader.abstract_reader.GeoData`.
    """

    reader: GeoData

    @property
    def domain(self) -> GeoData:
        return self.reader

    def select(self, window: Any) -> GeoTensor:
        """Read ``window`` (boundless) as a materialised `GeoTensor`.

        Lazy readers such as `RasterioReader` answer ``read_from_window``
        with another lazy reader; it is loaded here so every patch carries
        real pixels (dense aggregation, padding and `stack_patches` all
        need an ndarray, not a reader).
        """
        out = self.reader.read_from_window(window, boundless=True)
        if isinstance(out, GeoTensor):
            return out
        return out.load(boundless=True)

    def with_data(self, array: Any) -> GeoTensor:
        return _rewrap(self.reader, array, self.reader.transform, self.reader.crs)

    def __getstate__(self) -> dict[str, Any]:
        return _pack_reader_state(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(_unpack_reader_state(state))


@dataclass(eq=False)
class AsyncRasterField:
    """Async mirror of `RasterField` over an `AsyncGeoData`.

    `select` is a coroutine; otherwise the surface is identical.
    """

    reader: AsyncGeoData

    @property
    def domain(self) -> AsyncGeoData:
        return self.reader

    async def select(self, window: Any) -> GeoTensor:
        return await self.reader.read_from_window(window, boundless=True)

    async def aselect(self, window: Any) -> GeoTensor:
        return await self.select(window)

    def with_data(self, array: Any) -> GeoTensor:
        return _rewrap(self.reader, array, self.reader.transform, self.reader.crs)


#: ``attrs`` keys that hold one entry per band (band-name aliases and
#: per-band spectral metadata, as geotoolz reads and writes them). They go
#: stale when the band axis changes size, so `_rewrap` drops them then.
PER_BAND_KEYS: tuple[str, ...] = (
    "band_names",
    "descriptions",
    "bands",
    "band_descriptions",
    "wavelengths",
    "wavelengths_nm",
)


def _band_count(shape: tuple[int, ...]) -> int:
    """Size of the band axis (``-3``) of a carrier of ``shape``; 1 when absent."""
    return int(shape[-3]) if len(shape) >= 3 else 1


def _rewrap(reader: Any, array: Any, transform: Any, crs: Any) -> GeoTensor:
    """Wrap ``array`` as a `GeoTensor` on ``reader``'s grid.

    Used by every raster ``with_data`` and by the pipekit ``Stitch``
    operator, so a merged output keeps the source's georeferencing and
    follows the carrier contract:

    * ``attrs`` is a fresh copy of the source's; when the band axis
      changed size (four bands merged into one index), the per-band keys
      (`PER_BAND_KEYS`) are dropped instead of describing bands that are
      gone.
    * ``fill_value_default`` is the source's nodata while the output
      still carries the source's values. A float output on an integer
      source, or on a different band count, is a new quantity whose
      missing cells are NaN, so its fill is NaN.
    """
    values = np.asarray(array)
    source_shape = tuple(getattr(reader, "shape", None) or values.shape)
    same_bands = _band_count(source_shape) == _band_count(values.shape)
    attrs = dict(getattr(reader, "attrs", None) or {})
    if not same_bands:
        for key in PER_BAND_KEYS:
            attrs.pop(key, None)
    fill = getattr(reader, "fill_value_default", 0)
    source_dtype = getattr(reader, "dtype", None)
    source_float = source_dtype is not None and np.issubdtype(
        np.dtype(source_dtype), np.floating
    )
    if np.issubdtype(values.dtype, np.floating) and not (source_float and same_bands):
        fill = np.nan
    return GeoTensor(
        values=array,
        transform=transform,
        crs=crs,
        fill_value_default=fill,
        attrs=attrs,
    )


class _PickledGeoTensor(NamedTuple):
    """A `GeoTensor`'s pixels plus the metadata its own pickle drops."""

    values: np.ndarray
    transform: Any
    crs: Any
    fill_value_default: Any
    attrs: dict[str, Any]


def _pack_reader_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy of a field's ``__dict__`` whose `GeoTensor` reader pickles intact.

    `GeoTensor` subclasses `numpy.ndarray` without extending its pickle
    hook, so a pickled `GeoTensor` comes back with no ``transform`` /
    ``crs`` / ``fill_value_default`` / ``attrs`` — a field wrapping one
    would be unusable in a spawn / forkserver worker. Lazy readers
    (`RasterioReader`) pickle by path and pass through unchanged.
    """
    state = dict(state)
    reader = state.get("reader")
    if isinstance(reader, GeoTensor):
        state["reader"] = _PickledGeoTensor(
            values=np.asarray(reader),
            transform=reader.transform,
            crs=reader.crs,
            fill_value_default=reader.fill_value_default,
            attrs=reader.attrs,
        )
    return state


def _unpack_reader_state(state: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `_pack_reader_state`."""
    reader = state.get("reader")
    if isinstance(reader, _PickledGeoTensor):
        state = {**state, "reader": GeoTensor(**reader._asdict())}
    return state
