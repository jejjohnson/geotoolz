"""`XarrayField` — adapter for dense, labeled N-D grids.

Wraps an `xarray.DataArray` (typically reanalysis / climate-model
output) and exposes it under the `Field` Protocol with a `GridDomain`
view. The natural indexer is ``dict[str, slice]``, consumed by
`DataArray.isel`.

Optional extra: ``pip install 'geopatcher[grid]'``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.domains import GridDomain
from geopatcher._src.fields._extras import _missing_extra


try:
    import xarray as xr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    xr = None  # type: ignore[assignment]


@dataclass(eq=False)
class XarrayField:
    """Wrap an `xarray.DataArray` as a `Field[GridDomain]`.

    Args:
        da: The underlying `xarray.DataArray`. Any CRS must be exposed
            via the rioxarray accessor (``da.rio.crs``); ``None`` is
            allowed for non-georeferenced cubes.
    """

    da: Any

    def __post_init__(self) -> None:
        if xr is None:
            raise _missing_extra("XarrayField", "grid", "xarray>=2024.1")

    @property
    def domain(self) -> GridDomain:
        coords = {d: np.asarray(self.da[d].values) for d in self.da.dims}
        crs = getattr(self.da, "rio", None)
        crs = crs.crs if crs is not None else None
        return GridDomain(coords=coords, crs=crs)

    def select(self, indexer: dict[str, slice]) -> Any:
        """Read a sliced `xarray.DataArray` patch.

        Returns the bare `DataArray` (not another `XarrayField`) so the
        result is `np.asarray`-able and feeds straight into the spatial
        aggregations. Mirrors `RasterField.select → GeoTensor`: select
        returns the natural data payload, not another field wrapper. Use
        `XarrayField(da_sliced)` if you need to keep treating the slice
        as a sub-Field.
        """
        return self.da.isel(**indexer)

    def with_data(self, array: Any) -> XarrayField:
        new = self.da.copy(data=np.asarray(array))
        return XarrayField(new)

    def time_coord(self, name: str = "time") -> np.ndarray:
        """Return the 1-D time coordinate as a NumPy array.

        Helper for the coordinate-aware temporal patcher path. Resolves the
        coordinate by name (defaults to ``"time"``) and materialises its
        values so callers don't repeat ``ds[name].values`` boilerplate.

        Args:
            name: Name of the time-like coordinate. Defaults to ``"time"``.

        Returns:
            ``np.ndarray`` of dtype ``datetime64[ns]`` (or whatever NumPy
            unit xarray exposes for that coord).

        Raises:
            KeyError: If ``name`` is absent from the DataArray's coords.
            TypeError: If the coordinate is `cftime`-typed — convert via
                ``xarray.coding.times.convert_calendar(...)`` or
                ``ds.indexes['time'].to_datetimeindex()`` first.
        """
        if name not in self.da.coords:
            raise KeyError(
                f"XarrayField has no coord named {name!r}; "
                f"available: {list(self.da.coords)}"
            )
        values = np.asarray(self.da.coords[name].values)
        if values.dtype == np.dtype("O"):
            raise TypeError(
                f"Coord {name!r} has object dtype (likely cftime). "
                "Convert with xarray.coding.times.convert_calendar(...) "
                "or DataArray.indexes['time'].to_datetimeindex() before "
                "passing to TemporalPatcher."
            )
        return values

    def coords_per_patch(self, patches: Iterable[Any]) -> list[Any]:
        """xrpatcher.get_coords() equivalent — one coord Dataset per patch.

        For each patch, run ``self.da.isel(**patch.indices).coords.to_dataset()``
        so the caller gets the coord values aligned with the patch slice
        without re-reading data.

        Args:
            patches: Iterable of `Patch` (or anything with an `indices`
                attribute consumable by ``DataArray.isel``).

        Returns:
            ``list[xr.Dataset]`` — same length as ``patches``, each Dataset
            containing only the coord variables (no data variables).

        Raises:
            TypeError: If a patch's ``indices`` is not a ``dict`` and can't
                be coerced to ``DataArray.isel`` kwargs. Raster-style
                ``rasterio.windows.Window`` indices are converted using
                the underlying DataArray's last two dims as ``(row, col)``.
        """
        out: list[Any] = []
        for p in patches:
            indexer = _patch_indices_to_isel(p, self.da)
            out.append(self.da.isel(**indexer).coords.to_dataset())
        return out


def _patch_indices_to_isel(patch: Any, da: Any) -> dict[str, slice]:
    """Coerce a patch's indices into an `xarray.isel` kwarg dict."""
    indices = patch.indices
    if isinstance(indices, dict):
        return indices
    # rasterio.windows.Window-style — map to the last two dims.
    if hasattr(indices, "row_off") and hasattr(indices, "col_off"):
        if len(da.dims) < 2:
            raise TypeError(
                "Raster-style patch.indices needs at least 2 dims on the "
                f"DataArray; got dims={da.dims!r}."
            )
        row_dim, col_dim = da.dims[-2], da.dims[-1]
        r0 = int(indices.row_off)
        c0 = int(indices.col_off)
        h = int(indices.height)
        w = int(indices.width)
        return {row_dim: slice(r0, r0 + h), col_dim: slice(c0, c0 + w)}
    raise TypeError(
        f"Can't convert patch.indices of type {type(indices).__name__} "
        "to xarray.isel kwargs; only dict and rasterio.windows.Window are "
        "supported."
    )
