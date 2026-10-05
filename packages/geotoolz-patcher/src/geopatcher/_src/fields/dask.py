"""`DaskField` — adapter for dask-backed xarray arrays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from geopatcher._src._extras import missing_extra
from geopatcher._src.domains import GridDomain


try:
    import dask.array as da  # type: ignore[import-untyped]
    import xarray as xr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    da = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]


@dataclass(eq=False)
class DaskField:
    """Wrap a dask-backed `xarray.DataArray` as a lazy grid `Field`."""

    array: Any

    def __post_init__(self) -> None:
        if da is None or xr is None:
            raise missing_extra("DaskField", "dask", "dask[bag]>=2024.8.3")
        if not isinstance(self.array.data, da.Array):
            self.array = self.array.chunk()

    @classmethod
    def from_zarr(
        cls, store: Any, *, var: str | None = None, **kwargs: Any
    ) -> DaskField:
        """Open one variable of a zarr store with xarray as a `DaskField`.

        ``xr.open_zarr`` returns a `Dataset`; the field wraps one of its
        data variables.

        Args:
            store: Anything ``xarray.open_zarr`` accepts (path, URL, store).
            var: Data variable to wrap. May be omitted when the store holds
                exactly one data variable.
            **kwargs: Forwarded to ``xarray.open_zarr``.

        Raises:
            KeyError: ``var`` is not a data variable of the store.
            ValueError: ``var`` is omitted and the store holds zero or
                several data variables.
        """
        if xr is None:
            raise missing_extra("DaskField", "dask", "dask[bag]>=2024.8.3")
        ds = xr.open_zarr(store, **kwargs)
        names = list(ds.data_vars)
        if var is None:
            if len(names) != 1:
                raise ValueError(
                    f"DaskField.from_zarr: store {store!r} has data variables "
                    f"{names}; pass var=<name> to choose one."
                )
            var = names[0]
        elif var not in ds.data_vars:
            raise KeyError(
                f"DaskField.from_zarr: {var!r} is not a data variable of "
                f"{store!r}; available: {names}."
            )
        return cls(ds[var])

    @cached_property
    def domain(self) -> GridDomain:
        coords = {d: np.asarray(self.array[d].values) for d in self.array.dims}
        crs = getattr(self.array, "rio", None)
        crs = crs.crs if crs is not None else None
        return GridDomain(coords=coords, crs=crs)

    def select(self, indexer: dict[str, slice]) -> Any:
        """Read one patch as a materialised `xarray.DataArray`.

        Returns the computed slice (not another `DaskField`), so the
        payload is `np.asarray`-able and feeds the spatial aggregations
        directly — mirroring `XarrayField.select`. Coordinates come along
        with ``isel``; when the source is georeferenced through rioxarray
        (it carries a grid-mapping coordinate) the chip also gets an
        explicit window transform, since ``rio.transform()`` on a
        coordinate-free slice would otherwise report the full-array affine.
        """
        sub = _write_window_transform(self.array, self.array.isel(**indexer), indexer)
        return sub.compute()

    def with_data(self, array: Any) -> DaskField:
        return DaskField(self.array.copy(data=array))


def _write_window_transform(full: Any, sub: Any, indexer: dict[str, Any]) -> Any:
    """Write the window affine onto ``sub`` when ``full`` is rioxarray-georeferenced."""
    try:
        rio = full.rio  # only registered once rioxarray has been imported
    except AttributeError:
        return sub
    try:
        y_dim, x_dim = rio.y_dim, rio.x_dim
        georeferenced = rio.grid_mapping in full.coords
    except Exception:  # rioxarray's MissingSpatialDimensionError & co.
        return sub
    if not georeferenced:
        return sub
    offsets = []
    for dim in (y_dim, x_dim):
        index = indexer.get(dim, slice(None))
        if not isinstance(index, slice) or index.step not in (None, 1):
            return sub
        offsets.append(index.indices(full.sizes[dim])[0])
    from rasterio.windows import Window, transform as window_transform

    window = Window(
        col_off=offsets[1],
        row_off=offsets[0],
        width=sub.sizes[x_dim],
        height=sub.sizes[y_dim],
    )
    return sub.rio.write_transform(window_transform(window, rio.transform()))
