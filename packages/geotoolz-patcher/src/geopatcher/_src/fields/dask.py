"""`DaskField` — adapter for dask-backed xarray arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.domains import GridDomain
from geopatcher._src.fields._extras import _missing_extra


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
            raise _missing_extra("DaskField", "dask", "dask[bag]>=2024.8.3")
        if not isinstance(self.array.data, da.Array):
            self.array = self.array.chunk()

    @classmethod
    def from_zarr(cls, store: Any, **kwargs: Any) -> DaskField:
        """Open a zarr-backed array with xarray and wrap it as a `DaskField`."""
        if xr is None:
            raise _missing_extra("DaskField", "dask", "dask[bag]>=2024.8.3")
        return cls(xr.open_zarr(store, **kwargs))

    @property
    def domain(self) -> GridDomain:
        coords = {d: np.asarray(self.array[d].values) for d in self.array.dims}
        crs = getattr(self.array, "rio", None)
        crs = crs.crs if crs is not None else None
        return GridDomain(coords=coords, crs=crs)

    def select(self, indexer: dict[str, slice]) -> DaskField:
        return DaskField(self.array.isel(**indexer))

    def with_data(self, array: Any) -> DaskField:
        return DaskField(self.array.copy(data=array))
