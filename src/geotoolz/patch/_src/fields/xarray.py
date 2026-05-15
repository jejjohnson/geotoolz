"""`XarrayField` — adapter for dense, labeled N-D grids.

Wraps an `xarray.DataArray` (typically reanalysis / climate-model
output) and exposes it under the `Field` Protocol with a `GridDomain`
view. The natural indexer is ``dict[str, slice]``, consumed by
`DataArray.isel`.

Optional extra: ``pip install 'geotoolz[grid]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geotoolz.patch._src.domains import GridDomain
from geotoolz.patch._src.fields._extras import _missing_extra


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

    def select(self, indexer: dict[str, slice]) -> XarrayField:
        return XarrayField(self.da.isel(**indexer))

    def with_data(self, array: Any) -> XarrayField:
        new = self.da.copy(data=np.asarray(array))
        return XarrayField(new)
