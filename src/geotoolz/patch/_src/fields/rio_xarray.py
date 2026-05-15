"""`RioXarrayField` — raster `Field` adapter on an `xarray.DataArray`.

For users who want the xarray surface end-to-end (chunked Dask reads,
unified xarray pipelines) instead of `GeoTensor`. The domain is still
the raster one — affine + CRS + shape — so all `Rectangular` patching
works the same.

Optional extra: ``pip install 'geotoolz[xarray-raster]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geotoolz.patch._src.fields._extras import _missing_extra


try:
    import rioxarray  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    rioxarray = None  # type: ignore[assignment]


@dataclass(eq=False)
class RioXarrayField:
    """Wrap a rioxarray-flavoured `xarray.DataArray` as a raster `Field`.

    The `domain` view exposes the rasterio affine, the array shape, and
    the CRS — i.e. it satisfies the same metadata surface as
    `GeoDataBase` from georeader, so it interoperates with the existing
    raster patching path.

    Args:
        da: An `xarray.DataArray` with a working ``da.rio`` accessor.
    """

    da: Any

    def __post_init__(self) -> None:
        if rioxarray is None:  # pragma: no cover
            raise _missing_extra("RioXarrayField", "xarray-raster", "rioxarray>=0.15")

    @property
    def domain(self) -> Any:
        return _RioDomain(self.da)

    def select(self, window: Any) -> RioXarrayField:
        # rasterio.windows.Window → row/col slice via `isel` on the
        # spatial dims of the DataArray.
        y_dim, x_dim = self.da.rio.y_dim, self.da.rio.x_dim
        rs = slice(int(window.row_off), int(window.row_off + window.height))
        cs = slice(int(window.col_off), int(window.col_off + window.width))
        return RioXarrayField(self.da.isel({y_dim: rs, x_dim: cs}))

    def with_data(self, array: Any) -> RioXarrayField:
        return RioXarrayField(self.da.copy(data=np.asarray(array)))


@dataclass(eq=False)
class _RioDomain:
    """`GeoDataBase`-shaped view over a rioxarray DataArray."""

    da: Any

    @property
    def transform(self) -> Any:
        return self.da.rio.transform()

    @property
    def crs(self) -> Any:
        return self.da.rio.crs

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.da.shape)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(self.da.rio.bounds())  # type: ignore[return-value]
