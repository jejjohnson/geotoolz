"""`RioXarrayField` — raster `Field` adapter on an `xarray.DataArray`.

For users who want the xarray surface end-to-end (chunked Dask reads,
unified xarray pipelines) instead of `GeoTensor`. The domain is still
the raster one — affine + CRS + shape — so all `Rectangular` patching
works the same.

Optional extra: ``pip install 'geotoolz-patcher[xarray-raster]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src._extras import missing_extra


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
            raise missing_extra("RioXarrayField", "xarray-raster", "rioxarray>=0.15")

    @property
    def domain(self) -> Any:
        return _RioDomain(self.da)

    def select(self, window: Any) -> Any:
        """Return the window as a georeferenced `xarray.DataArray`.

        The chip carries an explicit window transform
        (``rasterio.windows.transform(window, domain.transform)``) and,
        when the source has spatial coordinates, coordinates rebuilt from
        it — so ``chip.rio.transform()`` / ``chip.rio.bounds()`` are exact
        whether the source is coordinate-bearing or ``write_transform``-only.

        Windows that extend past the raster are read *boundless*, like
        `RasterField`: the in-range part is read and the rest padded to
        the requested size with ``rio.nodata`` (``0`` when unset), so the
        chip shape always equals the window shape. The returned chip is
        materialised (``.compute()``), also for dask-backed sources.
        """
        from rasterio.windows import Window, transform as window_transform

        y_dim, x_dim = self.da.rio.y_dim, self.da.rio.x_dim
        height, width = int(self.da.sizes[y_dim]), int(self.da.sizes[x_dim])
        r0, c0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        if h < 0 or w < 0:
            raise ValueError(f"window {window!r} has a negative size")
        cr0, cr1 = max(r0, 0), min(r0 + h, height)
        cc0, cc1 = max(c0, 0), min(c0 + w, width)
        if cr1 <= cr0 or cc1 <= cc0:
            # No overlap with the raster: an empty read anchored at the
            # window origin, padded below/right to the full window.
            cr0 = cr1 = r0
            cc0 = cc1 = c0
            sub = self.da.isel({y_dim: slice(0, 0), x_dim: slice(0, 0)})
        else:
            sub = self.da.isel({y_dim: slice(cr0, cr1), x_dim: slice(cc0, cc1)})
        clipped = Window(col_off=cc0, row_off=cr0, width=cc1 - cc0, height=cr1 - cr0)
        transform = window_transform(clipped, self.da.rio.transform())
        sub = _georeference(sub, transform)
        pads = (cr0 - r0, r0 + h - cr1, cc0 - c0, c0 + w - cc1)
        if any(pads):
            sub = pad_dataarray(
                sub, pads, mode="constant", fill=_nodata(self.da), transform=transform
            )
        # Materialise lazily-backed sources (``open_rasterio(chunks=...)``)
        # so the payload is a concrete array, like every other adapter's.
        return sub.compute()

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


def _nodata(da: Any) -> Any:
    """``rio.nodata`` of ``da``, or ``0`` (the `GeoTensor` default) when unset."""
    nodata = da.rio.nodata
    return 0 if nodata is None else nodata


def _georeference(da: Any, transform: Any) -> Any:
    """Write ``transform`` onto ``da`` and rebuild its spatial coords from it.

    Coordinates are only (re)built for spatial dims that already carry
    a coordinate variable, so a ``write_transform``-only array stays
    coordinate-free and relies on the written transform alone.
    """
    y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
    if transform.is_rectilinear:
        updates = {}
        if x_dim in da.coords:
            cols = np.arange(da.sizes[x_dim]) + 0.5
            updates[x_dim] = (x_dim, transform.c + cols * transform.a, da[x_dim].attrs)
        if y_dim in da.coords:
            rows = np.arange(da.sizes[y_dim]) + 0.5
            updates[y_dim] = (y_dim, transform.f + rows * transform.e, da[y_dim].attrs)
        if updates:
            da = da.assign_coords(updates)
    return da.rio.write_transform(transform)


def pad_dataarray(
    da: Any,
    pads: tuple[int, int, int, int],
    *,
    mode: str,
    fill: Any,
    transform: Any | None = None,
) -> Any:
    """Pad a georeferenced chip on its spatial dims, keeping it exact.

    ``xarray.DataArray.pad`` fills padded coordinate values with NaN
    (``mode="constant"``) or mirrors them (``mode="reflect"``); both
    corrupt the georeferencing. The padded array's transform is the
    input's shifted by ``(-left, -top)`` pixels, and its coordinates
    are rebuilt from that transform.

    Args:
        da: A rioxarray-flavoured `DataArray` whose ``rio.transform()``
            is correct for its current extent.
        pads: ``(top, bottom, left, right)`` pad widths in pixels.
        mode: ``"constant"`` or ``"reflect"``.
        fill: Constant for ``mode="constant"``.
        transform: The affine of ``da`` when already known; defaults to
            ``da.rio.transform()``.
    """
    from rasterio.windows import Window, transform as window_transform

    pt, pb, pl, pr = pads
    y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
    transform = window_transform(
        Window(
            col_off=-pl,
            row_off=-pt,
            width=da.sizes[x_dim] + pl + pr,
            height=da.sizes[y_dim] + pt + pb,
        ),
        da.rio.transform() if transform is None else transform,
    )
    const = {"constant_values": fill} if mode == "constant" else {}
    padded = da.pad({y_dim: (pt, pb), x_dim: (pl, pr)}, mode=mode, **const)
    return _georeference(padded, transform)
