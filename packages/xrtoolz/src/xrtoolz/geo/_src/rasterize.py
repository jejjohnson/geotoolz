"""Vector → raster: burn geometries onto a grid.

The target grid is read with :mod:`rioxarray` (``like.rio.transform()``,
``like.rio.crs``, ``like.rio.shape``); burning is
:func:`rasterio.features.rasterize` from the rasterio stack that
``rioxarray`` is built on. The algorithms mirror ``georeader.rasterize``;
only the I/O is xarray.

``geopandas`` / ``shapely`` are imported lazily — install them with
``pip install 'xrtoolz[vector]'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import rioxarray  # noqa: F401  — needed so da.rio is populated
import xarray as xr
from pyproj import CRS
from rioxarray.rioxarray import affine_to_coords

from xrtoolz.geo._src.vectorize import _require_vector, transform_polygon


if TYPE_CHECKING:
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry.base import BaseGeometry

    Geometries = (
        gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | Sequence[BaseGeometry]
    )


def _burn_pairs(
    geometries: Geometries,
    *,
    column: str | None,
    value: float,
    geometries_crs: Any,
    dst_crs: Any,
) -> list[tuple[BaseGeometry, Any]]:
    """Normalise ``geometries`` to ``(geometry, burn value)`` pairs in ``dst_crs``."""
    gpd = _require_vector("geopandas", "rasterize")
    shapely_base = _require_vector("shapely.geometry.base", "rasterize")

    if isinstance(geometries, gpd.GeoDataFrame | gpd.GeoSeries):
        src_crs = geometries.crs if geometries.crs is not None else geometries_crs
        if src_crs is not None and dst_crs is not None:
            geometries = geometries.set_crs(src_crs, allow_override=True).to_crs(
                dst_crs
            )
        geoms = list(geometries.geometry)
        if column is None:
            values = [value] * len(geoms)
        elif isinstance(geometries, gpd.GeoDataFrame):
            values = list(geometries[column])
        else:
            raise ValueError("column= requires a GeoDataFrame input.")
    else:
        if column is not None:
            raise ValueError("column= requires a GeoDataFrame input.")
        if isinstance(geometries, shapely_base.BaseGeometry):
            geometries = [geometries]
        geoms = list(geometries)
        if geometries_crs is not None and dst_crs is not None:
            geoms = [transform_polygon(g, geometries_crs, dst_crs) for g in geoms]
        values = [value] * len(geoms)
    return [(g, v) for g, v in zip(geoms, values, strict=True) if not g.is_empty]


def _burn(
    pairs: list[tuple[BaseGeometry, Any]],
    *,
    transform: Affine,
    out_shape: tuple[int, int],
    fill: float,
    all_touched: bool,
    dtype: str,
) -> np.ndarray:
    from rasterio import features

    if not pairs:  # rasterio refuses an empty shape list
        return np.full(out_shape, fill, dtype=dtype)
    return features.rasterize(
        pairs,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )


def rasterize_like(
    geometries: Geometries,
    like: xr.DataArray | xr.Dataset,
    *,
    column: str | None = None,
    fill: float = 0.0,
    value: float = 1.0,
    all_touched: bool = False,
    dtype: str = "float32",
    geometries_crs: Any = None,
) -> xr.DataArray:
    """Burn vector geometries onto the grid defined by ``like``.

    Geometries are reprojected to ``like.rio.crs`` first. Overlapping
    geometries resolve last-wins (rasterio's order).

    Args:
        geometries: ``GeoDataFrame`` / ``GeoSeries``, a single shapely
            geometry, or a sequence of them.
        like: Raster whose spatial coords, transform and CRS define the
            output grid (``like.rio.*``).
        column: ``GeoDataFrame`` column holding each geometry's burn
            value. ``None`` burns ``value`` everywhere.
        fill: Value for pixels no geometry covers.
        value: Burn value when ``column`` is ``None``.
        all_touched: Burn every pixel a geometry touches, not only those
            whose centre falls inside it.
        dtype: Output dtype.
        geometries_crs: CRS of bare shapely geometries (or of a
            ``GeoDataFrame`` without one). ``None`` assumes they are
            already in ``like``'s CRS.

    Returns:
        2-D ``(y, x)`` DataArray on ``like``'s grid, with ``like``'s
        spatial coords and CRS.

    Raises:
        ImportError: If the ``vector`` extra (geopandas/shapely) is missing.
        ValueError: If ``column`` is given for non-``GeoDataFrame`` input.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from shapely.geometry import box
        >>> from xrtoolz.geo import rasterize_like
        >>> like = xr.DataArray(
        ...     np.zeros((3, 3)),
        ...     dims=("y", "x"),
        ...     coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
        ... ).rio.write_crs("EPSG:32630")
        >>> rasterize_like(box(1, 1, 3, 3), like, dtype="uint8").values
        array([[0, 1, 1],
               [0, 1, 1],
               [0, 0, 0]], dtype=uint8)
    """
    y_dim, x_dim = like.rio.y_dim, like.rio.x_dim
    transform = like.rio.transform(recalc=True)
    crs = like.rio.crs
    pairs = _burn_pairs(
        geometries,
        column=column,
        value=value,
        geometries_crs=geometries_crs,
        dst_crs=crs,
    )
    data = _burn(
        pairs,
        transform=transform,
        out_shape=(like.rio.height, like.rio.width),
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )
    out = xr.DataArray(
        data,
        dims=(y_dim, x_dim),
        coords={y_dim: like[y_dim].variable, x_dim: like[x_dim].variable},
        name=column or "rasterized",
    )
    if crs is not None:
        out = out.rio.write_crs(crs)
    return out.rio.write_transform(transform)


def rasterize(
    geometries: Geometries,
    *,
    transform: Affine,
    out_shape: tuple[int, int],
    crs: Any = None,
    column: str | None = None,
    fill: float = 0.0,
    value: float = 1.0,
    all_touched: bool = False,
    dtype: str = "float32",
    geometries_crs: Any = None,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    """Burn vector geometries onto an explicitly specified grid.

    The lower-level sibling of :func:`rasterize_like` for when there is
    no reference cube: the caller supplies the affine ``transform``,
    ``out_shape`` and ``crs``. Pixel-centre coordinates are derived from
    the transform with :func:`rioxarray.rioxarray.affine_to_coords`.

    Args:
        geometries: ``GeoDataFrame`` / ``GeoSeries``, a single shapely
            geometry, or a sequence of them.
        transform: Affine pixel → CRS transform of the output grid.
        out_shape: Output ``(height, width)``.
        crs: CRS of the output grid; geometries are reprojected to it.
        column: ``GeoDataFrame`` column holding each geometry's burn value.
        fill: Value for pixels no geometry covers.
        value: Burn value when ``column`` is ``None``.
        all_touched: Burn every pixel a geometry touches.
        dtype: Output dtype.
        geometries_crs: CRS of bare shapely geometries. ``None`` assumes
            they are already in ``crs``.
        x_dim: Name of the output x dimension.
        y_dim: Name of the output y dimension.

    Returns:
        2-D ``(y_dim, x_dim)`` DataArray with pixel-centre coords, the
        transform and (if given) the CRS attached.

    Example:
        >>> from affine import Affine
        >>> from shapely.geometry import box
        >>> from xrtoolz.geo import rasterize
        >>> out = rasterize(
        ...     box(0, 0, 2, 2),
        ...     transform=Affine(1, 0, 0, 0, -1, 3),
        ...     out_shape=(3, 3),
        ...     crs="EPSG:32630",
        ...     dtype="uint8",
        ... )
        >>> out.values
        array([[0, 0, 0],
               [1, 1, 0],
               [1, 1, 0]], dtype=uint8)
        >>> out["x"].values.tolist(), out.rio.crs.to_epsg()
        ([0.5, 1.5, 2.5], 32630)
    """
    height, width = out_shape
    pairs = _burn_pairs(
        geometries,
        column=column,
        value=value,
        geometries_crs=geometries_crs,
        dst_crs=crs,
    )
    data = _burn(
        pairs,
        transform=transform,
        out_shape=(height, width),
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )
    coords = affine_to_coords(transform, width, height, x_dim=x_dim, y_dim=y_dim)
    out = xr.DataArray(
        data,
        dims=(y_dim, x_dim),
        coords=coords,
        name=column or "rasterized",
    )
    if crs is not None:
        out = out.rio.write_crs(CRS.from_user_input(crs))
    return out.rio.write_transform(transform)


__all__ = ["rasterize", "rasterize_like"]
