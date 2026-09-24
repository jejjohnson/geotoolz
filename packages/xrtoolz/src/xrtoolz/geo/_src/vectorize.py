"""Raster → vector: polygonize masks and reproject geometries.

Georeferencing comes from :mod:`rioxarray` (``mask.rio.transform()`` /
``mask.rio.crs``); polygonization is :func:`rasterio.features.shapes`
and geometry reprojection is :func:`rasterio.warp.transform_geom` —
both from the rasterio stack that ``rioxarray`` is built on. The
algorithms mirror ``georeader.vectorize``; only the I/O is xarray.

``geopandas`` / ``shapely`` are imported lazily — install them with
``pip install 'xrtoolz[vector]'``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import rioxarray  # noqa: F401  — needed so da.rio is populated
import xarray as xr
from pyproj import CRS

from xrtoolz.utils._src.optional_imports import _require_optional


if TYPE_CHECKING:
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry.base import BaseGeometry


# dtypes accepted by rasterio.features.shapes.
_SHAPES_DTYPES = frozenset({"int16", "int32", "uint8", "uint16", "float32"})


def _require_vector(module: str, feature: str) -> Any:
    package = module.split(".", 1)[0]
    return _require_optional(module, extra="vector", feature=feature, package=package)


def _as_raster_2d(obj: xr.DataArray | xr.Dataset, *, feature: str) -> xr.DataArray:
    """Coerce ``obj`` to a 2-D ``(y, x)`` DataArray carrying its CRS.

    A single-variable ``Dataset`` is unwrapped (so the function also
    works on ``DataTree`` leaves); size-1 non-spatial dims are squeezed.
    """
    if isinstance(obj, xr.Dataset):
        if len(obj.data_vars) != 1:
            raise TypeError(
                f"{feature} expects a DataArray or a single-variable Dataset; "
                f"got a Dataset with variables {list(obj.data_vars)}."
            )
        da = next(iter(obj.data_vars.values()))
        if da.rio.crs is None and obj.rio.crs is not None:
            da = da.rio.write_crs(obj.rio.crs)
    elif isinstance(obj, xr.DataArray):
        da = obj
    else:
        raise TypeError(
            f"{feature} expects an xarray DataArray, got {type(obj).__name__}."
        )
    y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
    size1 = [d for d in da.dims if d not in (y_dim, x_dim) and da.sizes[d] == 1]
    da = da.squeeze(size1) if size1 else da
    if da.ndim != 2:
        raise ValueError(
            f"{feature} expects a 2-D ({y_dim}, {x_dim}) raster; "
            f"got dims {tuple(da.dims)}."
        )
    return da.transpose(y_dim, x_dim)


def _to_shapes_dtype(values: np.ndarray) -> np.ndarray:
    """Cast ``values`` to a dtype :func:`rasterio.features.shapes` accepts."""
    if values.dtype == np.bool_:
        return values.astype(np.uint8)
    if values.dtype.name in _SHAPES_DTYPES:
        return values
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(np.int32)
        if values.size and (values.min() < info.min or values.max() > info.max):
            raise ValueError("Integer labels must fit in int32 to be vectorized.")
        return values.astype(np.int32)
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32)
    raise TypeError(f"Cannot vectorize a mask of dtype {values.dtype}.")


def _polygonize(
    values: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    *,
    connectivity: int,
) -> Iterator[tuple[BaseGeometry, float]]:
    """Yield ``(polygon, value)`` for every connected region of ``valid``."""
    from rasterio import features

    shapely_geometry = _require_vector("shapely.geometry", "vectorize")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")
    for geom, value in features.shapes(
        _to_shapes_dtype(values),
        mask=valid,
        connectivity=connectivity,
        transform=transform,
    ):
        yield shapely_geometry.shape(geom), value


def vectorize(
    mask: xr.DataArray,
    *,
    min_area: float = 0.0,
    connectivity: int = 4,
    attr_name: str = "value",
) -> gpd.GeoDataFrame:
    """Polygonize a label / binary mask.

    Every connected region of equal, non-zero, finite value becomes one
    polygon (zeros and NaNs are background). Pixel edges are kept as-is
    — no simplification — so ``rasterize_like(vectorize(m), m)``
    reproduces ``m`` exactly.

    Args:
        mask: 2-D ``(y, x)`` integer / boolean / float raster with
            spatial coordinates (and ideally a CRS via ``mask.rio.crs``).
            A single-variable ``Dataset`` is also accepted.
        min_area: Drop polygons whose area is below this, in CRS units².
            ``0`` disables filtering.
        connectivity: Pixel connectivity, ``4`` (edges) or ``8`` (edges
            and corners).
        attr_name: Name of the column holding each polygon's mask value.

    Returns:
        ``GeoDataFrame`` with one row per polygon, the pixel value in
        ``attr_name`` (cast back to the mask dtype) and
        ``crs=mask.rio.crs``.

    Raises:
        ImportError: If the ``vector`` extra (geopandas/shapely) is missing.
        ValueError: If ``mask`` is not 2-D or ``connectivity`` ∉ {4, 8}.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import vectorize
        >>> m = xr.DataArray(
        ...     np.array([[0, 1, 1], [0, 1, 1], [2, 0, 0]], dtype="uint8"),
        ...     dims=("y", "x"),
        ...     coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
        ... ).rio.write_crs("EPSG:32630")
        >>> gdf = vectorize(m)
        >>> gdf["value"].tolist(), gdf.area.tolist(), gdf.crs.to_epsg()
        ([1, 2], [4.0, 1.0], 32630)
    """
    gpd = _require_vector("geopandas", "vectorize")
    da = _as_raster_2d(mask, feature="vectorize")
    values = np.asarray(da.values)
    valid = values != 0
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)

    geoms, vals = [], []
    for geom, value in _polygonize(
        values, valid, da.rio.transform(recalc=True), connectivity=connectivity
    ):
        if min_area > 0 and geom.area < min_area:
            continue
        geoms.append(geom)
        vals.append(value)
    return gpd.GeoDataFrame(
        {attr_name: np.asarray(vals, dtype=values.dtype)},
        geometry=geoms,
        crs=da.rio.crs,
    )


def transform_polygon(geometry: BaseGeometry, src_crs: Any, dst_crs: Any) -> Any:
    """Reproject a shapely geometry between CRSs.

    Wraps :func:`rasterio.warp.transform_geom` (GDAL/OGR under the hood),
    so vertices are transformed exactly — no densification. Equivalent
    CRSs short-circuit and return ``geometry`` unchanged.

    Args:
        geometry: Any shapely geometry, in ``src_crs``.
        src_crs: Source CRS — anything :class:`pyproj.CRS` accepts.
        dst_crs: Destination CRS — anything :class:`pyproj.CRS` accepts.

    Returns:
        The geometry (same type) in ``dst_crs``.

    Example:
        >>> from shapely.geometry import box
        >>> from xrtoolz.geo import transform_polygon
        >>> geom = box(-3.0, 40.0, -2.9, 40.1)
        >>> utm = transform_polygon(geom, "EPSG:4326", "EPSG:32630")
        >>> back = transform_polygon(utm, "EPSG:32630", "EPSG:4326")
        >>> [round(v, 6) for v in back.bounds]
        [-3.0, 40.0, -2.9, 40.1]
    """
    from rasterio.warp import transform_geom

    shapely_geometry = _require_vector("shapely.geometry", "transform_polygon")
    src, dst = CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs)
    if src == dst:
        return geometry
    return shapely_geometry.shape(
        transform_geom(src.to_wkt(), dst.to_wkt(), shapely_geometry.mapping(geometry))
    )


__all__ = ["transform_polygon", "vectorize"]
