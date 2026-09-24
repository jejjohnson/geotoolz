"""Raster extent → polygon: bounding and valid-pixel footprints.

The extent is read with :mod:`rioxarray` (``ds.rio.transform()``,
``ds.rio.width`` / ``height``, ``ds.rio.crs``, ``da.rio.nodata``); the
valid-pixel outline is polygonized with :func:`rasterio.features.shapes`.
Mirrors ``georeader.GeoTensor.footprint`` / ``valid_footprint``; these
are the geometries a catalog / STAC item needs.

``shapely`` is imported lazily — install it with
``pip install 'xrtoolz[vector]'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr

from xrtoolz.geo._src.vectorize import _polygonize, _require_vector, transform_polygon


if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon


def _to_crs(geometry: Any, ds: xr.Dataset | xr.DataArray, crs: Any) -> Any:
    if crs is None:
        return geometry
    if ds.rio.crs is None:
        raise ValueError(
            "Cannot reproject the footprint: the input has no CRS "
            "(attach one with xrtoolz.geo.assign_crs)."
        )
    return transform_polygon(geometry, ds.rio.crs, crs)


def footprint(ds: xr.Dataset | xr.DataArray, crs: Any = None) -> Polygon:
    """Bounding polygon of the raster extent.

    Built from the four outer pixel corners of ``ds.rio.transform()``, so
    it covers whole pixels (not just pixel centres) and follows the grid
    even for rotated transforms. Nodata pixels are included — see
    :func:`valid_footprint` for the data-only outline.

    Args:
        ds: Raster with spatial coordinates (``ds.rio.x_dim`` /
            ``ds.rio.y_dim``).
        crs: Output CRS. ``None`` keeps the dataset CRS (``ds.rio.crs``);
            otherwise the polygon's vertices are reprojected with
            :func:`transform_polygon`.

    Returns:
        Five-vertex (closed) ``shapely.Polygon``.

    Raises:
        ValueError: If ``crs`` is given but ``ds`` has no CRS.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import footprint
        >>> da = xr.DataArray(
        ...     np.ones((2, 3)),
        ...     dims=("y", "x"),
        ...     coords={"y": [15.0, 5.0], "x": [5.0, 15.0, 25.0]},
        ... ).rio.write_crs("EPSG:32630")
        >>> footprint(da).bounds
        (0.0, 0.0, 30.0, 20.0)
    """
    shapely_geometry = _require_vector("shapely.geometry", "footprint")
    transform = ds.rio.transform(recalc=True)
    width, height = ds.rio.width, ds.rio.height
    # Counter-clockwise for a north-up grid, matching georeader's vertex order.
    corners = [(0, 0), (0, height), (width, height), (width, 0)]
    polygon = shapely_geometry.Polygon([transform * c for c in corners])
    return _to_crs(polygon, ds, crs)


def _valid_mask(
    da: xr.DataArray,
    nodata: float | None,
    method: Literal["all", "any"],
) -> xr.DataArray:
    """``(y, x)`` boolean mask of pixels that are finite and not ``nodata``."""
    y_dim, x_dim = da.rio.y_dim, da.rio.x_dim
    valid = da.notnull()
    fill = da.rio.nodata if nodata is None else nodata
    if fill is not None and not np.isnan(fill):
        valid &= da != fill
    extra = [d for d in valid.dims if d not in (y_dim, x_dim)]
    if extra:
        valid = valid.all(extra) if method == "all" else valid.any(extra)
    return valid.transpose(y_dim, x_dim)


def valid_footprint(
    ds: xr.Dataset | xr.DataArray,
    *,
    crs: Any = None,
    nodata: float | None = None,
    method: Literal["all", "any"] = "all",
    connectivity: int = 4,
) -> Polygon | MultiPolygon:
    """Polygon enclosing the valid (finite, non-nodata) pixels.

    Builds a boolean valid-mask, polygonizes it at pixel resolution and
    dissolves the pieces into a single (Multi)Polygon — the catalog /
    STAC geometry of a granule, which unlike :func:`footprint` excludes
    nodata fill.

    Args:
        ds: Raster. For a ``Dataset`` every data variable spanning both
            spatial dims contributes; extra dims (``band``, ``time``, …)
            are reduced with ``method``.
        crs: Output CRS. ``None`` keeps the dataset CRS.
        nodata: Value marking invalid pixels. ``None`` uses each
            variable's ``rio.nodata``. NaN is always invalid.
        method: ``"all"`` — a pixel is valid only if valid in every
            variable / band / time step; ``"any"`` — valid in at least one.
        connectivity: Pixel connectivity for polygonization, ``4`` or ``8``.

    Returns:
        ``Polygon`` if the valid pixels form one region, else
        ``MultiPolygon``.

    Raises:
        ValueError: If there are no valid pixels, no variable spans the
            spatial dims, ``method`` is unknown, or ``crs`` is given but
            ``ds`` has no CRS.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import valid_footprint
        >>> data = np.full((4, 4), -9999.0)
        >>> data[1:3, 1:3] = 1.0
        >>> da = xr.DataArray(
        ...     data,
        ...     dims=("y", "x"),
        ...     coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]},
        ... ).rio.write_crs("EPSG:32630").rio.write_nodata(-9999.0)
        >>> valid_footprint(da).bounds
        (1.0, 1.0, 3.0, 3.0)
    """
    shapely = _require_vector("shapely", "valid_footprint")
    if method not in ("all", "any"):
        raise ValueError(f"method must be 'all' or 'any', got {method!r}.")
    y_dim, x_dim = ds.rio.y_dim, ds.rio.x_dim
    arrays = (
        [ds]
        if isinstance(ds, xr.DataArray)
        else [v for v in ds.data_vars.values() if {y_dim, x_dim} <= set(v.dims)]
    )
    if not arrays:
        raise ValueError(f"No data variable spans the spatial dims ({y_dim}, {x_dim}).")
    masks = [_valid_mask(da, nodata, method) for da in arrays]
    combine = np.logical_and if method == "all" else np.logical_or
    valid = np.asarray(combine.reduce([m.values for m in masks]))
    if not valid.any():
        raise ValueError("valid_footprint: the raster has no valid pixels.")

    polygons = [
        geom
        for geom, _ in _polygonize(
            valid.astype(np.uint8),
            valid,
            ds.rio.transform(recalc=True),
            connectivity=connectivity,
        )
    ]
    return _to_crs(shapely.unary_union(polygons), ds, crs)


__all__ = ["footprint", "valid_footprint"]
