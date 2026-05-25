"""Tier-B Operators — cross-modality coregistration.

Each operator wraps either a primitive in
``geotoolz.geom._src.coregister.array`` or an existing geom op
(``Reproject`` / ``Resample`` / ``Rasterize``) and pipes through the
carrier-aware metadata handling on either side.

Operators that pin a concrete ``GeoTensor`` / ``GeoDataFrame`` /
``xvec.DataArray`` in their constructor (anything ``*Like``, anything
taking a sample point cloud) set ``forbid_in_yaml = True`` so
hydra-zen `builds()` does not try to serialize the reference — same
discipline as the existing ``ReprojectLike`` / ``RasterizeLike``.

See ``docs/design/query-matchup.md`` §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pipekit import Operator

from geotoolz.geom._src.operators import ReprojectLike


if TYPE_CHECKING:
    from collections.abc import Sequence

    import shapely.geometry.base
    from georeader.geotensor import GeoTensor


def _pixel_center_coords(
    transform: Any, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(x_centers, y_centers)`` for a raster from its affine.

    Returns 1-D arrays of length ``W`` and ``H`` respectively — the
    pixel-center coordinates suitable for an xarray-backed extraction.
    """
    h, w = shape[-2], shape[-1]
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    return x, y


def _geotensor_to_dataarray(tensor: GeoTensor) -> Any:
    """Convert a `GeoTensor` to an `xarray.DataArray` with proper coords.

    The result has ``x`` / ``y`` 1-D coords at pixel centers and a
    ``band`` dim if the input is 3-D ``(C, H, W)``. The CRS isn't
    attached here — callers pass it as a kwarg to `extract_points`
    so we don't need a full xvec geometry index on the raster side.
    """
    import xarray as xr

    arr = np.asarray(tensor)
    x, y = _pixel_center_coords(tensor.transform, tensor.shape)
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=("y", "x"), coords={"x": x, "y": y})
    if arr.ndim == 3:
        bands = np.arange(arr.shape[0])
        return xr.DataArray(
            arr,
            dims=("band", "y", "x"),
            coords={"band": bands, "x": x, "y": y},
        )
    raise ValueError(
        "RasterToPoints expects 2-D (H, W) or 3-D (C, H, W) GeoTensor input; "
        f"got ndim={arr.ndim}."
    )


def _resolve_geometries(points: Any) -> Sequence[shapely.geometry.base.BaseGeometry]:
    """Pull the geometry array out of common point inputs.

    Accepts:
    * `list[shapely.Point]` / any iterable of geometries
    * `geopandas.GeoSeries` / `GeoDataFrame` (uses `.geometry`)
    * `xarray.DataArray` with an xvec geometry index (uses
      ``.xvec.geom_coords``)
    """
    # Fast path: already a sequence of shapely geometries.
    if hasattr(points, "__iter__") and not hasattr(points, "geometry"):
        try:
            seq = list(points)
        except TypeError:
            seq = None
        if seq is not None and seq and hasattr(seq[0], "geom_type"):
            return seq

    # geopandas.GeoSeries / GeoDataFrame.
    if hasattr(points, "geometry") and not hasattr(points, "xvec"):
        return list(points.geometry)

    # xarray with xvec geometry index.
    import xarray as xr

    if isinstance(points, xr.DataArray):
        # `geom_coords` is a Mapping[name, IndexVariable]; one entry
        # in the typical single-axis case.
        coords = points.xvec.geom_coords
        if not coords:
            raise ValueError(
                "RasterToPoints needs `points` to be either a sequence "
                "of shapely geometries, a geopandas GeoSeries/GeoDataFrame, "
                "or an xarray.DataArray with an xvec geometry index. The "
                "DataArray you passed has no xvec geometry index."
            )
        # Return the geometry array from the first geometry-indexed coord.
        name = next(iter(coords))
        return list(points[name].values)

    raise TypeError(
        f"Unsupported `points` input: {type(points).__name__}. "
        "Pass a list of shapely Points, a GeoSeries/GeoDataFrame, "
        "or an xvec-indexed xarray DataArray."
    )


class RasterToRasterLike(Operator):
    """Align one raster onto another's CRS + grid in a single op.

    Thin binary wrapper around `geom.ReprojectLike` — the underlying
    rasterio warp handles both CRS reprojection and target-grid
    resampling in one pass. The wrapper exists because
    `ReprojectLike` pins ``like`` at construction (unary call),
    whereas matchup coregistration needs both inputs at call time
    to match the ``(secondary_raw, primary_patch) -> aligned_secondary``
    shape of `MatchedField.coreg`.

    Useful as a default in ``MatchedField.coreg`` for raster↔raster
    pairs where the secondary needs to be coregistered to the
    primary's grid before stacking.

    Args:
        resampling: One of ``"nearest"``, ``"bilinear"``, ``"cubic"``,
            ``"cubic_spline"``, ``"lanczos"``, ``"average"``,
            ``"mode"``.

    Examples:
        >>> import geotoolz as gz
        >>> coreg = gz.geom.coregister.RasterToRasterLike(resampling="bilinear")
        >>> aligned_s2 = coreg(s2_chip, modis_chip)
        >>> # `aligned_s2` has MODIS's CRS, transform, and spatial shape.
    """

    def __init__(self, *, resampling: str = "bilinear") -> None:
        self.resampling = resampling

    def get_config(self) -> dict[str, Any]:
        return {"resampling": self.resampling}

    def __call__(self, src: GeoTensor, like: GeoTensor) -> GeoTensor:
        # Delegate to the existing single-input `ReprojectLike` so
        # we share the rasterio warp code path. The price is one
        # extra Python-level Operator construction per call; the
        # warp itself dominates, so it's negligible.
        return ReprojectLike(like=like, resampling=self.resampling)(src)


class SwathToGrid(Operator):
    """Project a LEO swath product onto a regular grid.

    Handles MODIS/VIIRS bowtie growth in the default ``method``;
    the ``"naive"`` mode is faster but double-counts scan-edge pixels.

    Args:
        target_crs: Target CRS (string accepted by ``pyproj``).
        target_res: ``(x_res, y_res)`` in destination units.
        method: ``"bowtie_aware"`` (default) or ``"naive"``.
        bounds: Optional ``(xmin, ymin, xmax, ymax)`` clip.
    """

    def __init__(
        self,
        *,
        target_crs: str,
        target_res: tuple[float, float],
        method: Literal["bowtie_aware", "naive"] = "bowtie_aware",
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.target_crs = target_crs
        self.target_res = target_res
        self.method = method
        self.bounds = bounds

    def get_config(self) -> dict[str, Any]:
        return {
            "target_crs": self.target_crs,
            "target_res": list(self.target_res),
            "method": self.method,
            "bounds": list(self.bounds) if self.bounds is not None else None,
        }

    def __call__(self, swath: GeoTensor) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class GridToSwath(Operator):
    """Resample a regular grid (typically GEO) onto a swath's acquisition geometry.

    Time-matched: each pixel of the output corresponds to the
    swath's acquisition time at that pixel; the input grid is sampled
    at the nearest available frame within ``dt_max``.

    Args:
        time_match: How to pick the source frame; ``"nearest"`` only
            for v1.
        dt_max: Maximum allowed time offset.
    """

    def __init__(
        self,
        *,
        time_match: Literal["nearest"] = "nearest",
        dt_max: str = "15min",
    ) -> None:
        self.time_match = time_match
        self.dt_max = dt_max

    def get_config(self) -> dict[str, Any]:
        return {"time_match": self.time_match, "dt_max": self.dt_max}

    def __call__(self, grid_series: GeoTensor, swath_like: GeoTensor) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class RasterToPoints(Operator):
    """Extract raster values at a vector cube of points.

    Requires the ``[vector-cube]`` extra (``xvec``).

    The ``points`` input accepts three forms:

    * a sequence of `shapely.Point` (or any iterable of shapely
      geometries),
    * a `geopandas.GeoSeries` / `GeoDataFrame` (uses ``.geometry``),
    * an `xarray.DataArray` with an xvec geometry index (uses
      ``.xvec.geom_coords``).

    The return value is an `xarray.DataArray` indexed by a
    ``geometry`` dim, with the raster's values sampled at each
    point. For 3-D ``(C, H, W)`` rasters the result carries an
    additional ``band`` dim.

    Args:
        extract: Interpolation mode — ``"nearest"`` (default; uses
            xvec's per-pixel nearest sampling) or ``"bilinear"``
            (uses xarray's linear interpolation on the raster
            DataArray before extraction).
        out_var: Name attached to the output DataArray.
    """

    def __init__(
        self,
        *,
        extract: Literal["nearest", "bilinear"] = "nearest",
        out_var: str = "value",
    ) -> None:
        if extract not in {"nearest", "bilinear"}:
            raise ValueError(
                f"RasterToPoints.extract must be 'nearest' or 'bilinear'; got {extract!r}"
            )
        self.extract = extract
        self.out_var = out_var

    def get_config(self) -> dict[str, Any]:
        return {"extract": self.extract, "out_var": self.out_var}

    def __call__(self, raster: GeoTensor, points: Any) -> Any:
        try:
            import xvec  # noqa: F401 — registers the .xvec accessor
        except ImportError as exc:
            raise ImportError(
                "RasterToPoints requires the `[vector-cube]` extra (xvec). "
                "Install with `pip install 'geotoolz[vector-cube]'`."
            ) from exc

        da = _geotensor_to_dataarray(raster)
        geoms = _resolve_geometries(points)
        crs = str(raster.crs)

        if self.extract == "bilinear":
            # xarray's `interp` does linear interpolation in 1-D per
            # axis; combining x + y gives bilinear. Apply before
            # extract_points so the points sample the interpolated
            # raster.
            point_xs = np.asarray([g.x for g in geoms])
            point_ys = np.asarray([g.y for g in geoms])
            interp = da.interp(x=("geometry", point_xs), y=("geometry", point_ys))
            result = interp.rename(self.out_var)
            # Attach the geometry array as a coord so the output
            # matches the nearest-mode shape (geometry-indexed).

            result = result.assign_coords(geometry=("geometry", list(geoms)))
            return result

        # Nearest: defer to xvec.extract_points.
        result = da.xvec.extract_points(geoms, x_coords="x", y_coords="y", crs=crs)
        return result.rename(self.out_var)


class PointsToRaster(Operator):
    """Bin a vector cube of point measurements onto a target raster grid.

    Implementation uses ``scipy.stats.binned_statistic_2d`` with the
    ``like`` raster's pixel-edge grid as bins. The ``"idw"`` method
    (inverse-distance weighting via KDTree) is reserved for a
    follow-up; today only ``"binned_stat"`` is implemented and
    construction with ``method="idw"`` raises NotImplementedError
    at call time so YAML round-trip still works.

    The ``points`` input accepts the same forms as `RasterToPoints`:

    * a sequence of `shapely.Point` paired with a values array,
    * a `geopandas.GeoSeries` / `GeoDataFrame` (uses ``.geometry``
      + an attribute column),
    * an `xarray.DataArray` with an xvec geometry index.

    Args:
        method: Reduction strategy — ``"binned_stat"`` (scipy
            ``binned_statistic_2d``, default) or ``"idw"`` (deferred).
        stat: For ``"binned_stat"``: one of ``"mean"``, ``"median"``,
            ``"sum"``, ``"count"``, ``"max"``, ``"min"``.
        attribute: For non-DataArray inputs, the column / attribute
            name to bin. Ignored when ``points`` is already a 1-D
            xarray DataArray of values indexed by geometry.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(
        self,
        *,
        method: Literal["binned_stat", "idw"] = "binned_stat",
        stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
        attribute: str | None = None,
    ) -> None:
        if method not in {"binned_stat", "idw"}:
            raise ValueError(
                f"PointsToRaster.method must be 'binned_stat' or 'idw'; got {method!r}"
            )
        self.method = method
        self.stat = stat
        self.attribute = attribute

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "stat": self.stat,
            "attribute": self.attribute,
        }

    def __call__(self, points: Any, like: GeoTensor) -> GeoTensor:
        if self.method == "idw":
            raise NotImplementedError(
                "PointsToRaster(method='idw') is not yet implemented; "
                "use method='binned_stat' for now."
            )
        from scipy.stats import binned_statistic_2d

        # 1) Resolve geometries + parallel values array.
        xs, ys, values = _points_with_values(points, attribute=self.attribute)

        # 2) Build bin edges from the `like` raster's affine. The
        # affine maps pixel (col, row) corners → CRS coords; for an
        # H x W grid we need W+1 x edges and H+1 y edges.
        h, w = like.shape[-2], like.shape[-1]
        x_edges = like.transform.c + like.transform.a * np.arange(w + 1)
        y_edges = like.transform.f + like.transform.e * np.arange(h + 1)
        # binned_statistic_2d requires monotonically-increasing edges;
        # rasterio's "north-up" affine has negative e, so y_edges
        # decrease. Flip both edges and the resulting statistic
        # so the output's row 0 stays the top of the image.
        flip_y = y_edges[0] > y_edges[-1]
        y_edges_sorted = y_edges[::-1] if flip_y else y_edges

        # 3) Run the bin.
        stat_out, _, _, _ = binned_statistic_2d(
            xs, ys, values, statistic=self.stat, bins=[x_edges, y_edges_sorted]
        )
        # `binned_statistic_2d` returns shape (n_x_bins, n_y_bins);
        # transpose to (H, W) raster convention, with rows = y.
        result = stat_out.T
        if flip_y:
            result = result[::-1, :]

        # 4) Wrap back into a GeoTensor on the `like` grid.
        return like.array_as_geotensor(result.astype(np.float64))


def _points_with_values(
    points: Any, *, attribute: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parallel ``(xs, ys, values)`` from common point inputs.

    Helper for `PointsToRaster`. Resolves geometries to coordinate
    arrays and pulls the values either from the input's value array
    (xarray) or from a named attribute (GeoDataFrame).
    """
    import xarray as xr

    if isinstance(points, xr.DataArray):
        coords = points.xvec.geom_coords
        if not coords:
            raise ValueError(
                "PointsToRaster needs the xarray DataArray to have an "
                "xvec geometry index. Set one with "
                "`.xvec.set_geom_indexes(...)` before calling."
            )
        name = next(iter(coords))
        geoms = list(points[name].values)
        values = np.asarray(points.values, dtype=np.float64)
    elif hasattr(points, "geometry") and not hasattr(points, "xvec"):
        geoms = list(points.geometry)
        if attribute is None:
            raise ValueError(
                "PointsToRaster on a GeoDataFrame input requires "
                "`attribute=...` naming the column to bin."
            )
        values = np.asarray(points[attribute].values, dtype=np.float64)
    else:
        raise TypeError(
            "PointsToRaster expects either an xvec-indexed xarray DataArray "
            "or a geopandas GeoDataFrame with an `attribute` column. Got "
            f"{type(points).__name__}."
        )

    xs = np.asarray([g.x for g in geoms], dtype=np.float64)
    ys = np.asarray([g.y for g in geoms], dtype=np.float64)
    return xs, ys, values


class RasterToPointCloud(Operator):
    """Sample raster values onto each node of a point cloud.

    A point cloud here is an ``(N, 2)`` or ``(N, 3)`` array of XY(Z)
    coordinates in the raster's CRS, with an optional attribute table.

    Args:
        k: Number of nearest pixels to consider (``k=1`` is plain
            nearest-neighbour).
        max_radius: Optional distance ceiling. Points beyond this
            radius receive NaN.
        method: ``"nearest"``, ``"bilinear"``, or ``"idw"`` (when
            ``k > 1``).
    """

    def __init__(
        self,
        *,
        k: int = 1,
        max_radius: float | None = None,
        method: Literal["nearest", "bilinear", "idw"] = "nearest",
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self.k = k
        self.max_radius = max_radius
        self.method = method

    def get_config(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "max_radius": self.max_radius,
            "method": self.method,
        }

    def __call__(self, raster: GeoTensor, cloud: Any) -> Any:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class PointCloudToRaster(Operator):
    """Rasterize a point cloud onto a target grid.

    Args:
        method: ``"binned_stat"`` (fast, no smoothing) or ``"idw"``
            (smooth, KDTree-backed).
        stat: For ``"binned_stat"``: ``"mean"``, ``"median"``,
            ``"sum"``, ``"count"``, ``"max"``, ``"min"``.
        power: For ``"idw"``: inverse-distance exponent.
    """

    def __init__(
        self,
        *,
        method: Literal["binned_stat", "idw"] = "binned_stat",
        stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
        power: float = 2.0,
    ) -> None:
        self.method = method
        self.stat = stat
        self.power = power

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "stat": self.stat, "power": self.power}

    def __call__(self, cloud: Any, like: GeoTensor) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class VectorToRasterAgg(Operator):
    """Rasterize a vector layer with an aggregation policy for overlaps.

    Extends the existing ``Rasterize`` (which burns vector geometries
    onto a grid) with a per-pixel aggregation when multiple features
    overlap a pixel — mean of attribute, majority class, count, etc.

    Args:
        agg: ``"mean"``, ``"majority"``, ``"count"``, ``"sum"``,
            ``"max"``, ``"min"``, ``"first"``, ``"last"``.
        attribute: Column name to aggregate (None for boolean burn).
        all_touched: If ``True``, every pixel touched by the geometry
            counts; otherwise only pixels whose centre falls inside.
    """

    def __init__(
        self,
        *,
        agg: Literal[
            "mean", "majority", "count", "sum", "max", "min", "first", "last"
        ] = "mean",
        attribute: str | None = None,
        all_touched: bool = False,
    ) -> None:
        self.agg = agg
        self.attribute = attribute
        self.all_touched = all_touched

    def get_config(self) -> dict[str, Any]:
        return {
            "agg": self.agg,
            "attribute": self.attribute,
            "all_touched": self.all_touched,
        }

    def __call__(self, vector: Any, like: GeoTensor) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")
