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


def _require_axis_aligned(transform: Any, op_name: str) -> None:
    """Reject affine transforms with rotation / shear terms.

    ``_pixel_center_coords`` and the bin-edge logic in
    `PointsToRaster` derive ``x``/``y`` from only the ``a/c`` and
    ``e/f`` affine terms. For non-axis-aligned grids (``b != 0`` or
    ``d != 0``) that's silently wrong — the world coordinates would
    drop the rotation/shear contribution. Fail loudly upfront
    rather than emit misregistered output.
    """
    if transform.b != 0.0 or transform.d != 0.0:
        raise ValueError(
            f"{op_name} requires an axis-aligned affine "
            "(rotation/shear terms b and d must be 0); got "
            f"b={transform.b!r}, d={transform.d!r}. Reproject to an "
            "axis-aligned grid first (e.g. via "
            "`gz.geom.Reproject(dst_crs=...)`)."
        )


def _pixel_center_coords(
    transform: Any, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(x_centers, y_centers)`` for a raster from its affine.

    Returns 1-D arrays of length ``W`` and ``H`` respectively — the
    pixel-center coordinates suitable for an xarray-backed extraction.
    Requires an axis-aligned affine (no rotation/shear) — call
    ``_require_axis_aligned`` first.
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

    _require_axis_aligned(tensor.transform, "RasterToPoints")
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
    * `list[shapely.Point]` / any iterable of geometries (empty
      list is fine — produces an empty extraction).
    * `geopandas.GeoSeries` / `GeoDataFrame` (uses `.geometry`)
    * `xarray.DataArray` with an xvec geometry index (uses
      ``.xvec.geom_coords``)

    Only `Point` geometries are accepted at the output — passing
    LineStrings or Polygons silently fails downstream on `.x`/`.y`
    access, so we validate the type here.
    """
    import xarray as xr

    geoms: Sequence[shapely.geometry.base.BaseGeometry]
    if isinstance(points, xr.DataArray):
        # xarray with xvec geometry index.
        coords = points.xvec.geom_coords
        if not coords:
            raise ValueError(
                "RasterToPoints needs `points` to be either a sequence "
                "of shapely Points, a geopandas GeoSeries/GeoDataFrame, "
                "or an xarray.DataArray with an xvec geometry index. The "
                "DataArray you passed has no xvec geometry index."
            )
        name = next(iter(coords))
        geoms = list(points[name].values)
    elif hasattr(points, "geometry") and not hasattr(points, "xvec"):
        # geopandas.GeoSeries / GeoDataFrame.
        geoms = list(points.geometry)
    elif hasattr(points, "__iter__"):
        # Generic iterable. Empty is fine — return [] without
        # falling through to the "unsupported" branch.
        try:
            geoms = list(points)
        except TypeError as exc:
            raise TypeError(
                f"Unsupported `points` input: {type(points).__name__}. "
                "Pass a list of shapely Points, a GeoSeries/GeoDataFrame, "
                "or an xvec-indexed xarray DataArray."
            ) from exc
        if geoms and not hasattr(geoms[0], "geom_type"):
            raise TypeError(
                "Iterable `points` must contain shapely geometries; got "
                f"{type(geoms[0]).__name__}."
            )
    else:
        raise TypeError(
            f"Unsupported `points` input: {type(points).__name__}. "
            "Pass a list of shapely Points, a GeoSeries/GeoDataFrame, "
            "or an xvec-indexed xarray DataArray."
        )

    # Type guard: extract_points and our bilinear path both need
    # `.x` / `.y`, which only `Point` provides. Reject non-Point
    # geometries with a clear message instead of an AttributeError
    # buried in xvec/numpy frames.
    for g in geoms:
        if g.geom_type != "Point":
            raise ValueError(
                "RasterToPoints / PointsToRaster require Point geometries; "
                f"got {g.geom_type!r}. Use `.centroid` upfront if you want "
                "to sample at polygon centers."
            )
    return geoms


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

    The ``points`` input accepts:

    * a `geopandas.GeoSeries` / `GeoDataFrame` (uses ``.geometry``;
      requires the ``attribute`` argument naming the column to bin).
    * an `xarray.DataArray` with an xvec geometry index (values
      taken from the array itself, no ``attribute`` needed).

    A bare ``(geoms, values)`` form is **not** supported — wrap in
    a `GeoDataFrame` or DataArray upfront. Like `RasterToPoints`,
    only `Point` geometries are accepted; non-axis-aligned target
    rasters are rejected.

    Args:
        method: Reduction strategy — ``"binned_stat"`` (scipy
            ``binned_statistic_2d``, default) or ``"idw"`` (deferred).
        stat: For ``"binned_stat"``: one of ``"mean"``, ``"median"``,
            ``"sum"``, ``"count"``, ``"max"``, ``"min"``.
        attribute: For GeoDataFrame inputs, the column to bin.
            Required for GeoDataFrame; ignored for xvec-indexed
            DataArray.
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

        # Reject non-axis-aligned target rasters — bin edges derived
        # from only a/c and e/f would silently drop rotation/shear.
        _require_axis_aligned(like.transform, "PointsToRaster")

        # 1) Resolve geometries + parallel values array. CRS sanity
        # check happens inside `_points_with_values` for GeoDataFrame
        # inputs where the .crs attribute is set.
        xs, ys, values = _points_with_values(
            points, attribute=self.attribute, like_crs=like.crs
        )

        # 2) Build bin edges from the `like` raster's affine. The
        # affine maps pixel (col, row) corners → CRS coords; for an
        # H x W grid we need W+1 x edges and H+1 y edges.
        h, w = like.shape[-2], like.shape[-1]
        x_edges = like.transform.c + like.transform.a * np.arange(w + 1)
        y_edges = like.transform.f + like.transform.e * np.arange(h + 1)
        # binned_statistic_2d requires monotonically-increasing edges.
        # rasterio's "north-up" affine has negative `e` (rows go
        # south-to-north backwards), which makes y_edges decrease;
        # "west-up" rasters with negative `a` likewise make x_edges
        # decrease. Sort each axis independently and flip the
        # corresponding output dimension so the result keeps the
        # raster convention (row 0 = top, col 0 = left).
        flip_y = y_edges[0] > y_edges[-1]
        flip_x = x_edges[0] > x_edges[-1]
        y_edges_sorted = y_edges[::-1] if flip_y else y_edges
        x_edges_sorted = x_edges[::-1] if flip_x else x_edges

        # 3) Run the bin.
        stat_out, _, _, _ = binned_statistic_2d(
            xs, ys, values, statistic=self.stat, bins=[x_edges_sorted, y_edges_sorted]
        )
        # `binned_statistic_2d` returns shape (n_x_bins, n_y_bins);
        # transpose to (H, W) raster convention, with rows = y.
        result = stat_out.T
        if flip_y:
            result = result[::-1, :]
        if flip_x:
            result = result[:, ::-1]

        # 4) Wrap back into a GeoTensor on the `like` grid.
        return like.array_as_geotensor(result.astype(np.float64))


def _points_with_values(
    points: Any, *, attribute: str | None, like_crs: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parallel ``(xs, ys, values)`` from common point inputs.

    Helper for `PointsToRaster`. Resolves geometries to coordinate
    arrays and pulls the values either from the input's value array
    (xarray) or from a named attribute (GeoDataFrame). If
    ``like_crs`` is given and the input carries a CRS, mismatches
    raise rather than silently producing misregistered output.
    """
    import xarray as xr

    if isinstance(points, xr.DataArray):
        try:
            coords = points.xvec.geom_coords
        except AttributeError as exc:
            raise ImportError(
                "PointsToRaster with an xarray DataArray input requires "
                "the `[vector-cube]` extra (xvec). Install with "
                "`pip install 'geotoolz[vector-cube]'`."
            ) from exc
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
        # CRS sanity: a silent mismatch would put points in the
        # wrong cells. Validate when both sides have a CRS set.
        src_crs = getattr(points, "crs", None)
        if src_crs is not None and like_crs is not None:
            import pyproj

            src = pyproj.CRS.from_user_input(src_crs)
            dst = pyproj.CRS.from_user_input(like_crs)
            if src != dst:
                raise ValueError(
                    "PointsToRaster: input GeoDataFrame CRS "
                    f"{src.to_string()!r} differs from the target raster CRS "
                    f"{dst.to_string()!r}. Reproject the input first "
                    "(e.g. `gdf.to_crs(like.crs)`); silent reprojection is "
                    "intentionally disabled to avoid hidden coordinate drift."
                )
    else:
        raise TypeError(
            "PointsToRaster expects either an xvec-indexed xarray DataArray "
            "or a geopandas GeoDataFrame with an `attribute` column. Got "
            f"{type(points).__name__}."
        )

    # Validate Point types — non-Point geometries would silently
    # fail later on `.x`/`.y` access.
    for g in geoms:
        if g.geom_type != "Point":
            raise ValueError(
                f"PointsToRaster requires Point geometries; got {g.geom_type!r}."
            )
    xs = np.asarray([g.x for g in geoms], dtype=np.float64)
    ys = np.asarray([g.y for g in geoms], dtype=np.float64)
    return xs, ys, values


class RasterToPointCloud(Operator):
    """Sample raster values onto each node of a point cloud.

    A point cloud here is an ``(N, 2)`` numpy array of XY coordinates
    in the raster's CRS (Z columns, if present, are ignored —
    sampling is 2-D). For shapely-Point inputs use `RasterToPoints`;
    the distinction is the input form, not the underlying op.

    Args:
        k: Number of nearest pixels per point. ``k=1`` is plain
            nearest-neighbour (matches `RasterToPoints` with
            ``extract='nearest'``). ``k>1`` requires ``method="idw"``
            since "nearest" / "bilinear" are k=1 concepts.
        max_radius: Optional distance ceiling in CRS units. Points
            farther than this from any pixel center get ``NaN``.
            ``None`` disables the radius gate.
        method: ``"nearest"`` (k=1; KDTree single-NN), ``"bilinear"``
            (k=1; bilinear interp via xarray.interp), or ``"idw"``
            (k>=1; inverse-distance weighted mean of the k nearest).
    """

    def __init__(
        self,
        *,
        k: int = 1,
        max_radius: float | None = None,
        method: Literal["nearest", "bilinear", "idw"] = "nearest",
        power: float = 2.0,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if method in {"nearest", "bilinear"} and k != 1:
            raise ValueError(
                f"method={method!r} only supports k=1; got k={k}. "
                "Use method='idw' for k>1."
            )
        self.k = k
        self.max_radius = max_radius
        self.method = method
        self.power = power

    def get_config(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "max_radius": self.max_radius,
            "method": self.method,
            "power": self.power,
        }

    def __call__(self, raster: GeoTensor, cloud: Any) -> Any:
        _require_axis_aligned(raster.transform, "RasterToPointCloud")
        xy = _cloud_to_xy_array(cloud)
        if xy.size == 0:
            # Empty cloud → empty result with the right last-axis shape.
            arr = np.asarray(raster)
            if arr.ndim == 2:
                return np.zeros((0,), dtype=arr.dtype)
            return np.zeros((arr.shape[0], 0), dtype=arr.dtype)

        if self.method == "bilinear":
            return _raster_to_cloud_bilinear(raster, xy, self.max_radius)
        # Nearest + IDW both rely on a KDTree built on pixel centers.
        return _raster_to_cloud_kdtree(
            raster,
            xy,
            k=self.k,
            max_radius=self.max_radius,
            method=self.method,
            power=self.power,
        )


def _cloud_to_xy_array(cloud: Any) -> np.ndarray:
    """Normalize the cloud input to an ``(N, 2)`` XY float array.

    Accepts:
    * numpy array of shape ``(N, 2)`` or ``(N, 3)`` (Z ignored)
    * geopandas GeoDataFrame / GeoSeries of Points
    * any iterable of shapely Points
    """
    if isinstance(cloud, np.ndarray):
        if cloud.ndim != 2 or cloud.shape[1] < 2:
            raise ValueError(
                "RasterToPointCloud expects cloud as an (N, 2) or (N, 3) "
                f"ndarray; got shape {cloud.shape}."
            )
        return cloud[:, :2].astype(np.float64, copy=False)
    if hasattr(cloud, "geometry"):
        geoms = list(cloud.geometry)
    elif hasattr(cloud, "__iter__"):
        geoms = list(cloud)
    else:
        raise TypeError(
            f"Unsupported cloud input: {type(cloud).__name__}. "
            "Pass an (N, 2) ndarray, a GeoSeries / GeoDataFrame of Points, "
            "or a sequence of shapely Points."
        )
    for g in geoms:
        if getattr(g, "geom_type", None) != "Point":
            raise ValueError(
                "RasterToPointCloud requires Point geometries; "
                f"got {getattr(g, 'geom_type', type(g).__name__)!r}."
            )
    return np.asarray([[g.x, g.y] for g in geoms], dtype=np.float64).reshape(-1, 2)


def _raster_to_cloud_bilinear(
    raster: GeoTensor, xy: np.ndarray, max_radius: float | None
) -> np.ndarray:
    """Bilinear sample of `raster` at every point in `xy`."""
    da = _geotensor_to_dataarray(raster)
    point_xs = xy[:, 0]
    point_ys = xy[:, 1]
    interp = da.interp(x=("point", point_xs), y=("point", point_ys))
    values = np.asarray(interp.values)
    if max_radius is not None:
        # Use KDTree to compute nearest-pixel distance; mask out
        # points whose nearest pixel sits outside the radius.
        _, distances = _nearest_pixel_distances(raster, xy)
        mask = distances > max_radius
        if values.ndim == 1:
            values = values.astype(np.float64, copy=True)
            values[mask] = np.nan
        else:
            values = values.astype(np.float64, copy=True)
            values[:, mask] = np.nan
    return values


def _raster_to_cloud_kdtree(
    raster: GeoTensor,
    xy: np.ndarray,
    *,
    k: int,
    max_radius: float | None,
    method: Literal["nearest", "bilinear", "idw"],
    power: float,
) -> np.ndarray:
    """KDTree-based sampling: nearest (k=1) or IDW (k>=1)."""
    from scipy.spatial import KDTree

    arr = np.asarray(raster)
    x_centers, y_centers = _pixel_center_coords(raster.transform, raster.shape)
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    pixel_xy = np.column_stack([xx.ravel(), yy.ravel()])
    tree = KDTree(pixel_xy)

    distances, indices = tree.query(xy, k=k)
    # `KDTree.query` returns scalars / 1D arrays for k=1; promote
    # to (N, 1) so the downstream IDW code path is uniform.
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Flatten the raster's spatial dims for fancy-index lookup.
    if arr.ndim == 2:
        flat = arr.reshape(-1)
        gathered = flat[indices]  # (N, k)
    else:
        # 3-D (C, H, W) → (C, H*W); gather along last axis.
        c = arr.shape[0]
        flat = arr.reshape(c, -1)
        gathered = flat[:, indices]  # (C, N, k)

    if method in {"nearest"} or (method == "idw" and k == 1):
        if arr.ndim == 2:
            result = gathered[:, 0].astype(np.float64)
        else:
            result = gathered[:, :, 0].astype(np.float64)
    else:
        # IDW with k >= 2.
        weights = 1.0 / np.maximum(distances**power, 1e-12)
        if arr.ndim == 2:
            num = (gathered * weights).sum(axis=1)
            den = weights.sum(axis=1)
            result = (num / den).astype(np.float64)
        else:
            num = (gathered * weights[None, :, :]).sum(axis=2)
            den = weights.sum(axis=1)[None, :]
            result = (num / den).astype(np.float64)

    # Out-of-radius gate: use the *closest* pixel's distance.
    if max_radius is not None:
        nearest_dist = distances[:, 0]
        out = nearest_dist > max_radius
        if arr.ndim == 2:
            result[out] = np.nan
        else:
            result[:, out] = np.nan
    return result


def _nearest_pixel_distances(
    raster: GeoTensor, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Helper: (indices, distances) of the single nearest pixel per point."""
    from scipy.spatial import KDTree

    x_centers, y_centers = _pixel_center_coords(raster.transform, raster.shape)
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    pixel_xy = np.column_stack([xx.ravel(), yy.ravel()])
    tree = KDTree(pixel_xy)
    distances, indices = tree.query(xy, k=1)
    return indices, distances


class PointCloudToRaster(Operator):
    """Rasterize a point cloud onto a target grid.

    Numpy-first counterpart to `PointsToRaster`. The cloud input is
    a tuple ``(xy_array, values_array)`` so callers don't need to
    wrap their raw arrays in a GeoDataFrame.

    Args:
        method: ``"binned_stat"`` (scipy ``binned_statistic_2d`` —
            fast, no smoothing) or ``"idw"`` (KDTree-backed inverse-
            distance weighting, smoother but slower).
        stat: For ``"binned_stat"``: one of ``"mean"``, ``"median"``,
            ``"sum"``, ``"count"``, ``"max"``, ``"min"``.
        power: For ``"idw"``: inverse-distance exponent.
        k: For ``"idw"``: how many nearest points contribute to each
            pixel. Defaults to 8 (a reasonable balance between
            smoothness and locality).
        max_radius: For ``"idw"``: distance ceiling in CRS units.
            Pixels with no points within this radius receive NaN.
    """

    def __init__(
        self,
        *,
        method: Literal["binned_stat", "idw"] = "binned_stat",
        stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
        power: float = 2.0,
        k: int = 8,
        max_radius: float | None = None,
    ) -> None:
        if method not in {"binned_stat", "idw"}:
            raise ValueError(
                f"PointCloudToRaster.method must be 'binned_stat' or 'idw'; "
                f"got {method!r}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self.method = method
        self.stat = stat
        self.power = power
        self.k = k
        self.max_radius = max_radius

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "stat": self.stat,
            "power": self.power,
            "k": self.k,
            "max_radius": self.max_radius,
        }

    def __call__(self, cloud: Any, like: GeoTensor) -> GeoTensor:
        _require_axis_aligned(like.transform, "PointCloudToRaster")
        xy, values = _cloud_to_xy_values(cloud)
        if self.method == "binned_stat":
            return _point_cloud_binned_stat(xy, values, like, stat=self.stat)
        return _point_cloud_idw(
            xy,
            values,
            like,
            power=self.power,
            k=self.k,
            max_radius=self.max_radius,
        )


def _cloud_to_xy_values(cloud: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalise the cloud input to ``(xy, values)`` parallel arrays.

    Accepts:
    * a tuple ``(xy_ndarray, values_ndarray)`` — the canonical form
    * a structured numpy array with fields ``x``, ``y``, ``value``
    """
    if isinstance(cloud, tuple) and len(cloud) == 2:
        xy, values = cloud
        xy_arr = np.asarray(xy, dtype=np.float64)
        if xy_arr.ndim != 2 or xy_arr.shape[1] < 2:
            raise ValueError(
                "PointCloudToRaster expects xy as an (N, 2) ndarray; "
                f"got shape {xy_arr.shape}."
            )
        values_arr = np.asarray(values, dtype=np.float64)
        if values_arr.ndim != 1 or values_arr.shape[0] != xy_arr.shape[0]:
            raise ValueError(
                "PointCloudToRaster values must be 1-D with the same N as "
                f"xy; got xy.shape={xy_arr.shape}, values.shape={values_arr.shape}."
            )
        return xy_arr[:, :2], values_arr
    if isinstance(cloud, np.ndarray) and cloud.dtype.names is not None:
        # Structured array: pull `x`, `y`, `value` fields.
        missing = [n for n in ("x", "y", "value") if n not in cloud.dtype.names]
        if missing:
            raise ValueError(
                "PointCloudToRaster structured-array input must have "
                f"fields x, y, value; missing {missing!r}."
            )
        xy = np.column_stack([cloud["x"], cloud["y"]]).astype(np.float64)
        values = np.asarray(cloud["value"], dtype=np.float64)
        return xy, values
    raise TypeError(
        "PointCloudToRaster expects cloud as `(xy_ndarray, values_ndarray)` "
        "or a structured ndarray with fields {x, y, value}; got "
        f"{type(cloud).__name__}."
    )


def _point_cloud_binned_stat(
    xy: np.ndarray, values: np.ndarray, like: GeoTensor, *, stat: str
) -> GeoTensor:
    """Bin via scipy.stats.binned_statistic_2d (shares logic with PointsToRaster)."""
    from scipy.stats import binned_statistic_2d

    h, w = like.shape[-2], like.shape[-1]
    x_edges = like.transform.c + like.transform.a * np.arange(w + 1)
    y_edges = like.transform.f + like.transform.e * np.arange(h + 1)
    flip_y = y_edges[0] > y_edges[-1]
    flip_x = x_edges[0] > x_edges[-1]
    y_edges_sorted = y_edges[::-1] if flip_y else y_edges
    x_edges_sorted = x_edges[::-1] if flip_x else x_edges

    stat_out, _, _, _ = binned_statistic_2d(
        xy[:, 0],
        xy[:, 1],
        values,
        statistic=stat,
        bins=[x_edges_sorted, y_edges_sorted],
    )
    result = stat_out.T
    if flip_y:
        result = result[::-1, :]
    if flip_x:
        result = result[:, ::-1]
    return like.array_as_geotensor(result.astype(np.float64))


def _point_cloud_idw(
    xy: np.ndarray,
    values: np.ndarray,
    like: GeoTensor,
    *,
    power: float,
    k: int,
    max_radius: float | None,
) -> GeoTensor:
    """KDTree-based inverse-distance weighting onto the `like` grid."""
    from scipy.spatial import KDTree

    h, w = like.shape[-2], like.shape[-1]
    x_centers, y_centers = _pixel_center_coords(like.transform, like.shape)
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    pixel_xy = np.column_stack([xx.ravel(), yy.ravel()])  # (H*W, 2)

    tree = KDTree(xy)
    eff_k = min(k, xy.shape[0])
    distances, indices = tree.query(pixel_xy, k=eff_k)
    if eff_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    sampled = values[indices]  # (H*W, eff_k)
    weights = 1.0 / np.maximum(distances**power, 1e-12)
    num = (sampled * weights).sum(axis=1)
    den = weights.sum(axis=1)
    result = (num / den).reshape(h, w).astype(np.float64)

    if max_radius is not None:
        # Mask pixels whose nearest point exceeds the radius.
        nearest_dist = distances[:, 0].reshape(h, w)
        result = np.where(nearest_dist > max_radius, np.nan, result)
    return like.array_as_geotensor(result)


class VectorToRasterAgg(Operator):
    """Rasterize a vector layer with an aggregation policy for overlaps.

    Extends `geom.Rasterize` (which burns features in order and lets
    later ones overwrite earlier ones) with a per-pixel aggregation
    when multiple features overlap a pixel — useful for mean of an
    attribute, count of features per pixel, etc.

    Implementation strategy: rasterize each feature separately to a
    boolean mask (O(H*W) per feature, single accumulator pass) and
    update a running aggregator. This is more memory-friendly than
    stacking N feature rasters and reducing.

    Args:
        agg: One of ``"mean"``, ``"count"``, ``"sum"``, ``"max"``,
            ``"min"``, ``"first"``, ``"last"``. ``"majority"`` is
            deferred (raises at call time so YAML round-trip works).
        attribute: Column name to aggregate. Required for
            ``mean / sum / max / min / first / last``; ignored for
            ``count``.
        all_touched: If ``True``, every pixel whose envelope is
            touched by the geometry counts (rasterio default). If
            ``False`` (default), only pixels whose centre falls
            inside.
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
        valid = {"mean", "majority", "count", "sum", "max", "min", "first", "last"}
        if agg not in valid:
            raise ValueError(
                f"VectorToRasterAgg.agg must be one of {sorted(valid)!r}; got {agg!r}"
            )
        if agg != "count" and agg != "majority" and attribute is None:
            raise ValueError(
                f"VectorToRasterAgg(agg={agg!r}) requires `attribute` naming "
                "the column to aggregate."
            )
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
        if self.agg == "majority":
            raise NotImplementedError(
                "VectorToRasterAgg(agg='majority') not yet implemented; "
                "the others (mean / count / sum / max / min / first / last) "
                "are wired up."
            )
        _require_axis_aligned(like.transform, "VectorToRasterAgg")
        import geopandas as gpd
        from rasterio.features import rasterize as rio_rasterize

        if not isinstance(vector, gpd.GeoDataFrame):
            raise TypeError(
                "VectorToRasterAgg expects a geopandas.GeoDataFrame; got "
                f"{type(vector).__name__}."
            )
        if vector.crs is not None and like.crs is not None:
            import pyproj as _pp

            src = _pp.CRS.from_user_input(vector.crs)
            dst = _pp.CRS.from_user_input(like.crs)
            if src != dst:
                raise ValueError(
                    f"VectorToRasterAgg: vector CRS {src.to_string()!r} differs "
                    f"from raster CRS {dst.to_string()!r}. Reproject the input "
                    "first (e.g. `gdf.to_crs(like.crs)`)."
                )

        h, w = like.shape[-2], like.shape[-1]
        # Per-feature mask burned at value=1 using the like raster's affine.
        # For pixel-centre semantics use all_touched=False; for envelope-
        # overlap use all_touched=True.

        # Pull attribute values once (None for count / majority).
        attr_values = (
            vector[self.attribute].to_numpy() if self.attribute is not None else None
        )

        # Initialise the accumulator(s).
        sum_arr = (
            np.zeros((h, w), dtype=np.float64) if self.agg in {"mean", "sum"} else None
        )
        count_arr = np.zeros((h, w), dtype=np.int64)
        max_arr = (
            np.full((h, w), -np.inf, dtype=np.float64) if self.agg == "max" else None
        )
        min_arr = (
            np.full((h, w), np.inf, dtype=np.float64) if self.agg == "min" else None
        )
        # For "first" / "last": keep an array of attribute values written
        # in order. "first" walks the GDF in reverse so the *first* feature
        # writes last and wins; "last" walks forwards so the *last* feature
        # writes last and wins.
        ordinal_arr = (
            np.full((h, w), np.nan, dtype=np.float64)
            if self.agg in {"first", "last"}
            else None
        )

        iterator = (
            zip(
                reversed(range(len(vector))),
                reversed(list(vector.geometry)),
                strict=True,
            )
            if self.agg == "first"
            else zip(range(len(vector)), vector.geometry, strict=True)
        )

        for idx, geom in iterator:
            mask = rio_rasterize(
                [(geom, 1)],
                out_shape=(h, w),
                transform=like.transform,
                all_touched=self.all_touched,
                dtype=np.uint8,
            ).astype(bool)
            if not mask.any():
                continue
            count_arr += mask.astype(np.int64)
            if self.agg == "count":
                continue
            v = float(attr_values[idx]) if attr_values is not None else 1.0
            if self.agg in {"sum", "mean"}:
                assert sum_arr is not None
                sum_arr[mask] += v
            elif self.agg == "max":
                assert max_arr is not None
                np.maximum(max_arr, np.where(mask, v, -np.inf), out=max_arr)
            elif self.agg == "min":
                assert min_arr is not None
                np.minimum(min_arr, np.where(mask, v, np.inf), out=min_arr)
            elif self.agg in {"first", "last"}:
                assert ordinal_arr is not None
                ordinal_arr[mask] = v

        # Build the result per agg policy.
        if self.agg == "count":
            result = count_arr.astype(np.float64)
        elif self.agg == "sum":
            assert sum_arr is not None
            result = sum_arr
            # Pixels with no features: NaN (sum of nothing is undefined
            # in the aggregate-attribute sense).
            result = np.where(count_arr > 0, result, np.nan)
        elif self.agg == "mean":
            assert sum_arr is not None
            result = np.where(count_arr > 0, sum_arr / np.maximum(count_arr, 1), np.nan)
        elif self.agg == "max":
            assert max_arr is not None
            result = np.where(count_arr > 0, max_arr, np.nan)
        elif self.agg == "min":
            assert min_arr is not None
            result = np.where(count_arr > 0, min_arr, np.nan)
        else:  # first / last
            assert ordinal_arr is not None
            result = ordinal_arr

        return like.array_as_geotensor(result)
