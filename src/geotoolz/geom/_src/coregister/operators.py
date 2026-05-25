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

from pipekit import Operator

from geotoolz.geom._src.operators import ReprojectLike


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class RasterToRasterLike(Operator):
    """Reproject + resample one raster onto another's grid.

    Convenience that bundles ``Reproject + ResampleLike`` into a
    single matchup-shaped op: two inputs at call time, one aligned
    output out. Useful as a default in ``MatchedField.coreg`` for
    raster↔raster pairs where the secondary needs to be coregistered
    to the primary's grid before stacking.

    Distinct from `geom.ReprojectLike`, which pins ``like`` at
    construction (unary call). `RasterToRasterLike` takes both
    inputs at call time, matching the
    ``(secondary_raw, primary_patch) -> aligned_secondary`` shape
    of `MatchedField.coreg`.

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
        # we share the warp / resample code path. The price is one
        # extra Python-level Operator construction per call; the
        # underlying rasterio warp dominates, so it's negligible.
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

    Requires the ``[vector-cube]`` extra (``xvec``). Input ``points``
    is an ``xvec.DataArray`` of point geometries (e.g. AERONET
    stations, buoy positions); output is the same vector cube with
    a new variable holding the sampled raster values.

    Args:
        extract: Interpolation mode — ``"nearest"`` (default) or
            ``"bilinear"``.
        out_var: Name of the new variable on the vector cube.
    """

    def __init__(
        self,
        *,
        extract: Literal["nearest", "bilinear"] = "nearest",
        out_var: str = "value",
    ) -> None:
        self.extract = extract
        self.out_var = out_var

    def get_config(self) -> dict[str, Any]:
        return {"extract": self.extract, "out_var": self.out_var}

    def __call__(self, raster: GeoTensor, points: Any) -> Any:
        raise NotImplementedError("Phase 3 PR — see design §5.")


class PointsToRaster(Operator):
    """Bin a vector cube of point measurements onto a target raster grid.

    Requires the ``[vector-cube]`` extra (``xvec``).

    Args:
        method: Reduction strategy — ``"binned_stat"`` (scipy) or
            ``"idw"`` (inverse-distance weighting via KDTree).
        stat: For ``"binned_stat"``: ``"mean"``, ``"median"``, ``"sum"``,
            ``"count"``, ``"max"``, ``"min"``.
        like_required: Whether ``like`` (target grid) is required at
            call time; if ``False``, ``__call__`` accepts a CRS +
            resolution instead. Default ``True`` for the typical
            matchup case.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(
        self,
        *,
        method: Literal["binned_stat", "idw"] = "binned_stat",
        stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
        like_required: bool = True,
    ) -> None:
        self.method = method
        self.stat = stat
        self.like_required = like_required

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "stat": self.stat,
            "like_required": self.like_required,
        }

    def __call__(self, points: Any, like: GeoTensor) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")


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
