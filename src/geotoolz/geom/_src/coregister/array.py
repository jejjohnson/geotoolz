"""Tier-A array primitives for cross-modality coregistration.

The two-tier discipline (``_src/array.py`` pure numpy / scipy; the
sibling ``_src/operators.py`` carrier-aware) mirrors the rest of
geotoolz. Operators wrap these primitives and add GeoTensor /
GeoDataFrame / xvec metadata handling.

Scaffolding — all primitives below raise `NotImplementedError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    import numpy as np
    from affine import Affine
    from jaxtyping import Float, Num


def reproject_like(
    src_values: Num[np.ndarray, "*batch h w"],
    src_transform: Affine,
    src_crs: Any,
    *,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: Any,
    resampling: Literal[
        "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average"
    ] = "bilinear",
) -> Num[np.ndarray, "*batch dst_h dst_w"]:
    """Reproject ``src`` onto the destination grid in one shot.

    Composes the (reproject CRS) + (resample to dst pixel grid)
    steps that ``Reproject`` + ``Resample`` would do back-to-back.
    A single call lets the underlying rasterio warp pipeline pick
    a more efficient path than two sequential warps.

    Args:
        src_values: Source pixel values, ``(H, W)`` or ``(..., H, W)``.
        src_transform: Affine transform of the source grid.
        src_crs: CRS of the source grid.
        dst_shape: Destination spatial shape ``(H', W')``.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        resampling: Rasterio resampling kernel name.

    Returns:
        The warped values on the destination grid, same leading dims
        as ``src_values``.

    Raises:
        NotImplementedError: Always — Phase 3 scaffolding.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def swath_to_grid(
    swath_values: Num[np.ndarray, "*batch h w"],
    *,
    lat: Float[np.ndarray, "h w"],
    lon: Float[np.ndarray, "h w"],
    target_crs: Any,
    target_res: tuple[float, float],
    bounds: tuple[float, float, float, float] | None = None,
    method: Literal["bowtie_aware", "naive"] = "bowtie_aware",
) -> tuple[Num[np.ndarray, "*batch dst_h dst_w"], Affine]:
    """Project a swath product (per-pixel lat/lon) onto a regular grid.

    The "bowtie_aware" path handles MODIS/VIIRS scan-edge pixel
    growth that a naive nearest-neighbour resample double-counts.

    Args:
        swath_values: Sensor-space values, ``(H, W)`` or ``(..., H, W)``.
        lat: Per-pixel latitudes, shape ``(H, W)``.
        lon: Per-pixel longitudes, shape ``(H, W)``.
        target_crs: CRS of the output grid.
        target_res: ``(x_res, y_res)`` in destination units.
        bounds: Optional ``(xmin, ymin, xmax, ymax)`` clip.
        method: ``"bowtie_aware"`` (default) or ``"naive"``.

    Returns:
        ``(grid_values, dst_transform)`` ready for GeoTensor
        construction.

    Raises:
        NotImplementedError: Always — Phase 3 scaffolding.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def points_to_raster_binned(
    points_xy: Float[np.ndarray, "n 2"],
    point_values: Num[np.ndarray, " n"],
    *,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
) -> Float[np.ndarray, "h w"]:
    """Bin scattered point measurements onto a regular grid.

    Uses ``scipy.stats.binned_statistic_2d`` under the hood.

    Args:
        points_xy: Point coordinates as an ``(N, 2)`` XY array in the
            destination grid's CRS.
        point_values: Parallel ``(N,)`` array of measurement values.
        dst_shape: Destination spatial shape ``(H, W)``.
        dst_transform: Affine transform of the destination grid.
        stat: Per-cell reduction statistic.

    Returns:
        The binned ``(H, W)`` grid; cells with no points are ``NaN``.

    Raises:
        NotImplementedError: Always — Phase 3 scaffolding.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def raster_to_point_cloud(
    raster_values: Num[np.ndarray, "*batch h w"],
    raster_transform: Affine,
    cloud_xy: Float[np.ndarray, "n 2"],
    *,
    k: int = 1,
    max_radius: float | None = None,
    method: Literal["nearest", "bilinear", "idw"] = "nearest",
) -> Float[np.ndarray, "*batch n"]:
    """Sample raster values onto each point in a point cloud.

    Args:
        raster_values: Raster pixel values, ``(H, W)`` or ``(..., H, W)``.
        raster_transform: Affine transform of the raster grid.
        cloud_xy: Point coordinates as an ``(N, 2)`` XY array in the
            raster's CRS.
        k: Number of nearest pixels per point (``k > 1`` requires
            ``method="idw"``).
        max_radius: Optional distance ceiling in CRS units; points
            farther than this from any pixel centre get ``NaN``.
        method: ``"nearest"``, ``"bilinear"``, or ``"idw"``.

    Returns:
        Sampled values, ``(N,)`` for 2-D input or ``(..., N)`` for
        multi-band input.

    Raises:
        NotImplementedError: Always — Phase 3 scaffolding.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")
