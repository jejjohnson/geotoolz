"""Tier-A array primitives for cross-modality coregistration.

The two-tier discipline (``_src/array.py`` pure numpy / scipy; the
sibling ``_src/operators.py`` carrier-aware) mirrors the rest of
geotoolz. Operators wrap these primitives and add GeoTensor /
GeoDataFrame / xvec metadata handling.

Scaffolding — all primitives below raise `NotImplementedError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    import numpy as np


def reproject_like(
    src_values: np.ndarray,
    src_transform,
    src_crs,
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Literal[
        "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average"
    ] = "bilinear",
) -> np.ndarray:
    """Reproject ``src`` onto the destination grid in one shot.

    Composes the (reproject CRS) + (resample to dst pixel grid)
    steps that ``Reproject`` + ``Resample`` would do back-to-back.
    A single call lets the underlying rasterio warp pipeline pick
    a more efficient path than two sequential warps.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def swath_to_grid(
    swath_values: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    target_crs,
    target_res: tuple[float, float],
    bounds: tuple[float, float, float, float] | None = None,
    method: Literal["bowtie_aware", "naive"] = "bowtie_aware",
) -> tuple[np.ndarray, object]:
    """Project a swath product (per-pixel lat/lon) onto a regular grid.

    The "bowtie_aware" path handles MODIS/VIIRS scan-edge pixel
    growth that a naive nearest-neighbour resample double-counts.

    Returns:
        ``(grid_values, dst_transform)`` ready for GeoTensor
        construction.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def points_to_raster_binned(
    points_xy: np.ndarray,
    point_values: np.ndarray,
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
) -> np.ndarray:
    """Bin scattered point measurements onto a regular grid.

    Uses ``scipy.stats.binned_statistic_2d`` under the hood.
    """
    raise NotImplementedError("Phase 3 PR — see design §5.3.")


def raster_to_point_cloud(
    raster_values: np.ndarray,
    raster_transform,
    cloud_xy: np.ndarray,
    *,
    k: int = 1,
    max_radius: float | None = None,
    method: Literal["nearest", "bilinear", "idw"] = "nearest",
) -> np.ndarray:
    """Sample raster values onto each point in a point cloud."""
    raise NotImplementedError("Phase 3 PR — see design §5.3.")
