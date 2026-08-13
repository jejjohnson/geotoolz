"""Unstructured points → gridded value resampling."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from jaxtyping import Float
from scipy.stats import binned_statistic_2d
from sklearn.neighbors import KernelDensity

from xrtoolz.interpolate._src.binning import Grid
from xrtoolz.interpolate._src.spatial import _bandwidth_rule, _to_metric_xy
from xrtoolz.utils._src.finite import _finite_mask


BandwidthRule = Literal["scott", "silverman"]
KernelName = Literal[
    "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
]
MetricName = Literal["euclidean", "haversine"]
AlgorithmName = Literal["auto", "kd_tree", "ball_tree"]
OutputMode = Literal["density", "counts", "counts_per_area"]


def points_to_grid(
    lons: Float[np.ndarray, "n"],
    lats: Float[np.ndarray, "n"],
    values: Float[np.ndarray, "n"],
    grid: Grid,
    statistic: str = "mean",
) -> xr.DataArray:
    """Bin raw (lon, lat, value) tuples onto ``grid``.

    Thin wrapper around :func:`scipy.stats.binned_statistic_2d` that
    doesn't require constructing a scattered DataArray first.
    """
    finite = np.isfinite(values)
    lon_edges, lat_edges = grid.bin_edges()
    stat, _, _, _ = binned_statistic_2d(
        np.ravel(lons)[finite],
        np.ravel(lats)[finite],
        np.ravel(values)[finite],
        statistic=statistic,
        bins=[lon_edges, lat_edges],
    )
    return xr.DataArray(
        data=stat.T,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
    )


def kde_to_grid(
    lons: Float[np.ndarray, "n"],
    lats: Float[np.ndarray, "n"],
    grid: Grid,
    *,
    weights: Float[np.ndarray, "n"] | None = None,
    bandwidth: float | BandwidthRule = "scott",
    kernel: KernelName = "gaussian",
    metric: MetricName = "euclidean",
    algorithm: AlgorithmName = "auto",
    output: OutputMode = "density",
    rtol: float = 1e-4,
) -> xr.DataArray:
    """Evaluate a KDE of scattered lon/lat points on ``grid``.

    Args:
        lons: 1-D point longitudes. Non-finite entries are dropped.
        lats: 1-D point latitudes. Non-finite entries are dropped.
        grid: Target :class:`Grid`.
        weights: Optional per-point weights passed to
            :class:`sklearn.neighbors.KernelDensity`. Non-finite weights are
            dropped with their corresponding lon/lat points.
        bandwidth: Positive float, or ``"scott"`` / ``"silverman"``. For
            ``metric="haversine"``, float bandwidths are radians.
        kernel: sklearn KDE kernel name.
        metric: ``"euclidean"`` treats lon/lat as planar; ``"haversine"``
            converts lon/lat to radians and uses great-circle distances.
        algorithm: sklearn tree backend. ``"haversine"`` uses ``"ball_tree"``.
        output: Output normalization. See semantics below.
        rtol: Relative tolerance for sklearn tree pruning.

    Output modes:
        ``"density"`` is a probability density that integrates to 1 over the
        grid (per unit area). For ``metric="haversine"`` the output is
        renormalized to integrate to 1 on the sphere using
        ``cos(lat) * dlat * dlon`` (steradian) cell areas, since sklearn's
        ``KernelDensity`` only normalizes correctly for the Euclidean metric.

        ``"counts_per_area"`` is the expected per-unit-area count
        (``density * n_eff``). Integrating over the grid recovers the input
        point count (or weight sum). Comparable across grids of different
        resolutions.

        ``"counts"`` is the expected per-cell count
        (``density * n_eff * cell_area``). Summing over the grid recovers
        the input point count (or weight sum). Scales with cell size, so it
        is only comparable across grids of identical resolution.

    Returns:
        DataArray on ``grid`` with ``("lat", "lon")`` dimensions.
    """
    if output not in ("density", "counts", "counts_per_area"):
        raise ValueError(f"unknown output mode {output!r}")
    if metric not in ("euclidean", "haversine"):
        raise ValueError(f"unknown metric {metric!r}")
    if grid.lon.size < 2 or grid.lat.size < 2:
        raise ValueError(
            "kde_to_grid requires grid.lon and grid.lat to each have at least "
            f"2 points; got {grid.lon.size} lon, {grid.lat.size} lat"
        )

    lon_values = np.ravel(np.asarray(lons, dtype=float))
    lat_values = np.ravel(np.asarray(lats, dtype=float))
    if lon_values.shape != lat_values.shape:
        raise ValueError("lons and lats must have the same shape")

    finite = _finite_mask(lon_values, lat_values)
    if weights is None:
        sample_weight = None
    else:
        weight_values = np.ravel(np.asarray(weights, dtype=float))
        if weight_values.shape != lon_values.shape:
            raise ValueError("weights must have the same shape as lons and lats")
        finite &= _finite_mask(weight_values)
        sample_weight = weight_values[finite]

    pts = np.column_stack([lon_values[finite], lat_values[finite]])
    if pts.shape[0] < 2:
        raise ValueError("KDE requires at least 2 finite points")

    if metric == "haversine":
        fit_pts = _to_metric_xy(pts[:, 0], pts[:, 1], "haversine")
        algorithm = "ball_tree"
    else:
        fit_pts = pts

    if isinstance(bandwidth, str):
        bandwidth_value = _bandwidth_rule(fit_pts, bandwidth)
    else:
        bandwidth_value = float(bandwidth)
        if bandwidth_value <= 0 or not np.isfinite(bandwidth_value):
            raise ValueError(f"bandwidth must be positive, got {bandwidth!r}")

    kde = KernelDensity(
        bandwidth=bandwidth_value,
        kernel=kernel,
        metric=metric,
        algorithm=algorithm,
        rtol=rtol,
    ).fit(fit_pts, sample_weight=sample_weight)

    lon_grid, lat_grid = np.meshgrid(grid.lon, grid.lat, indexing="xy")
    if metric == "haversine":
        queries = _to_metric_xy(lon_grid.ravel(), lat_grid.ravel(), "haversine")
    else:
        queries = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    density = np.exp(kde.score_samples(queries)).reshape(len(grid.lat), len(grid.lon))

    # Cell area: deg² for Euclidean, steradian (cos(lat)*dlat*dlon, radians²)
    # for haversine. sklearn normalizes assuming flat (lat, lon) integration in
    # haversine mode, so we re-normalize on the sphere to recover ∫p dA = 1.
    if metric == "haversine":
        dlat = float(np.mean(np.abs(np.diff(np.deg2rad(np.asarray(grid.lat))))))
        dlon = float(np.mean(np.abs(np.diff(np.deg2rad(np.asarray(grid.lon))))))
        cell_area = (
            np.cos(np.deg2rad(np.asarray(grid.lat, dtype=float)))[:, None] * dlat * dlon
        )
        total = float((density * cell_area).sum())
        if total > 0:
            density = density / total
    else:
        dlat = float(np.mean(np.abs(np.diff(np.asarray(grid.lat, dtype=float)))))
        dlon = float(np.mean(np.abs(np.diff(np.asarray(grid.lon, dtype=float)))))
        cell_area = np.full(density.shape, dlat * dlon)

    n_eff = pts.shape[0] if sample_weight is None else float(sample_weight.sum())
    if output == "density":
        data = density
    elif output == "counts_per_area":
        data = density * n_eff
    else:  # "counts" — per-cell expected count
        data = density * n_eff * cell_area

    return xr.DataArray(
        data=data,
        dims=("lat", "lon"),
        coords={"lon": grid.lon, "lat": grid.lat},
    )
