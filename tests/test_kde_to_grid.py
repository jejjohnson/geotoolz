from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.interpolate import Grid, KDEToGrid, kde_to_grid


def _cell_area(grid: Grid) -> float:
    return float(np.mean(np.diff(grid.lon)) * np.mean(np.diff(grid.lat)))


def test_kde_to_grid_density_integrates_to_one() -> None:
    grid = Grid.from_bounds((-8.0, 8.0), (-8.0, 8.0), resolution=0.25)
    out = kde_to_grid(
        [-1.0, 1.0],
        [0.0, 0.0],
        grid,
        bandwidth=0.5,
        output="density",
    )

    assert out.dims == ("lat", "lon")
    assert np.isclose(float((out * _cell_area(grid)).sum()), 1.0, rtol=2e-3)


def test_kde_to_grid_counts_mode_sums_to_sample_count() -> None:
    # "counts" returns per-cell expected counts; summing recovers n_eff.
    grid = Grid.from_bounds((-8.0, 8.0), (-8.0, 8.0), resolution=0.25)
    out = kde_to_grid(
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        grid,
        bandwidth=0.5,
        output="counts",
    )

    assert np.isclose(float(out.sum()), 3.0, rtol=2e-3)


def test_kde_to_grid_counts_per_area_integrates_to_sample_count() -> None:
    # "counts_per_area" returns per-area density; ∫ over grid recovers n_eff.
    grid = Grid.from_bounds((-8.0, 8.0), (-8.0, 8.0), resolution=0.25)
    out = kde_to_grid(
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        grid,
        bandwidth=0.5,
        output="counts_per_area",
    )

    assert np.isclose(float((out * _cell_area(grid)).sum()), 3.0, rtol=2e-3)


def test_kde_to_grid_counts_per_area_invariant_under_resolution() -> None:
    # Per-area density should not scale with cell size.
    pts = ([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    coarse = Grid.from_bounds((-6.0, 6.0), (-6.0, 6.0), resolution=0.5)
    fine = Grid.from_bounds((-6.0, 6.0), (-6.0, 6.0), resolution=0.25)
    a = kde_to_grid(*pts, coarse, bandwidth=0.5, output="counts_per_area")
    b = kde_to_grid(*pts, fine, bandwidth=0.5, output="counts_per_area")
    # Peaks should match at the same coordinate to within KDE quadrature error.
    assert np.isclose(
        float(a.sel(lat=0.0, lon=-1.0)),
        float(b.sel(lat=0.0, lon=-1.0)),
        rtol=5e-2,
    )


def test_kde_to_grid_haversine_density_integrates_to_one_on_sphere() -> None:
    # Sphere-renormalization: ∫ density · cos(lat) dlat dlon = 1 (radians).
    grid = Grid(
        lon=np.linspace(-180.0, 180.0, 73, endpoint=False),
        lat=np.linspace(-89.0, 89.0, 90),
    )
    out = kde_to_grid(
        [0.0, 90.0, -90.0],
        [10.0, -10.0, 30.0],
        grid,
        bandwidth=0.4,
        metric="haversine",
        output="density",
    )
    dlat = float(np.mean(np.abs(np.diff(np.deg2rad(grid.lat)))))
    dlon = float(np.mean(np.abs(np.diff(np.deg2rad(grid.lon)))))
    cos_lat = np.cos(np.deg2rad(np.asarray(grid.lat)))
    cell_area = cos_lat[:, None] * dlat * dlon
    surface_integral = float((out * cell_area).sum())
    assert np.isclose(surface_integral, 1.0, rtol=5e-2)


def test_kde_to_grid_rejects_short_grid_axes() -> None:
    grid = Grid(lon=np.array([0.0]), lat=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least 2 points"):
        kde_to_grid([0.0, 0.5], [0.0, 0.5], grid, bandwidth=0.5)


def test_kde_to_grid_bandwidth_rules_are_finite_and_equal_in_2d() -> None:
    grid = Grid.from_bounds((-3.0, 3.0), (-3.0, 3.0), resolution=0.5)
    lons = np.array([-1.0, -0.5, 0.25, 1.5])
    lats = np.array([0.0, 0.75, -0.25, 1.0])

    explicit = kde_to_grid(lons, lats, grid, bandwidth=0.5)
    scott = kde_to_grid(lons, lats, grid, bandwidth="scott")
    silverman = kde_to_grid(lons, lats, grid, bandwidth="silverman")

    assert np.isfinite(explicit).all()
    assert np.isfinite(scott).all()
    assert np.isfinite(silverman).all()
    assert not np.allclose(explicit, scott)
    xr.testing.assert_allclose(scott, silverman)


@pytest.mark.parametrize(
    "kernel",
    ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"],
)
def test_kde_to_grid_kernel_dispatch(kernel: str) -> None:
    grid = Grid.from_bounds((-3.0, 3.0), (-3.0, 3.0), resolution=1.0)
    out = kde_to_grid(
        [-0.1, 0.1],
        [0.0, 0.0],
        grid,
        bandwidth=0.5,
        kernel=kernel,
    )

    assert np.isfinite(out).all()
    if kernel in {"tophat", "epanechnikov", "linear", "cosine"}:
        assert out.sel(lon=3.0, lat=3.0).item() == 0.0


def test_kde_to_grid_haversine_reduces_pole_longitude_bias() -> None:
    grid = Grid(lon=np.array([89.0, 90.0]), lat=np.array([0.0, 89.0]))
    lons = np.array([0.0, 180.0, 0.0, 180.0])
    lats = np.array([0.0, 0.0, 89.0, 89.0])

    spherical = kde_to_grid(
        lons,
        lats,
        grid,
        bandwidth=0.1,
        metric="haversine",
    )
    # Spherical KDE at the pole sees the four scattered points as nearby; the
    # equator point at the same longitude should be relatively quieter.
    assert spherical.sel(lon=90.0, lat=89.0) > spherical.sel(lon=90.0, lat=0.0)


def test_kde_to_grid_weights_and_non_finite_inputs_are_honored() -> None:
    grid = Grid.from_bounds((-3.0, 3.0), (-3.0, 3.0), resolution=0.5)
    lons = np.array([-1.0, 1.0, np.nan, 2.0])
    lats = np.array([0.0, 0.0, 0.0, 2.0])
    weights = np.array([1.0, 1.0, 1.0, np.nan])

    weighted = kde_to_grid(lons, lats, grid, weights=weights, bandwidth=0.5)
    unweighted = kde_to_grid([-1.0, 1.0], [0.0, 0.0], grid, bandwidth=0.5)
    xr.testing.assert_allclose(weighted, unweighted)

    concentrated = kde_to_grid(
        [-1.0, 1.0],
        [0.0, 0.0],
        grid,
        weights=[9.0, 1.0],
        bandwidth=0.5,
    )
    assert concentrated.sel(lon=-1.0, lat=0.0) > concentrated.sel(lon=1.0, lat=0.0)


def test_kde_to_grid_requires_at_least_two_finite_points() -> None:
    grid = Grid.from_bounds((-1.0, 1.0), (-1.0, 1.0), resolution=1.0)

    with pytest.raises(ValueError, match="at least 2 finite points"):
        kde_to_grid([0.0, np.nan], [0.0, 1.0], grid, bandwidth=0.5)


def test_kde_to_grid_operator_matches_function_and_config_round_trips() -> None:
    grid = Grid.from_bounds((-2.0, 2.0), (-2.0, 2.0), resolution=0.5)
    lons = np.array([-1.0, 1.0])
    lats = np.array([0.0, 0.0])
    op = KDEToGrid(grid, bandwidth=0.5, output="counts")

    expected = kde_to_grid(lons, lats, grid, bandwidth=0.5, output="counts")
    xr.testing.assert_allclose(op((lons, lats)), expected)

    cfg = op.get_config()
    assert cfg["grid"] == "<Grid>"
    assert json.loads(json.dumps(cfg)) == cfg
