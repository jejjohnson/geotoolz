from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.interpolate import Grid, GriddataToGrid, griddata_to_grid


def _analytic(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Smooth field with curvature in both axes — separates cubic from linear."""
    return np.sin(lon) * np.cos(lat)


def _swath(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    lons = rng.uniform(-1.0, 1.0, n)
    lats = rng.uniform(-1.0, 1.0, n)
    return lons, lats, _analytic(lons, lats)


def _interior_grid() -> Grid:
    # Well inside the swath's convex hull, so no cell is extrapolated.
    return Grid(lon=np.linspace(-0.6, 0.6, 21), lat=np.linspace(-0.6, 0.6, 21))


def _truth_on(grid: Grid) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(grid.lon, grid.lat, indexing="xy")
    return _analytic(lon_grid, lat_grid)


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [("nearest", 5e-2), ("linear", 1e-3), ("cubic", 1e-4)],
)
def test_reconstructs_analytic_field_on_the_interior(
    method: str, tolerance: float
) -> None:
    grid = _interior_grid()

    out = griddata_to_grid(*_swath(), grid, method=method)

    assert out.dims == ("lat", "lon")
    assert out.shape == (len(grid.lat), len(grid.lon))
    rmse = float(np.sqrt(np.nanmean((out.values - _truth_on(grid)) ** 2)))
    assert rmse < tolerance


def test_cubic_beats_linear_beats_nearest() -> None:
    grid = _interior_grid()
    swath = _swath()
    truth = _truth_on(grid)

    def rmse(method: str) -> float:
        out = griddata_to_grid(*swath, grid, method=method)
        return float(np.sqrt(np.nanmean((out.values - truth) ** 2)))

    assert rmse("cubic") < rmse("linear") < rmse("nearest")


def test_out_of_hull_cells_get_fill_value() -> None:
    # Grid extends well beyond the swath's [-1, 1] box.
    grid = Grid(lon=np.linspace(-3.0, 3.0, 7), lat=np.linspace(-3.0, 3.0, 7))

    default = griddata_to_grid(*_swath(), grid, method="cubic")
    explicit = griddata_to_grid(*_swath(), grid, method="linear", fill_value=-999.0)

    assert np.isnan(default.values).any()
    assert np.isnan(default.sel(lon=-3.0, lat=-3.0, method="nearest"))
    assert float(explicit.sel(lon=-3.0, lat=-3.0, method="nearest")) == -999.0


def test_nearest_extrapolates_and_ignores_fill_value() -> None:
    # "nearest" is a Voronoi assignment — every cell has a nearest point.
    grid = Grid(lon=np.linspace(-3.0, 3.0, 7), lat=np.linspace(-3.0, 3.0, 7))

    out = griddata_to_grid(*_swath(), grid, method="nearest", fill_value=-999.0)

    assert np.all(np.isfinite(out.values))
    assert not np.any(out.values == -999.0)


def test_non_finite_points_are_dropped() -> None:
    grid = _interior_grid()
    lons, lats, values = _swath()
    clean = griddata_to_grid(lons, lats, values, grid, method="linear")

    dirty_values = values.copy()
    dirty_values[:10] = np.nan
    dirty_lons = lons.copy()
    dirty_lons[10:20] = np.nan

    out = griddata_to_grid(dirty_lons, lats, dirty_values, grid, method="linear")

    # Dropping 20 of 3000 points perturbs the reconstruction only slightly,
    # and crucially does not poison the whole grid with NaN.
    assert np.all(np.isfinite(out.values))
    assert float(np.abs(out.values - clean.values).max()) < 1e-2


def test_two_dimensional_inputs_are_flattened() -> None:
    grid = _interior_grid()
    lons, lats, values = _swath(n=2500)
    shape = (50, 50)

    flat = griddata_to_grid(lons, lats, values, grid, method="linear")
    swathed = griddata_to_grid(
        lons.reshape(shape),
        lats.reshape(shape),
        values.reshape(shape),
        grid,
        method="linear",
    )

    xr.testing.assert_allclose(flat, swathed)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        griddata_to_grid(*_swath(), _interior_grid(), method="bogus")


def test_mismatched_input_shapes_raise() -> None:
    lons, lats, values = _swath()
    with pytest.raises(ValueError, match="same shape"):
        griddata_to_grid(lons, lats[:-1], values, _interior_grid())


@pytest.mark.parametrize(("method", "n_points"), [("cubic", 2), ("nearest", 0)])
def test_too_few_finite_points_raise(method: str, n_points: int) -> None:
    lons, lats, values = _swath()
    with pytest.raises(ValueError, match="at least"):
        griddata_to_grid(
            lons[:n_points],
            lats[:n_points],
            values[:n_points],
            _interior_grid(),
            method=method,
        )


def test_equal_size_transposed_shapes_are_rejected() -> None:
    # (2, 3) and (3, 2) both ravel to (6,), so a post-ravel check would
    # accept them and silently pair unrelated coordinates with values.
    with pytest.raises(ValueError, match=r"same shape"):
        griddata_to_grid(
            np.ones((2, 3)), np.ones((3, 2)), np.ones((2, 3)), _interior_grid()
        )


def test_float32_input_yields_float64_output() -> None:
    lons, lats, values = (a.astype("float32") for a in _swath(n=500))

    out = griddata_to_grid(lons, lats, values, _interior_grid(), method="linear")

    assert out.dtype == np.dtype("float64")


def test_operator_matches_function_and_config_round_trips() -> None:
    grid = _interior_grid()
    payload = _swath()
    op = GriddataToGrid(grid, method="linear", fill_value=0.0)

    xr.testing.assert_allclose(
        op(payload), griddata_to_grid(*payload, grid, method="linear", fill_value=0.0)
    )
    config = op.get_config()
    assert config == {"grid": "<Grid>", "method": "linear", "fill_value": 0.0}
    assert json.loads(json.dumps(config)) == config


def test_operator_default_config_encodes_nan_fill_json_safely() -> None:
    config = GriddataToGrid(_interior_grid()).get_config()

    assert config == {"grid": "<Grid>", "method": "cubic", "fill_value": "nan"}
    assert json.loads(json.dumps(config)) == config


def test_operator_output_signature_matches_grid() -> None:
    from xrcore import Signature

    grid = _interior_grid()

    signature = GriddataToGrid(grid).compute_output_signature(
        Signature({"points": 3000}, dtype=np.dtype("float32"))
    )

    assert signature.dims == {"lat": len(grid.lat), "lon": len(grid.lon)}
    # The kernel casts to float64, so a float32 payload must not be
    # reported as float32 downstream.
    assert signature.dtype == np.dtype("float64")
