"""Tests for gap-filling primitives in :mod:`xrtoolz.interpolate`."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.interpolate import fillnan_laplacian
from xrtoolz.interpolate._src.gap_fill import _fillnan_laplacian_slice
from xrtoolz.interpolate.operators import FillNaNLaplacian


def _harmonic_da() -> xr.DataArray:
    """Return a linear field satisfying ∇²u = 0 exactly."""
    lat = np.linspace(-1.0, 1.0, 21)
    lon = np.linspace(-2.0, 2.0, 25)
    vals = 1.0 + 2.0 * lat[:, None] - 0.5 * lon[None, :]
    return xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_fillnan_laplacian_matches_harmonic_field() -> None:
    expected = _harmonic_da()
    masked = expected.copy()
    masked.values[8:13, 10:15] = np.nan

    filled = fillnan_laplacian(masked, max_iter=5000, tol=1e-8, relaxation=1.5)

    xr.testing.assert_allclose(filled, expected, atol=2e-4)


def test_fillnan_laplacian_preserves_all_nan_and_all_finite_slices() -> None:
    finite = _harmonic_da()
    xr.testing.assert_identical(fillnan_laplacian(finite), finite)

    all_nan = finite.copy(data=np.full(finite.shape, np.nan))
    out = fillnan_laplacian(all_nan)
    assert np.isnan(out.values).all()


def test_fillnan_laplacian_sor_converges_faster_than_gauss_seidel() -> None:
    masked = _harmonic_da().values
    masked[7:14, 9:16] = np.nan

    _, gs_iters = _fillnan_laplacian_slice(
        masked,
        max_iter=5000,
        tol=1e-6,
        relaxation=1.0,
        boundary="reflect",
    )
    _, sor_iters = _fillnan_laplacian_slice(
        masked,
        max_iter=5000,
        tol=1e-6,
        relaxation=1.5,
        boundary="reflect",
    )

    assert sor_iters < gs_iters


def test_fillnan_laplacian_honours_max_iter_and_tol() -> None:
    masked = _harmonic_da().values
    masked[8:13, 10:15] = np.nan

    _, capped_iters = _fillnan_laplacian_slice(
        masked,
        max_iter=2,
        tol=0.0,
        relaxation=1.0,
        boundary="reflect",
    )
    _, early_iters = _fillnan_laplacian_slice(
        masked,
        max_iter=100,
        tol=10.0,
        relaxation=1.0,
        boundary="reflect",
    )

    assert capped_iters == 2
    assert early_iters == 1


def test_fillnan_laplacian_reflect_boundary_fills_corner_from_two_neighbors() -> None:
    vals = np.array(
        [
            [np.nan, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    da = xr.DataArray(
        vals, dims=("lat", "lon"), coords={"lat": [0, 1, 2], "lon": [0, 1, 2]}
    )

    filled = fillnan_laplacian(da, max_iter=1000, tol=1e-10)

    assert float(filled.isel(lat=0, lon=0)) == pytest.approx(3.0, abs=1e-8)


def test_fillnan_laplacian_wrap_boundary_uses_longitude_seam() -> None:
    vals = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [np.nan, 2.0, 0.0, 10.0],
            [8.0, 0.0, 0.0, 0.0],
        ]
    )
    da = xr.DataArray(
        vals,
        dims=("lat", "lon"),
        coords={"lat": [0, 1, 2], "lon": [0, 90, 180, 270]},
    )

    filled = fillnan_laplacian(da, max_iter=1000, tol=1e-10, boundary="wrap")

    assert float(filled.isel(lat=1, lon=0)) == pytest.approx(6.0, abs=1e-8)


def test_fillnan_laplacian_fills_each_leading_slice_independently() -> None:
    base = _harmonic_da().isel(lat=slice(8, 13), lon=slice(10, 15))
    stacked = xr.concat([base, base + 10.0], dim=xr.IndexVariable("time", [0, 1]))
    masked = stacked.copy()
    masked.values[:, 2, 2] = np.nan

    filled = fillnan_laplacian(masked, max_iter=1000, tol=1e-8)

    xr.testing.assert_allclose(filled, stacked, atol=1e-6)


def test_fillnan_laplacian_operator_round_trips_and_skips_non_spatial_vars() -> None:
    da = _harmonic_da()
    masked = da.copy()
    masked.values[10, 12] = np.nan
    ds = xr.Dataset(
        {"field": masked, "series": ("time", [1.0, np.nan])},
        coords={"time": [0, 1]},
    )
    op = FillNaNLaplacian(max_iter=1000, tol=1e-8, relaxation=1.5)

    cfg = json.loads(json.dumps(op.get_config()))
    out = op(ds)

    assert cfg == {
        "max_iter": 1000,
        "tol": 1e-8,
        "relaxation": 1.5,
        "boundary": "reflect",
        "lon": "lon",
        "lat": "lat",
    }
    assert np.isfinite(out["field"].values).all()
    xr.testing.assert_identical(out["series"], ds["series"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": -1}, "max_iter"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": -1.0}, "tol"),
        ({"relaxation": 0.0}, "relaxation"),
        ({"relaxation": 2.0}, "relaxation"),
        ({"boundary": "banana"}, "boundary"),
    ],
)
def test_fillnan_laplacian_operator_validates_args_eagerly(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        FillNaNLaplacian(**kwargs)


def test_fillnan_laplacian_inf_does_not_propagate_into_fill() -> None:
    da = _harmonic_da().copy()
    # +inf in a finite cell would historically poison np.nanmean and
    # propagate into every NaN; the seed should now ignore it.
    da.values[0, 0] = np.inf
    da.values[8:13, 10:15] = np.nan

    filled = fillnan_laplacian(da, max_iter=200, tol=1e-6)

    filled_region = filled.values[8:13, 10:15]
    assert np.isfinite(filled_region).all()
