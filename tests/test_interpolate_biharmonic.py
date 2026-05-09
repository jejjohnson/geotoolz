"""Biharmonic gap-fill tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("skimage")

from skimage.restoration import inpaint_biharmonic

from xr_toolz.interpolate import FillNaNBiharmonic, fillnan_biharmonic


def _analytic_field(n: int = 64) -> tuple[xr.DataArray, np.ndarray]:
    lat = np.linspace(-1.0, 1.0, n)
    lon = np.linspace(-1.0, 1.0, n)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(lon_grid) * np.cos(lat_grid)
    da = xr.DataArray(
        values.copy(),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="signal",
    )
    return da, values


def test_biharmonic_identity_on_finite_input() -> None:
    da, values = _analytic_field(16)

    out = fillnan_biharmonic(da)

    np.testing.assert_array_equal(out.values, values)


def test_biharmonic_smooth_square_gap() -> None:
    da, truth = _analytic_field()
    da.values[27:37, 27:37] = np.nan

    out = fillnan_biharmonic(da)

    gap = np.s_[27:37, 27:37]
    assert not np.isnan(out.values[gap]).any()
    assert np.max(np.abs(out.values[gap] - truth[gap])) < 1e-2


def test_biharmonic_explicit_mask_preserves_unmasked_nans() -> None:
    da, _ = _analytic_field(32)
    mask = xr.zeros_like(da, dtype=bool)
    mask.values[14:18, 14:18] = True
    da = da.copy()
    da.values[14:18, 14:18] = np.nan
    da.values[0, 0] = np.nan

    out = fillnan_biharmonic(da, mask=mask)

    assert not np.isnan(out.values[14:18, 14:18]).any()
    assert np.isnan(out.values[0, 0])


def test_biharmonic_cloud_like_mask_stays_bounded() -> None:
    da, _ = _analytic_field()
    lon_grid, lat_grid = np.meshgrid(da.lon.values, da.lat.values, indexing="xy")
    mask = ((lon_grid + 0.15) ** 2 / 0.28**2 + (lat_grid - 0.05) ** 2 / 0.18**2) < 1
    mask |= ((lon_grid - 0.18) ** 2 / 0.16**2 + (lat_grid + 0.08) ** 2 / 0.12**2) < 1
    da = da.copy()
    da.values[mask] = np.nan

    out = fillnan_biharmonic(da)

    assert not np.isnan(out.values[mask]).any()
    finite_values = da.values[np.isfinite(da.values)]
    assert out.values[mask].min() >= finite_values.min() - 1e-6
    assert out.values[mask].max() <= finite_values.max() + 1e-6


def test_biharmonic_disjoint_regions_with_and_without_split() -> None:
    da, _ = _analytic_field(40)
    da = da.copy()
    da.values[8:13, 8:13] = np.nan
    da.values[25:31, 26:32] = np.nan

    split = fillnan_biharmonic(da, split_into_regions=True)
    coupled = fillnan_biharmonic(da, split_into_regions=False)

    assert not np.isnan(split.values).any()
    assert not np.isnan(coupled.values).any()


def test_biharmonic_fully_nan_slice_passes_through() -> None:
    da, _ = _analytic_field(12)
    da = da.copy()
    da.values[:] = np.nan

    out = fillnan_biharmonic(da)

    assert np.isnan(out.values).all()


def test_biharmonic_edge_gap_matches_skimage_reference() -> None:
    da, _ = _analytic_field(24)
    da = da.copy()
    mask = np.zeros(da.shape, dtype=bool)
    mask[:5, :4] = True
    da.values[mask] = np.nan

    out = fillnan_biharmonic(da, split_into_regions=False)
    reference = inpaint_biharmonic(
        np.where(np.isfinite(da.values), da.values, 0.0),
        mask,
        split_into_regions=False,
        channel_axis=None,
    )

    np.testing.assert_allclose(out.values, reference)


def test_biharmonic_leading_dim_matches_manual_loop() -> None:
    da, _ = _analytic_field(32)
    cube = xr.concat([da, da + 0.1], dim=xr.IndexVariable("time", [0, 1]))
    cube = cube.copy()
    cube.values[0, 10:15, 10:15] = np.nan
    cube.values[1, 18:23, 17:22] = np.nan

    out = fillnan_biharmonic(cube)
    manual = xr.concat(
        [fillnan_biharmonic(cube.isel(time=i)) for i in range(cube.sizes["time"])],
        dim=cube.time,
    )

    xr.testing.assert_allclose(out, manual)


def test_biharmonic_dask_time_chunks_match_numpy() -> None:
    pytest.importorskip("dask.array")
    da, _ = _analytic_field(24)
    cube = xr.concat([da, da + 0.2], dim=xr.IndexVariable("time", [0, 1]))
    cube = cube.copy()
    cube.values[:, 9:14, 9:14] = np.nan

    expected = fillnan_biharmonic(cube)
    out = fillnan_biharmonic(cube.chunk({"time": 1}))

    assert hasattr(out.data, "chunks")
    xr.testing.assert_allclose(out.compute(), expected)
    with pytest.raises(ValueError, match="Core dimension"):
        fillnan_biharmonic(cube.chunk({"lat": 12}))


def test_biharmonic_operator_config_round_trip() -> None:
    op = FillNaNBiharmonic(lon="x", lat="y", split_into_regions=False)

    assert FillNaNBiharmonic(**op.get_config()).get_config() == {
        "lon": "x",
        "lat": "y",
        "split_into_regions": False,
    }
