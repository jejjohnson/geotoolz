"""Tests for the post-Phase-2 Layer-0 primitives in :mod:`xr_toolz.geo`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.geo import (
    Grid,
    Period,
    SpaceTimeGrid,
    add_country_mask,
    add_land_mask,
    add_ocean_mask,
    apply_mask,
    assign_crs,
    block_maxima,
    block_minima,
    calc_latlon,
    coarsen,
    cross_spectrum,
    cyclical_encode,
    encode_time_cyclical,
    encode_time_ordinal,
    fillnan_spatial,
    fillnan_temporal,
    find_intercept_1D,
    fourier_features,
    get_crs,
    histogram_2d,
    lonlat_to_xy,
    points_to_grid,
    positional_encoding,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
    psd_isotropic,
    psd_score,
    psd_spacetime,
    random_fourier_features,
    refine,
    reproject,
    resample_time,
    resolved_scale,
    time_rescale,
    time_unrescale,
    xy_to_lonlat,
)


# ---------- fixtures -------------------------------------------------------


@pytest.fixture
def ds_grid_daily() -> xr.Dataset:
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    lat = np.linspace(-40.0, 40.0, 9)
    lon = np.linspace(-60.0, 60.0, 13)
    rng = np.random.default_rng(0)
    # smooth-ish signal so PSDs and interpolation make sense
    tt = np.arange(len(time))
    base = np.sin(2 * np.pi * tt / 365.25)[:, None, None]
    data = base + 0.1 * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.Dataset(
        {"ssh": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ============== spectral ===================================================


def test_psd_spacetime_returns_freq_axes(ds_grid_daily):
    ds = ds_grid_daily.isel(time=slice(0, 64), lat=4)  # 1-D time/lon slice
    out = psd_spacetime(ds, "ssh", dims=["time", "lon"])
    assert "ssh" in out.data_vars
    assert any(d.startswith("freq_") for d in out.dims)


def test_psd_isotropic_returns_freq_r(ds_grid_daily):
    ds = ds_grid_daily.isel(time=4)
    out = psd_isotropic(ds, "ssh", dims=["lat", "lon"])
    assert "freq_r" in out.dims


def test_cross_spectrum_names_output_pair(ds_grid_daily):
    ds = ds_grid_daily.assign(ssh2=lambda d: d["ssh"] * 2.0).isel(
        time=slice(0, 64), lat=4
    )
    out = cross_spectrum(ds, "ssh", "ssh2", dims=["time", "lon"])
    assert "ssh_ssh2_cross" in out.data_vars


# ============== metrics with PSD ==========================================


def test_psd_score_perfect_prediction_is_one(ds_grid_daily):
    ds_small = ds_grid_daily.isel(time=slice(0, 64), lat=4)
    score = psd_score(ds_small, ds_small, "ssh", psd_dims=["time", "lon"])
    # error is zero everywhere -> score is 1 everywhere
    np.testing.assert_allclose(
        score["score"].values, np.ones_like(score["score"].values), atol=1e-10
    )


def test_find_intercept_1d_linear():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert find_intercept_1D(x=x, y=y, level=0.5) == pytest.approx(2.0)


def test_find_intercept_1d_handles_duplicate_y_values():
    """Regression: plateaued PSD scores (repeated y values) must not
    crash interp1d, which otherwise rejects non-monotone x."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Two pairs of duplicates at y=0.2 and y=0.8
    y = np.array([0.0, 0.2, 0.2, 0.8, 0.8, 1.0])
    crossover = find_intercept_1D(x=x, y=y, level=0.5)
    # Somewhere between the first x corresponding to y=0.2 (idx 1)
    # and the first x corresponding to y=0.8 (idx 3)
    assert 1.0 <= crossover <= 3.0


def test_resolved_scale_returns_wavelength():
    freq = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    score_vals = np.array([0.9, 0.8, 0.5, 0.2, 0.1])
    score = xr.DataArray(score_vals, coords={"freq_r": freq}, dims=("freq_r",))
    wavelength = resolved_scale(score, frequency="freq_r", level=0.5)
    assert wavelength == pytest.approx(1.0 / 0.3, rel=1e-3)


# ============== interpolate ================================================


def test_fillnan_spatial_fills_interior_nans():
    lat = np.linspace(0.0, 1.0, 11)
    lon = np.linspace(0.0, 1.0, 11)
    vals = np.add.outer(lat, lon).astype(float)  # simple linear surface
    vals[5, 5] = np.nan
    da = xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    filled = fillnan_spatial(da, method="linear")
    # Interior NaN should be interpolated to ~1.0 (0.5 + 0.5).
    assert float(filled.isel(lat=5, lon=5).values) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(filled.values).all()


def test_fillnan_rbf_preserves_finite_values():
    """Regression: the RBF filler must only patch NaNs, not overwrite
    valid observations."""
    from xr_toolz.geo import fillnan_rbf

    lat = np.linspace(0.0, 1.0, 6)
    lon = np.linspace(0.0, 1.0, 6)
    vals = np.add.outer(lat, lon).astype(float)
    original = vals.copy()
    # Drop a single interior cell; the rest stays finite.
    vals[2, 3] = np.nan
    da = xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    filled = fillnan_rbf(da)
    # Every originally-finite cell must be bit-for-bit unchanged.
    preserved = ~np.isnan(vals)
    np.testing.assert_array_equal(filled.values[preserved], original[preserved])
    # And the masked-out cell is now finite.
    assert np.isfinite(filled.isel(lat=2, lon=3).values)


def test_fillnan_temporal_uses_xarray_native():
    time = pd.date_range("2020-01-01", periods=10, freq="1D")
    data = np.arange(10.0)
    data[4] = np.nan
    ds = xr.Dataset({"x": ("time", data)}, coords={"time": time})
    out = fillnan_temporal(ds, method="linear")
    assert float(out["x"].isel(time=4).values) == pytest.approx(4.0)


def test_resample_time_daily_to_monthly(ds_grid_daily):
    monthly = resample_time(ds_grid_daily, freq="1ME", method="mean")
    # 2 years -> 24 months
    assert monthly.sizes["time"] == 24


def test_resample_time_rejects_unknown_method(ds_grid_daily):
    with pytest.raises(ValueError, match="Unknown resample method"):
        resample_time(ds_grid_daily, freq="1D", method="unicorn")


def test_coarsen_halves_resolution(ds_grid_daily):
    out = coarsen(ds_grid_daily.isel(time=slice(0, 10)), {"lon": 2})
    assert out.sizes["lon"] == ds_grid_daily.sizes["lon"] // 2


def test_refine_interpolates(ds_grid_daily):
    out = refine(ds_grid_daily.isel(time=0, lat=4), {"lon": 2})
    original_lon = ds_grid_daily.lon.size
    assert out.sizes["lon"] == (original_lon - 1) * 2 + 1


# ============== extremes ===================================================


def test_block_maxima_shrinks_time(ds_grid_daily):
    out = block_maxima(ds_grid_daily["ssh"], block_size=30)
    assert out.sizes["time"] == ds_grid_daily.sizes["time"] // 30


def test_block_minima_less_than_maxima(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    max_ = block_maxima(da, block_size=30)
    min_ = block_minima(da, block_size=30)
    assert bool((min_ <= max_).all())


def test_pot_threshold_monotone_in_quantile(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    lo = float(pot_threshold(da, quantile=0.5).values)
    hi = float(pot_threshold(da, quantile=0.95).values)
    assert lo < hi


def test_pot_exceedances_masks_below_threshold(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    thr = float(pot_threshold(da, quantile=0.9).values)
    out = pot_exceedances(da, quantile=0.9)
    # remaining non-NaN values are all >= threshold
    finite = out.values[np.isfinite(out.values)]
    assert (finite >= thr).all()


def test_pp_counts_shape(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    counts = pp_counts(da, quantile=0.9, block_size=30)
    assert counts.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    assert (counts.values >= 0).all()


def test_pp_stats_mean_finite(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    stats = pp_stats(da, quantile=0.9, block_size=30, statistic=np.mean)
    # Some blocks will have no exceedance -> NaN. But at least one should be finite.
    assert np.any(np.isfinite(stats.values))


def test_pp_counts_supports_gridded_input(ds_grid_daily):
    """Regression: the threshold must be broadcast over non-time dims,
    not scalarized with ``.item()`` (which fails on gridded data)."""
    counts = pp_counts(ds_grid_daily["ssh"], quantile=0.9, block_size=30)
    # Output retains lat/lon as outer dims and coarsens time.
    assert set(counts.dims) >= {"lat", "lon"}
    assert counts.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    assert (counts.values >= 0).all()


def test_pp_stats_supports_gridded_input(ds_grid_daily):
    stats = pp_stats(
        ds_grid_daily["ssh"], quantile=0.9, block_size=30, statistic=np.mean
    )
    assert set(stats.dims) >= {"lat", "lon"}
    assert stats.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    # At least one grid cell has exceedances across all blocks.
    assert np.any(np.isfinite(stats.values))


# ============== discretize =================================================


def test_grid_from_bounds_resolution():
    grid = Grid.from_bounds(
        lon_bnds=(-10.0, 10.0), lat_bnds=(-5.0, 5.0), resolution=1.0
    )
    assert grid.lon.size == 21
    assert grid.lat.size == 11


def test_grid_bin_edges_monotone_increasing():
    grid = Grid.from_bounds((-10.0, 10.0), (-5.0, 5.0), resolution=2.0)
    lon_edges, lat_edges = grid.bin_edges()
    assert (np.diff(lon_edges) > 0).all()
    assert (np.diff(lat_edges) > 0).all()


def test_bin_2d_counts_match_histogram():
    rng = np.random.default_rng(1)
    n = 500
    lons = rng.uniform(-10.0, 10.0, n)
    lats = rng.uniform(-5.0, 5.0, n)
    vals = rng.standard_normal(n)
    grid = Grid.from_bounds((-10.0, 10.0), (-5.0, 5.0), resolution=2.0)
    da = xr.DataArray(
        vals, dims=("obs",), coords={"lon": ("obs", lons), "lat": ("obs", lats)}
    )
    counts = histogram_2d(da, grid=grid)
    assert int(counts.sum()) == n


def test_points_to_grid_mean_matches_manual():
    # Grid cell centers at 0.0 and 1.0 -> bin edges at -0.5, 0.5, 1.5.
    # Place samples firmly inside (not on edges) so assignment is unambiguous.
    lons = np.array([0.2, 0.3, 1.2])
    lats = np.array([0.2, 0.3, 1.2])
    vals = np.array([1.0, 3.0, 7.0])
    grid = Grid(lon=np.array([0.0, 1.0]), lat=np.array([0.0, 1.0]))
    out = points_to_grid(lons, lats, vals, grid=grid, statistic="mean")
    assert float(out.sel(lon=0.0, lat=0.0).values) == pytest.approx(2.0)
    assert float(out.sel(lon=1.0, lat=1.0).values) == pytest.approx(7.0)


def test_period_date_range():
    period = Period(time_min="2020-01-01", time_max="2020-01-05", freq="1D")
    assert len(period.date_range) == 5


def test_space_time_grid_from_bounds():
    stg = SpaceTimeGrid.from_bounds(
        lon_bnds=(0.0, 5.0),
        lat_bnds=(0.0, 5.0),
        resolution=1.0,
        time_min="2020-01-01",
        time_max="2020-01-03",
    )
    assert stg.lon.size == 6
    assert stg.lat.size == 6
    assert len(stg.time) == 3


# ============== crs ========================================================


def test_lonlat_xy_round_trip_webmercator():
    lons = np.array([-10.0, 0.0, 20.0])
    lats = np.array([-30.0, 0.0, 45.0])
    x, y = lonlat_to_xy("EPSG:3857", lons, lats)
    lons_back, lats_back = xy_to_lonlat("EPSG:3857", x, y)
    np.testing.assert_allclose(lons_back, lons, atol=1e-6)
    np.testing.assert_allclose(lats_back, lats, atol=1e-6)


def test_assign_crs_and_get_crs(ds_grid_daily):
    ds = assign_crs(ds_grid_daily, "EPSG:4326")
    crs = get_crs(ds)
    assert crs is not None
    assert crs.to_epsg() == 4326


def test_reproject_changes_crs(ds_grid_daily):
    ds = (
        ds_grid_daily.isel(time=0)
        .rename({"lon": "x", "lat": "y"})
        .pipe(assign_crs, "EPSG:4326")
    )
    out = reproject(ds, target_crs="EPSG:3857")
    assert out.rio.crs.to_epsg() == 3857


def test_calc_latlon_adds_2d_coords():
    x = np.array([0.0, 10_000.0, 20_000.0])
    y = np.array([0.0, 10_000.0])
    ds = xr.Dataset({"z": (("y", "x"), np.zeros((2, 3)))}, coords={"x": x, "y": y})
    ds = assign_crs(ds, "EPSG:3857")
    out = calc_latlon(ds)
    assert "longitude" in out.coords
    assert "latitude" in out.coords
    assert out["longitude"].dims == ("y", "x")


# ============== encoders ===================================================


def test_cyclical_encode_unit_circle():
    sin, cos = cyclical_encode(np.array([0.0, 0.25, 0.5, 0.75]), period=1.0)
    np.testing.assert_allclose(sin**2 + cos**2, np.ones(4), atol=1e-10)


def test_fourier_features_shape():
    vals = np.linspace(0.0, 1.0, 10)
    out = fourier_features(vals, num_freqs=4)
    assert out.shape == (10, 8)


def test_random_fourier_features_reproducible_with_seed():
    vals = np.linspace(0.0, 1.0, 5)
    a = random_fourier_features(vals, num_features=16, seed=123)
    b = random_fourier_features(vals, num_features=16, seed=123)
    np.testing.assert_allclose(a, b)


def test_random_fourier_features_rejects_odd():
    with pytest.raises(ValueError, match="even"):
        random_fourier_features(np.array([0.0]), num_features=5)


def test_positional_encoding_includes_input_by_default():
    vals = np.array([0.1, 0.2, 0.3])
    out = positional_encoding(vals, num_freqs=2)
    assert out.shape == (3, 5)  # 2 * 2 + 1
    np.testing.assert_allclose(out[:, 0], vals)


def test_time_rescale_round_trip(ds_grid_daily):
    rescaled = time_rescale(ds_grid_daily, freq_dt=1, freq_unit="D")
    assert np.issubdtype(rescaled.time.dtype, np.floating)
    restored = time_unrescale(rescaled)
    # Daily cadence with the min as t0 -> restored times equal the originals.
    np.testing.assert_array_equal(
        restored.time.values.astype("datetime64[ns]"),
        ds_grid_daily.time.values.astype("datetime64[ns]"),
    )


def test_encode_time_cyclical_adds_paired_coords(ds_grid_daily):
    out = encode_time_cyclical(ds_grid_daily, components=["dayofyear"])
    assert "dayofyear_sin" in out.coords
    assert "dayofyear_cos" in out.coords
    np.testing.assert_allclose(
        out["dayofyear_sin"].values ** 2 + out["dayofyear_cos"].values ** 2,
        np.ones(ds_grid_daily.sizes["time"]),
        atol=1e-10,
    )


def test_encode_time_ordinal_is_monotone(ds_grid_daily):
    out = encode_time_ordinal(ds_grid_daily)
    values = out["time_ordinal"].values
    assert (np.diff(values) > 0).all()


# ============== masks (hit the API but keep it tolerant of offline CI) ====


def test_add_land_mask_adds_coord(ds_grid_daily):
    """Smoke-test; requires regionmask data (cached after first use)."""
    try:
        out = add_land_mask(ds_grid_daily.isel(time=0))
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
    assert "land_mask" in out.coords


def test_add_ocean_mask_global(ds_grid_daily):
    try:
        out = add_ocean_mask(ds_grid_daily.isel(time=0), ocean="global")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
    assert "ocean_mask" in out.coords


def test_add_ocean_mask_rejects_unknown_basin(ds_grid_daily):
    try:
        with pytest.raises(ValueError, match="not found"):
            add_ocean_mask(ds_grid_daily.isel(time=0), ocean="does-not-exist")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")


def test_apply_mask_by_name_drops_nothing_when_all_true():
    ds = xr.Dataset(
        {"x": ("i", np.arange(4.0))},
        coords={"i": [0, 1, 2, 3], "mask": ("i", np.ones(4, dtype=np.int16))},
    )
    out = apply_mask(ds, "mask")
    np.testing.assert_array_equal(out["x"].values, np.arange(4.0))


def test_add_country_mask_rejects_unknown(ds_grid_daily):
    try:
        with pytest.raises(ValueError, match="not found"):
            add_country_mask(ds_grid_daily.isel(time=0), country="Atlantis")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
