"""ODC-1.4 residual binning, regional scoring, regimes, and DM tests."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from xrtoolz.geo.regimes import coastal_regions, eddy_regions, equatorial_regions
from xrtoolz.metrics import (
    BinnedResiduals2D,
    DieboldMariano,
    RegionScores,
    bin_residuals_2d,
    dm_test,
    scores_by_region,
)


def test_bin_residuals_2d_matches_manual_reductions() -> None:
    ds = xr.Dataset(
        {
            "ref": ("point", np.zeros(4)),
            "pred": ("point", np.array([1.0, 3.0, 5.0, 7.0])),
        },
        coords={
            "lon": ("point", np.array([0.25, 0.75, 1.25, 0.25])),
            "lat": ("point", np.array([0.25, 0.25, 0.25, 1.25])),
        },
    )

    out = bin_residuals_2d(
        ds,
        var_ref="ref",
        var_pred="pred",
        lon_bins=[0.0, 1.0, 2.0],
        lat_bins=[0.0, 1.0, 2.0],
    )

    np.testing.assert_allclose(out["mean"].values[0, 0], 2.0)
    np.testing.assert_allclose(out["std"].values[0, 0], 1.0)
    np.testing.assert_allclose(out["count"].values, [[2.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(out["rmse"].values[0, 0], np.sqrt(5.0))
    assert np.isnan(out["mean"].values[1, 1])


def test_scores_by_region_with_categorical_dataarray() -> None:
    ds = _track_dataset()
    regions = xr.DataArray(
        np.array(["south", "south", "north", np.nan], dtype=object),
        dims="point",
        coords={"point": ds["point"]},
    )

    out = scores_by_region(
        ds,
        var_ref="ref",
        var_pred="pred",
        regions=regions,
        metrics=("rmse", "bias"),
    )

    expected_south = np.sqrt(np.mean(np.array([1.0, 2.0]) ** 2))
    np.testing.assert_allclose(out["rmse"].sel(region="south"), expected_south)
    np.testing.assert_allclose(out["bias"].sel(region="north"), 3.0)
    assert "nan" not in {str(v) for v in out["region"].values}


def test_scores_by_region_with_regionmask_regions() -> None:
    regionmask = pytest.importorskip("regionmask")
    from shapely.geometry import box

    ds = _track_dataset()
    regions = regionmask.Regions(
        [box(-2.0, -2.0, 0.0, 2.0), box(0.0, -2.0, 2.0, 2.0)],
        names=["west", "east"],
        abbrevs=["w", "e"],
    )

    out = scores_by_region(
        ds,
        var_ref="ref",
        var_pred="pred",
        regions=regions,
        metrics=("rmse",),
    )

    np.testing.assert_allclose(out["rmse"].sel(region="west"), np.sqrt(2.5))
    np.testing.assert_allclose(out["rmse"].sel(region="east"), np.sqrt(12.5))


def test_regime_constructors_mask_expected_points() -> None:
    # Coastal California, open Pacific, equator, and two extra-tropical points.
    lons = xr.DataArray([-122.5, -150.0, 0.0, 0.0, 0.0], dims="point")
    lats = xr.DataArray([37.0, 0.0, 0.0, 6.0, -6.0], dims="point")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        coastal_mask = coastal_regions().mask(lons[:2], lats[:2], method="shapely")
        equatorial_mask = equatorial_regions(lat_threshold=5.0).mask(
            lons[2:],
            lats[2:],
            method="shapely",
        )
    assert coastal_mask.values.tolist() == [0.0, 1.0]
    assert equatorial_mask.values.tolist() == [0.0, 1.0, 1.0]


def test_eddy_regions_returns_two_class_dataarray() -> None:
    values = np.zeros((7, 7), dtype=float)
    values[3, 3] = 10.0
    ds = xr.Dataset(
        {"ssh": (("lat", "lon"), values)},
        coords={"lat": np.arange(7), "lon": np.arange(7)},
    )

    out = eddy_regions(ds, var="ssh", threshold=1.0, window=(3, 3))

    labels = set(np.unique(out.values[~np.isnan(out.values)]).astype(int).tolist())
    assert labels == {0, 1}


def test_dm_test_identical_and_skewed_losses() -> None:
    losses = xr.DataArray(np.array([1.0, -2.0, 3.0, -4.0]), dims="time")
    out = dm_test(losses, losses)
    np.testing.assert_allclose(out["statistic"].values, 0.0)
    np.testing.assert_allclose(out["p_value"].values, 1.0)

    out = dm_test(
        xr.DataArray(np.arange(1.0, 7.0), dims="time"),
        xr.DataArray(np.ones(6), dims="time"),
        hln_correction=False,
    )
    assert out["statistic"].values > 0.0
    assert out["p_value"].values < 0.05


def test_dm_test_hln_correction_changes_p_value() -> None:
    a = xr.DataArray(np.array([2.0, 1.8, 2.2, 2.1, 1.9, 2.3]), dims="time")
    b = xr.DataArray(np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0]), dims="time")

    p_hln = dm_test(a, b, h=2, hln_correction=True)["p_value"].values
    p_normal = dm_test(a, b, h=2, hln_correction=False)["p_value"].values

    assert p_hln != p_normal


def test_dm_test_matches_statsmodels_hac_reference() -> None:
    sm = pytest.importorskip("statsmodels.api")
    cov_hac = pytest.importorskip("statsmodels.stats.sandwich_covariance").cov_hac
    a = np.array([2.0, 1.0, 3.0, 2.5, 4.0, 3.5, 5.0])
    b = np.array([1.0, 1.2, 1.1, 1.4, 1.3, 1.5, 1.6])
    d = np.abs(a) ** 2 - np.abs(b) ** 2
    fit = sm.OLS(d, np.ones((d.size, 1))).fit()
    cov = cov_hac(fit, nlags=1, use_correction=False)
    expected = float(fit.params[0] / np.sqrt(cov[0, 0]))

    out = dm_test(
        xr.DataArray(a, dims="time"),
        xr.DataArray(b, dims="time"),
        h=2,
        hln_correction=False,
    )

    np.testing.assert_allclose(out["statistic"].values, expected, rtol=1e-6, atol=1e-6)


def test_dm_test_dim_wise_broadcasts_over_remaining_dims() -> None:
    rng = np.random.default_rng(0)
    a = xr.DataArray(rng.standard_normal((3, 20)), dims=("site", "time"))
    b = xr.DataArray(rng.standard_normal((3, 20)), dims=("site", "time"))

    out = dm_test(a, b, dim="time")

    assert out["statistic"].dims == ("site",)
    assert out["p_value"].dims == ("site",)
    # Each site matches the per-slice scalar computation.
    for i in range(3):
        scalar = dm_test(a.isel(site=i), b.isel(site=i))
        np.testing.assert_allclose(
            out["statistic"].isel(site=i).values, scalar["statistic"].values
        )


def test_dieboldmariano_operator_and_config_round_trip() -> None:
    a = xr.DataArray(np.arange(1.0, 7.0), dims="time")
    b = xr.DataArray(np.ones(6), dims="time")
    ds_a = xr.Dataset({"error": a})
    ds_b = xr.Dataset({"error": b})

    op = DieboldMariano("error", hln_correction=False)
    out = op(ds_a, ds_b)
    expected = dm_test(a, b, hln_correction=False)
    xr.testing.assert_allclose(out, expected)

    cfg = op.get_config()
    json.dumps(cfg)  # JSON-roundtrip-safe
    xr.testing.assert_allclose(DieboldMariano(**cfg)(ds_a, ds_b), out)


def test_operator_configs_round_trip() -> None:
    ds = _track_dataset()
    binned = BinnedResiduals2D(
        var_ref="ref",
        var_pred="pred",
        lon_bins=[-2.0, 0.0, 2.0],
        lat_bins=[-2.0, 0.0, 2.0],
    )
    xr.testing.assert_identical(
        BinnedResiduals2D(**binned.get_config())(ds), binned(ds)
    )

    regions = xr.DataArray(
        ["a", "a", "b", "b"], dims="point", coords={"point": ds.point}
    )
    scores = RegionScores(var_ref="ref", var_pred="pred", regions=regions)
    # get_config returns a JSON-safe summary of regions, not the actual
    # object; callers must re-supply the regions when reconstructing.
    cfg = scores.get_config()
    assert cfg["regions"] == {
        "kind": "DataArray",
        "name": None,
        "dims": ["point"],
    }
    json.dumps(cfg)  # JSON-roundtrip-safe
    rebuilt_cfg = {**cfg, "regions": regions}
    xr.testing.assert_identical(RegionScores(**rebuilt_cfg)(ds), scores(ds))


def test_scores_by_region_count_is_zero_for_empty_region() -> None:
    ds = _track_dataset()
    # All four points share the same region, so a fixed second region
    # collects zero points and ``count`` should report 0, not NaN.
    regions = xr.DataArray(
        ["a", "a", "a", "a"], dims="point", coords={"point": ds.point}
    )
    out = scores_by_region(
        ds,
        var_ref="ref",
        var_pred="pred",
        regions=regions,
        metrics=("count", "rmse"),
    )
    np.testing.assert_array_equal(out["count"].values, np.array([4.0]))


def test_dm_test_accepts_errors_keyword_naming() -> None:
    """The renamed parameter names work as keyword args."""
    e = xr.DataArray(np.array([0.5, -0.5, 0.5, -0.5, 0.5]), dims="time")
    out = dm_test(errors_a=e, errors_b=e)
    np.testing.assert_allclose(out["statistic"].values, 0.0)


def _track_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "ref": ("point", np.zeros(4)),
            "pred": ("point", np.array([1.0, 2.0, 3.0, 4.0])),
        },
        coords={
            "point": np.arange(4),
            "lon": ("point", np.array([-1.0, -0.5, 0.5, 1.0])),
            "lat": ("point", np.zeros(4)),
        },
    )
