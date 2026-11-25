"""Pixel and ensemble scores on diagnostix's kernels (#800)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrops import metrics
from xrops.metrics import (
    CRPS,
    MSE,
    RMSE,
    EnsembleCoverage,
    SpreadSkillRatio,
    correlation,
    crps_ensemble,
    ensemble_coverage,
    r2_score,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)


PIXEL = [
    "mse",
    "rmse",
    "mae",
    "bias",
    "nrmse",
    "nrmse_score",
    "correlation",
    "r2_score",
]


def _pair() -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(0)
    lat = np.linspace(-60.0, 60.0, 12)
    ref = 3.0 + rng.standard_normal((5, 12, 8))
    pred = ref + 0.4 * rng.standard_normal((5, 12, 8))
    return tuple(
        xr.DataArray(a, dims=("time", "lat", "lon"), coords={"lat": lat})
        for a in (pred, ref)
    )


# ---------- pixel --------------------------------------------------------------


@pytest.mark.parametrize("name", PIXEL)
def test_a_gap_is_left_out_of_every_moment(name: str) -> None:
    """Correlation and R² took the cells of each moment as they came (#777)."""
    score = getattr(metrics, name)
    pred, ref = _pair()
    gone = np.zeros(8, dtype=bool)
    gone[[1, 4, 5]] = True
    gappy = ref.where(~xr.DataArray(gone, dims="lon"))
    expected = score(pred.isel(lon=~gone), ref.isel(lon=~gone), dim=("lat", "lon"))
    xr.testing.assert_allclose(score(pred, gappy, dim=("lat", "lon")), expected)
    xr.testing.assert_allclose(
        score(pred.where(gappy.notnull()), ref, dim=("lat", "lon")), expected
    )


def test_correlation_and_r2_of_a_pair_with_gaps() -> None:
    rng = np.random.default_rng(1)
    x = xr.DataArray(rng.standard_normal(4000) + np.linspace(0, 40, 4000), dims="x")
    y = (x + 0.3 * rng.standard_normal(4000)).where(np.arange(4000) < 500)
    expected = np.corrcoef(x[:500], y[:500])[0, 1]
    assert float(correlation(x, y, dim="x")) == pytest.approx(expected)
    residual = float(((x - y) ** 2)[:500].sum())
    total = float(((y - y.mean()) ** 2)[:500].sum())
    assert float(r2_score(x, y, dim="x")) == pytest.approx(1.0 - residual / total)


@pytest.mark.parametrize("name", PIXEL)
def test_weights_on_some_of_the_pooled_dims(name: str) -> None:
    score = getattr(metrics, name)
    pred, ref = _pair()
    area = np.cos(np.deg2rad(pred["lat"]))
    got = score(pred, ref, dim=("lat", "lon"), weights=area)
    assert got.dims == ("time",)
    everywhere = area * xr.ones_like(pred.isel(time=0, drop=True))
    xr.testing.assert_allclose(
        got, score(pred, ref, dim=("lat", "lon"), weights=everywhere)
    )
    whole = score(pred, ref, dim=("lat", "lon"), weights=xr.ones_like(area))
    xr.testing.assert_allclose(whole, score(pred, ref, dim=("lat", "lon")))


def test_a_weighted_mean_error() -> None:
    pred, ref = _pair()
    area = np.cos(np.deg2rad(pred["lat"]))
    expected = (pred - ref).weighted(area).mean(("lat", "lon"))
    xr.testing.assert_allclose(
        metrics.bias(pred, ref, dim=("lat", "lon"), weights=area), expected
    )


@pytest.mark.parametrize("name", PIXEL)
def test_dask_with_weights_is_lazy_and_equal(name: str) -> None:
    pytest.importorskip("dask.array")
    score = getattr(metrics, name)
    pred, ref = _pair()
    area = np.cos(np.deg2rad(pred["lat"]))
    lazy = score(
        pred.chunk({"time": 1, "lat": 5}),
        ref.chunk({"lon": 3}),
        dim=("lat", "lon"),
        weights=area,
    )
    assert lazy.chunks is not None
    expected = score(pred, ref, dim=("lat", "lon"), weights=area)
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_float32_stays_float32() -> None:
    pred, ref = (da.astype("float32") for da in _pair())
    assert metrics.rmse(pred, ref, dim="lat").dtype == np.float32


def test_a_constant_field_has_no_correlation() -> None:
    flat = xr.DataArray(np.full(7, 0.1), dims="x")
    other = xr.DataArray(np.arange(7.0), dims="x")
    assert np.isnan(correlation(flat, other, dim="x"))
    assert float(r2_score(flat, flat, dim="x")) == 1.0
    assert float(r2_score(other, flat, dim="x")) == -np.inf


def test_the_operator_takes_the_weights_from_the_reference() -> None:
    pred, ref = _pair()
    area = np.cos(np.deg2rad(pred["lat"]))
    op = RMSE("x", ("lat", "lon"), weights="area")
    out = op(pred.to_dataset(name="x"), xr.Dataset({"x": ref, "area": area}))
    xr.testing.assert_allclose(
        out, metrics.rmse(pred, ref, dim=("lat", "lon"), weights=area)
    )
    config = op.get_config()
    assert json.loads(json.dumps(config)) == config
    assert RMSE(**config).get_config() == config
    assert MSE("x", "lat").get_config()["weights"] is None


# ---------- ensembles ----------------------------------------------------------


def _reliable(members: int, n: int = 40_000) -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(members)
    ens = xr.DataArray(rng.standard_normal((members, n)), dims=("member", "x"))
    return ens, xr.DataArray(rng.standard_normal(n), dims="x")


@pytest.mark.parametrize("members", [2, 5, 20])
def test_a_reliable_ensemble_of_any_size(members: int) -> None:
    """It had a ratio of 0.91 (#775) and a coverage of 0.62 (#790) for five."""
    ens, ref = _reliable(members)
    assert float(spread_skill_ratio(ens, ref)) == pytest.approx(1.0, abs=0.015)
    for level in (0.5, 0.9):
        assert float(ensemble_coverage(ens, ref, level=level)) == pytest.approx(
            level, abs=0.015
        )


def test_a_missing_reference_is_in_no_rank() -> None:
    """It was counted as rank 0 (#776)."""
    ens, ref = _reliable(4, n=1000)
    gappy = ref.where(np.arange(1000) >= 300)
    out = rank_histogram(ens, gappy)["rank_count"]
    assert float(out.sum()) == 700.0
    xr.testing.assert_allclose(
        out, rank_histogram(ens[:, 300:], ref[300:])["rank_count"]
    )


def test_an_observation_that_equals_members_is_shared() -> None:
    ens = xr.DataArray([[1.0], [1.0], [2.0]], dims=("member", "x"))
    out = rank_histogram(ens, xr.DataArray([1.0], dims="x"))["rank_count"]
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3, 0.0])


@pytest.mark.parametrize("score", [spread_skill_ratio, ensemble_coverage])
def test_ensemble_scores_over_some_dims_with_gaps_weights_and_dask(score) -> None:
    rng = np.random.default_rng(3)
    ens = xr.DataArray(
        rng.standard_normal((6, 4, 10, 9)), dims=("member", "time", "lat", "lon")
    )
    ref = xr.DataArray(rng.standard_normal((4, 10, 9)), dims=("time", "lat", "lon"))
    kept = xr.DataArray(np.arange(9) > 3, dims="lon")
    area = xr.DataArray(np.linspace(0.2, 1.0, 10), dims="lat")
    expected = score(
        ens.isel(lon=kept.values),
        ref.isel(lon=kept.values),
        dims=("lat", "lon"),
        weights=area,
    )
    assert expected.dims == ("time",)
    got = score(ens.where(kept), ref, dims=("lat", "lon"), weights=area)
    xr.testing.assert_allclose(got, expected)
    turned = score(
        ens.where(kept).transpose("lon", "time", "member", "lat"),
        ref,
        dims=("lat", "lon"),
        weights=area,
    )
    xr.testing.assert_allclose(turned, expected)
    pytest.importorskip("dask.array")
    lazy = score(
        ens.where(kept).chunk({"time": 1, "lat": 4}),
        ref.chunk({"lon": 2}),
        dims=("lat", "lon"),
        weights=area,
    )
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_a_reference_without_a_dim_is_compared_with_each() -> None:
    rng = np.random.default_rng(4)
    ens = xr.DataArray(rng.standard_normal((6, 3, 50)), dims=("member", "lead", "x"))
    ref = xr.DataArray(rng.standard_normal(50), dims="x")
    out = ensemble_coverage(ens, ref, dims=("x",))
    assert out.dims == ("lead",)
    assert float(out[1]) == pytest.approx(
        float(ensemble_coverage(ens.isel(lead=1), ref))
    )


def test_operators_match_the_functions() -> None:
    ens, ref = _reliable(6, n=200)
    ds_e, ds_r = ens.to_dataset(name="x"), ref.to_dataset(name="x")
    xr.testing.assert_equal(
        EnsembleCoverage("x", level=0.8)(ds_e, ds_r),
        ensemble_coverage(ens, ref, level=0.8),
    )
    xr.testing.assert_equal(
        SpreadSkillRatio("x")(ds_e, ds_r), spread_skill_ratio(ens, ref)
    )
    config = EnsembleCoverage("x", level=0.8, dims=("x",)).get_config()
    assert json.loads(json.dumps(config)) == config
    assert EnsembleCoverage(**config).get_config() == config


def test_the_fair_crps_does_not_depend_on_the_number_of_members() -> None:
    small, large = (
        float(crps_ensemble(*_reliable(m), dims=("x",), fair=True)) for m in (2, 20)
    )
    assert small == pytest.approx(large, abs=0.01)
    assert float(crps_ensemble(*_reliable(2), dims=("x",))) > small + 0.1
    with pytest.raises(ValueError, match="at least 2"):
        crps_ensemble(*_reliable(1, n=5), fair=True)


def test_crps_with_weights_and_gaps() -> None:
    ens, ref = _reliable(4, n=100)
    weights = xr.DataArray(np.linspace(0.1, 1.0, 100), dims="x")
    each = crps_ensemble(ens, ref.where(np.arange(100) > 20))
    assert int(each.isnull().sum()) == 21
    out = crps_ensemble(
        ens, ref.where(np.arange(100) > 20), dims=("x",), weights=weights
    )
    expected = float((each * weights)[21:].sum() / weights[21:].sum())
    assert float(out) == pytest.approx(expected)
    assert CRPS("x").get_config()["variable"] == "x"


def test_a_probability_outside_the_bins_is_in_none() -> None:
    p = xr.DataArray([0.1, 0.3, 0.9, 1.0, np.nan], dims="x")
    o = xr.DataArray([0.0, 1.0, 1.0, 0.0, 1.0], dims="x")
    out = reliability_curve(p, o, probability_bins=np.array([0.2, 0.5, 1.0]))
    np.testing.assert_array_equal(out["count"], [1, 2])
    np.testing.assert_allclose(out["observed_frequency"], [1.0, 0.5])
    assert out["count"].dtype == np.int64
