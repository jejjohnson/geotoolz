"""Tests for V2 (Data Representation Metrics).

Covers:
- V2.1 structural: SSIM, GradientDifference, PhaseShiftError, CentroidDisplacement
- V2.2 probabilistic: SpreadSkillRatio, RankHistogram, EnsembleCoverage,
  ReliabilityCurve
- V2.3 distributional: CRPS, EnergyDistance, Wasserstein1
- V2.4 masked: MaskedMetric
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.metrics import (
    CRPS,
    RMSE,
    SSIM,
    CentroidDisplacement,
    EnergyDistance,
    EnsembleCoverage,
    GradientDifference,
    MaskedMetric,
    PhaseShiftError,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
    Wasserstein1,
    masked_metric,
    phase_shift_error,
)


# =========================================================================
# V2.4 — Masked
# =========================================================================


def test_masked_metric_matches_manual_where() -> None:
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((4, 5))
    ref = rng.standard_normal((4, 5))
    coords = {"lat": np.arange(4), "lon": np.arange(5)}
    ds_p = xr.Dataset({"x": (("lat", "lon"), pred)}, coords=coords)
    ds_r = xr.Dataset({"x": (("lat", "lon"), ref)}, coords=coords)
    mask = xr.DataArray(
        np.array([[True] * 3 + [False] * 2] * 4),
        dims=("lat", "lon"),
        coords=coords,
    )
    inner = RMSE("x", ("lat", "lon"))
    op = MaskedMetric(inner, mask=mask)
    out = op(ds_p, ds_r)
    expected = inner(ds_p.where(mask), ds_r.where(mask))
    np.testing.assert_allclose(out.values, expected.values)


def test_masked_metric_per_call_mask_overrides() -> None:
    rng = np.random.default_rng(1)
    pred = rng.standard_normal((6,))
    ref = rng.standard_normal((6,))
    ds_p = xr.Dataset({"x": (("t",), pred)})
    ds_r = xr.Dataset({"x": (("t",), ref)})
    op = MaskedMetric(RMSE("x", "t"))
    mask = xr.DataArray([True, True, False, False, True, True], dims="t")
    out = op(ds_p, ds_r, mask=mask)
    expected = RMSE("x", "t")(ds_p.where(mask), ds_r.where(mask))
    np.testing.assert_allclose(out.values, expected.values)


def test_masked_metric_no_mask_raises() -> None:
    op = MaskedMetric(RMSE("x", "t"))
    ds = xr.Dataset({"x": (("t",), np.zeros(3))})
    with pytest.raises(ValueError, match="requires a mask"):
        op(ds, ds)


def test_masked_metric_get_config_json_safe() -> None:
    op = MaskedMetric(RMSE("x", "t"))
    cfg = op.get_config()
    assert cfg["metric"]["class"] == "RMSE"
    assert cfg["mask"] is None
    assert json.loads(json.dumps(cfg)) == cfg


def test_masked_metric_layer0_function() -> None:
    rng = np.random.default_rng(2)
    ds_p = xr.Dataset({"x": (("t",), rng.standard_normal(8))})
    ds_r = xr.Dataset({"x": (("t",), rng.standard_normal(8))})
    mask = xr.DataArray([True] * 4 + [False] * 4, dims="t")
    out = masked_metric(ds_p, ds_r, metric=RMSE("x", "t"), mask=mask)
    expected = RMSE("x", "t")(ds_p.where(mask), ds_r.where(mask))
    np.testing.assert_allclose(out.values, expected.values)


# =========================================================================
# V2.3 — Distributional
# =========================================================================


@pytest.fixture
def ensemble_pair() -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(42)
    truth = rng.standard_normal((10,))
    ens = truth[None, :] + rng.standard_normal((20, 10))  # 20 members
    ds_e = xr.Dataset(
        {"x": (("member", "t"), ens)},
        coords={"member": np.arange(20), "t": np.arange(10)},
    )
    ds_r = xr.Dataset({"x": (("t",), truth)}, coords={"t": np.arange(10)})
    return ds_e, ds_r


def test_crps_against_xskillscore(ensemble_pair) -> None:
    import xskillscore as xs

    ds_e, ds_r = ensemble_pair
    op = CRPS("x")
    out = op(ds_e, ds_r)
    expected = xs.crps_ensemble(ds_r["x"], ds_e["x"], member_dim="member", dim=[])
    np.testing.assert_allclose(out.values, expected.values)


def test_energy_distance_identical_samples_zero() -> None:
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((50, 4))
    ds_a = xr.Dataset({"x": (("member", "t"), samples)})
    ds_b = xr.Dataset({"x": (("member", "t"), samples.copy())})
    out = EnergyDistance("x")(ds_a, ds_b)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-10)


def test_energy_distance_separated_distributions() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal((500, 1))
    b = rng.standard_normal((500, 1)) + 5.0
    ds_a = xr.Dataset({"x": (("member", "t"), a)})
    ds_b = xr.Dataset({"x": (("member", "t"), b)})
    out = EnergyDistance("x")(ds_a, ds_b)
    # Well-separated: energy distance grows with the gap.
    assert float(out.values.item()) > 1.0


def test_wasserstein_1_identical_samples_zero() -> None:
    rng = np.random.default_rng(5)
    samples = rng.standard_normal((50, 3))
    ds = xr.Dataset({"x": (("member", "t"), samples)})
    out = Wasserstein1("x")(ds, ds)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-10)


def test_wasserstein_1_recovers_shift() -> None:
    rng = np.random.default_rng(6)
    a = rng.standard_normal((1000, 1))
    b = a + 3.0
    ds_a = xr.Dataset({"x": (("member", "t"), a)})
    ds_b = xr.Dataset({"x": (("member", "t"), b)})
    out = Wasserstein1("x")(ds_a, ds_b)
    np.testing.assert_allclose(out.values, 3.0, atol=0.1)


def test_distributional_get_config_json() -> None:
    cfg = CRPS("x", ensemble_dim="ens", dims=("t",)).get_config()
    assert cfg == {"variable": "x", "ensemble_dim": "ens", "dims": ["t"]}
    assert json.loads(json.dumps(cfg)) == cfg


# =========================================================================
# V2.2 — Probabilistic
# =========================================================================


def test_spread_skill_ratio_calibrated_ensemble() -> None:
    """Calibrated ensemble: truth and members drawn from the same dist.

    Population spread/skill = sqrt(N/(N+1)).
    """
    rng = np.random.default_rng(7)
    n_t = 5000
    n_m = 30
    truth = rng.standard_normal(n_t)
    members = rng.standard_normal((n_m, n_t))
    ds_e = xr.Dataset(
        {"x": (("member", "t"), members)},
        coords={"member": np.arange(n_m), "t": np.arange(n_t)},
    )
    ds_r = xr.Dataset({"x": (("t",), truth)}, coords={"t": np.arange(n_t)})
    out = SpreadSkillRatio("x")(ds_e, ds_r)
    expected = np.sqrt(n_m / (n_m + 1))
    np.testing.assert_allclose(float(out.values), expected, rtol=0.05)


def test_rank_histogram_calibrated_ensemble_uniform() -> None:
    rng = np.random.default_rng(8)
    n_t, n_m = 4000, 20
    truth = rng.standard_normal(n_t)
    ens = rng.standard_normal((n_m, n_t))
    ds_e = xr.Dataset({"x": (("member", "t"), ens)})
    ds_r = xr.Dataset({"x": (("t",), truth)})
    out = RankHistogram("x")(ds_e, ds_r)
    counts = out["rank_count"].values
    expected = n_t / (n_m + 1)
    # χ²-like sanity: every bin within ~30% of expected.
    assert (np.abs(counts - expected) / expected < 0.3).all()


def test_ensemble_coverage_calibrated() -> None:
    rng = np.random.default_rng(9)
    n_t, n_m = 5000, 50
    truth = rng.standard_normal(n_t)
    ens = rng.standard_normal((n_m, n_t))
    ds_e = xr.Dataset({"x": (("member", "t"), ens)})
    ds_r = xr.Dataset({"x": (("t",), truth)})
    out = EnsembleCoverage("x", q=(0.05, 0.95))(ds_e, ds_r)
    np.testing.assert_allclose(float(out.values), 0.9, atol=0.05)


def test_reliability_curve_perfectly_calibrated() -> None:
    rng = np.random.default_rng(10)
    n = 20000
    p = rng.uniform(size=n)
    o = (rng.uniform(size=n) < p).astype(float)
    ds_p = xr.Dataset({"x": (("i",), p)})
    ds_o = xr.Dataset({"x": (("i",), o)})
    out = ReliabilityCurve("x")(ds_p, ds_o)
    fc = out["forecast_probability"].values
    obf = out["observed_frequency"].values
    # Observed frequency tracks forecast probability bin-by-bin.
    np.testing.assert_allclose(obf, fc, atol=0.03)


def test_probabilistic_missing_member_dim_raises() -> None:
    ds_e = xr.Dataset({"x": (("t",), np.zeros(5))})
    ds_r = xr.Dataset({"x": (("t",), np.zeros(5))})
    with pytest.raises(ValueError, match="ensemble dim"):
        SpreadSkillRatio("x")(ds_e, ds_r)


def test_ensemble_coverage_invalid_q() -> None:
    rng = np.random.default_rng(11)
    ds_e = xr.Dataset({"x": (("member", "t"), rng.standard_normal((5, 3)))})
    ds_r = xr.Dataset({"x": (("t",), np.zeros(3))})
    with pytest.raises(ValueError, match="q must be"):
        EnsembleCoverage("x", q=(0.95, 0.05))(ds_e, ds_r)


# =========================================================================
# V2.1 — Structural
# =========================================================================


def test_gradient_difference_constant_field_zero() -> None:
    """Constant fields have zero gradient — gradient difference is 0."""
    n = 8
    coords = {"lat": np.arange(n), "lon": np.arange(n)}
    ds_p = xr.Dataset({"x": (("lat", "lon"), np.full((n, n), 3.0))}, coords=coords)
    ds_r = xr.Dataset({"x": (("lat", "lon"), np.full((n, n), 7.0))}, coords=coords)
    out = GradientDifference("x", ("lat", "lon"))(ds_p, ds_r)
    np.testing.assert_allclose(float(out.values), 0.0, atol=1e-10)


def test_gradient_difference_step_proportional() -> None:
    n = 16
    coords = {"lat": np.arange(n), "lon": np.arange(n)}
    flat = np.zeros((n, n))
    step = np.zeros((n, n))
    step[:, n // 2 :] = 1.0
    ds_p = xr.Dataset({"x": (("lat", "lon"), flat)}, coords=coords)
    ds_r = xr.Dataset({"x": (("lat", "lon"), step)}, coords=coords)
    out_step = float(GradientDifference("x", ("lat", "lon"))(ds_p, ds_r).values)
    ds_r2 = xr.Dataset({"x": (("lat", "lon"), 2.0 * step)}, coords=coords)
    out_step2 = float(GradientDifference("x", ("lat", "lon"))(ds_p, ds_r2).values)
    np.testing.assert_allclose(out_step2, 2.0 * out_step, rtol=1e-6)


def test_phase_shift_error_recovers_1d_shift() -> None:
    n = 64
    x = np.arange(n)
    sig = np.sin(2 * np.pi * x / 16)
    shift = 5
    sig_shifted = np.roll(sig, shift)
    ds_p = xr.Dataset({"x": (("t",), sig_shifted)}, coords={"t": x})
    ds_r = xr.Dataset({"x": (("t",), sig)}, coords={"t": x})
    out = PhaseShiftError("x", "t", periodic=True)(ds_p, ds_r)
    assert int(out["shift_t"].values) == shift
    np.testing.assert_allclose(float(out["residual_rmse"].values), 0.0, atol=1e-10)


def test_phase_shift_error_2d_periodic() -> None:
    """Use a single Gaussian blob — non-periodic feature → unique correlation peak."""
    n = 32
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    blob = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
    shifted = np.roll(blob, shift=(3, -4), axis=(0, 1))
    coords = {"lat": np.arange(n), "lon": np.arange(n)}
    ds_p = xr.Dataset({"x": (("lat", "lon"), shifted)}, coords=coords)
    ds_r = xr.Dataset({"x": (("lat", "lon"), blob)}, coords=coords)
    out = phase_shift_error(
        ds_p, ds_r, variable="x", dims=("lat", "lon"), periodic=False
    )
    assert int(out["shift_lat"].values) == 3
    assert int(out["shift_lon"].values) == -4


def test_centroid_displacement_paired_blobs() -> None:
    n = 20
    coords = {"lat": np.arange(n, dtype=float), "lon": np.arange(n, dtype=float)}
    lab_p = np.zeros((n, n), dtype=np.int64)
    lab_r = np.zeros((n, n), dtype=np.int64)
    # Object 1: 3x3 block at (5,5) in pred, (5,7) in ref → lon shift = -2
    lab_p[5:8, 5:8] = 1
    lab_r[5:8, 7:10] = 1
    # Object 2: only in pred — should be dropped.
    lab_p[15:17, 15:17] = 2
    op_p = xr.Dataset({"label": (("lat", "lon"), lab_p)}, coords=coords)
    op_r = xr.Dataset({"label": (("lat", "lon"), lab_r)}, coords=coords)
    out = CentroidDisplacement(("lat", "lon"))(op_p, op_r)
    assert out.sizes["object"] == 1
    assert int(out["object_id"].values[0]) == 1
    np.testing.assert_allclose(float(out["displacement_lon"].values[0]), -2.0)
    np.testing.assert_allclose(float(out["displacement_lat"].values[0]), 0.0)


def test_centroid_displacement_no_common_objects() -> None:
    n = 10
    coords = {"lat": np.arange(n, dtype=float), "lon": np.arange(n, dtype=float)}
    lab_p = np.zeros((n, n), dtype=np.int64)
    lab_r = np.zeros((n, n), dtype=np.int64)
    lab_p[1, 1] = 1
    lab_r[5, 5] = 2
    op_p = xr.Dataset({"label": (("lat", "lon"), lab_p)}, coords=coords)
    op_r = xr.Dataset({"label": (("lat", "lon"), lab_r)}, coords=coords)
    out = CentroidDisplacement(("lat", "lon"))(op_p, op_r)
    assert out.sizes["object"] == 0


def test_ssim_identical_inputs_one() -> None:
    rng = np.random.default_rng(12)
    img = rng.standard_normal((32, 32))
    coords = {"lat": np.arange(32), "lon": np.arange(32)}
    ds = xr.Dataset({"x": (("lat", "lon"), img)}, coords=coords)
    out = SSIM("x", ("lat", "lon"))(ds, ds)
    np.testing.assert_allclose(float(out.values), 1.0, atol=1e-10)


def test_ssim_uncorrelated_noise_low() -> None:
    rng = np.random.default_rng(13)
    coords = {"lat": np.arange(32), "lon": np.arange(32)}
    ds_p = xr.Dataset(
        {"x": (("lat", "lon"), rng.standard_normal((32, 32)))}, coords=coords
    )
    ds_r = xr.Dataset(
        {"x": (("lat", "lon"), rng.standard_normal((32, 32)))}, coords=coords
    )
    out = SSIM("x", ("lat", "lon"))(ds_p, ds_r)
    assert abs(float(out.values)) < 0.3


def test_structural_get_config_json_safe() -> None:
    cfg = SSIM("x", ("lat", "lon"), window=7).get_config()
    assert json.loads(json.dumps(cfg)) == cfg
    cfg2 = PhaseShiftError("x", ("lat", "lon"), periodic=True).get_config()
    assert json.loads(json.dumps(cfg2)) == cfg2


# =========================================================================
# Composition (V2.4 inside Graph)
# =========================================================================


def test_masked_metric_inside_graph() -> None:
    from xr_toolz.core import Graph, Input

    rng = np.random.default_rng(14)
    ds_p = xr.Dataset({"x": (("t",), rng.standard_normal(8))})
    ds_r = xr.Dataset({"x": (("t",), rng.standard_normal(8))})
    mask = xr.DataArray([True] * 4 + [False] * 4, dims="t")
    p = Input("p")
    r = Input("r")
    out = MaskedMetric(RMSE("x", "t"), mask=mask)(p, r)
    g = Graph(inputs={"p": p, "r": r}, outputs={"y": out})
    res = g(p=ds_p, r=ds_r)
    expected = RMSE("x", "t")(ds_p.where(mask), ds_r.where(mask))
    np.testing.assert_allclose(res["y"].values, expected.values)
