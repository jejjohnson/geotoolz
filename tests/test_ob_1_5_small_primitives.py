"""OB-1.5 — ``nrmse_score`` (Mercator/OceanBench skill) + ``get_dataset_resolution``.

Issue: https://github.com/jejjohnson/xrtoolz/issues/136

Pins:

- ``nrmse_score = 1 - RMSE / std(ref)`` matches the upstream
  ``nrmse_ds`` reference formula exactly.
- ``nrmse_score`` diverges from the existing ``nrmse`` (which uses
  ``sqrt(<ref^2>)`` in the denominator) on non-zero-mean references.
- The ``NRMSEScore`` Operator round-trips through ``get_config`` and
  matches the Tier-B primitive.
- ``get_dataset_resolution`` classifies 1°, 1/4°, 1/12° lat/lon grids
  correctly and falls back to ``"other"`` for non-matching spacings.
- ``rtol`` is honoured — a near-canonical spacing that exceeds the
  tolerance falls into ``"other"``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.geo import get_dataset_resolution
from xrtoolz.metrics import NRMSEScore, nrmse, nrmse_score
from xrtoolz.metrics.array import nrmse_score as array_nrmse_score


# ---------- nrmse_score ----------------------------------------------------


def test_nrmse_score_array_matches_mercator_formula() -> None:
    """Tier A: ``1 - RMSE / std(ref)``."""
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((4, 32))
    ref = rng.standard_normal((4, 32))

    expected = 1.0 - np.sqrt(np.nanmean((pred - ref) ** 2, axis=-1)) / np.nanstd(
        ref, axis=-1
    )
    out = array_nrmse_score(pred, ref, axis=-1)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_nrmse_score_perfect_prediction_is_one() -> None:
    """Predicting the reference exactly yields a score of 1."""
    ref = np.linspace(0.0, 1.0, 64)
    np.testing.assert_allclose(array_nrmse_score(ref, ref, axis=-1), 1.0)


def test_nrmse_score_diverges_from_nrmse_on_nonzero_mean_reference() -> None:
    """The two metrics are NOT equivalent on non-zero-mean references.

    ``nrmse`` normalises by ``sqrt(<ref^2>)`` (raw signal magnitude) and
    ``nrmse_score`` normalises by ``std(ref)`` (anomaly magnitude). For
    SST-like references that sit far from zero they differ
    substantially.
    """
    rng = np.random.default_rng(7)
    # SST-like reference: ~ 285 K with a few-K seasonal swing.
    ref = 285.0 + 5.0 * rng.standard_normal((128,))
    pred = ref + 0.5 * rng.standard_normal((128,))

    pred_da = xr.DataArray(pred, dims=("time",))
    ref_da = xr.DataArray(ref, dims=("time",))

    score = float(nrmse_score(pred_da, ref_da, dim="time").values)
    other = float(nrmse(pred_da, ref_da, dim="time").values)

    # ``nrmse_score`` normalises by std (~5 K here); ``nrmse`` normalises
    # by raw magnitude (~285 K). The two scores must differ materially —
    # if they ever became identical, this assertion catches it.
    assert abs(score - other) > 0.05


def test_nrmse_score_operator_matches_function() -> None:
    """Layer-1 ``NRMSEScore`` selects the variable and forwards to Tier B."""
    rng = np.random.default_rng(3)
    pred = rng.standard_normal((6, 24))
    ref = rng.standard_normal((6, 24))
    ds_pred = xr.Dataset({"x": (("a", "time"), pred)})
    ds_ref = xr.Dataset({"x": (("a", "time"), ref)})

    op = NRMSEScore(variable="x", dims="time")
    op_out = op(ds_pred, ds_ref)
    fn_out = nrmse_score(ds_pred["x"], ds_ref["x"], dim="time")

    np.testing.assert_allclose(op_out.values, fn_out.values, rtol=1e-12, atol=1e-12)


def test_nrmse_score_operator_get_config_roundtrips() -> None:
    op = NRMSEScore(variable="x", dims=("a", "time"))
    cfg = op.get_config()
    # JSON-serialisable
    assert json.loads(json.dumps(cfg)) == cfg
    assert cfg == {"variable": "x", "dims": ["a", "time"]}


def test_nrmse_score_constant_reference_returns_one_for_perfect_prediction() -> None:
    """Regression (Copilot review): a constant reference (``std=0``) used
    to drive ``0/0 → NaN``. The documented contract is that perfect
    prediction yields 1 on any reference, so we branch explicitly on
    the zero-std case."""
    ref = np.full(32, 5.0)
    score = array_nrmse_score(ref, ref, axis=-1)
    np.testing.assert_allclose(score, 1.0)


def test_nrmse_score_constant_reference_returns_neg_inf_for_nonzero_error() -> None:
    """Non-zero error against a constant reference is infinitely bad —
    no variability to normalise against."""
    ref = np.full(32, 5.0)
    pred = ref + 1.0
    score = float(array_nrmse_score(pred, ref, axis=-1))
    assert score == -np.inf


# ---------- get_dataset_resolution ----------------------------------------


def _grid(spacing: float, n: int = 9) -> xr.Dataset:
    lat = np.arange(-((n - 1) // 2), ((n - 1) // 2) + 1, dtype=float) * spacing
    lon = np.arange(0.0, n * spacing, spacing, dtype=float)
    return xr.Dataset(
        {"x": (("lat", "lon"), np.zeros((n, n)))},
        coords={"lat": lat, "lon": lon},
    )


def test_get_dataset_resolution_one_degree() -> None:
    assert get_dataset_resolution(_grid(1.0)) == "one_degree"


def test_get_dataset_resolution_quarter_degree() -> None:
    assert get_dataset_resolution(_grid(0.25)) == "quarter_degree"


def test_get_dataset_resolution_twelfth_degree() -> None:
    assert get_dataset_resolution(_grid(1.0 / 12.0)) == "twelfth_degree"


def test_get_dataset_resolution_other_for_unknown_spacing() -> None:
    # 0.5° doesn't match any of the canonical targets.
    assert get_dataset_resolution(_grid(0.5)) == "other"


def test_get_dataset_resolution_respects_rtol() -> None:
    """A spacing slightly off the canonical target falls into ``other``
    when ``rtol`` is tighter than the offset."""
    # 1.05° is ~5% off 1° — accepted by the default rtol=0.05, but a
    # tighter rtol=0.01 should reject it.
    ds = _grid(1.05)
    assert get_dataset_resolution(ds, rtol=0.10) == "one_degree"
    assert get_dataset_resolution(ds, rtol=0.01) == "other"


def test_get_dataset_resolution_rejects_anisotropic_grid() -> None:
    """Regression (Codex P2 review): a grid with ``dlon=1.1`` and
    ``dlat=0.9`` averages to ``1.0`` but neither axis is canonical —
    must classify as ``"other"`` rather than ``"one_degree"``."""
    lat = np.arange(-4.5, 4.5, 0.9)
    lon = np.arange(0.0, 11.0, 1.1)
    ds = xr.Dataset(
        {"x": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )
    assert get_dataset_resolution(ds) == "other"


def test_get_dataset_resolution_rejects_curvilinear_grid() -> None:
    """Regression (Copilot review): a 2-D lon/lat coord used to silently
    classify via ``np.diff`` running along the last axis. Now raises
    ``ValueError`` so the caller knows to regrid first."""
    n = 6
    lon2d = np.tile(np.arange(n, dtype=float), (n, 1))
    lat2d = np.tile(np.arange(n, dtype=float)[:, None], (1, n))
    ds = xr.Dataset(
        {"x": (("y", "x"), np.zeros((n, n)))},
        coords={
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
        },
    )
    with pytest.raises(ValueError, match="1-D"):
        get_dataset_resolution(ds)
