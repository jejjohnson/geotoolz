"""Tests for V1.3 — FrequencyBandSkill + BandLimitedRMSE.

Covers:
- Two-tone test: error in one band has full RMSE there, ~0 in another.
- Bands above Nyquist emit a warning and produce NaN (no exception).
- ``dims=("lat","lon")`` works with ``degrees_north/east`` coord units.
- get_config / JSON round-trip introspection.
- Inner-Operator constraint mirrors SkillByLeadTime.
- Missing-units error pointer is informative.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.metrics import (
    RMSE,
    BandLimitedRMSE,
    FrequencyBandSkill,
    band_limited_rmse,
    evaluate_by_frequency_band,
)


# ---------- fixtures ------------------------------------------------------


@pytest.fixture
def two_tone() -> tuple[xr.Dataset, xr.Dataset]:
    n = 256
    x = np.arange(n, dtype=float)
    x_da = xr.DataArray(x, dims=("x",), name="x", attrs={"units": "m"})
    # Use integer cycle counts (8 and 64 cycles in 256 samples) so the
    # FFT bins land exactly on the tones — no leakage.
    truth = np.sin(2 * np.pi * (8.0 / n) * x)
    pred = truth + np.sin(2 * np.pi * (64.0 / n) * x)
    ds_pred = xr.Dataset({"phi": (("x",), pred)}, coords={"x": x_da})
    ds_ref = xr.Dataset({"phi": (("x",), truth)}, coords={"x": x_da})
    return ds_pred, ds_ref


@pytest.fixture
def latlon_pair() -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(7)
    lat = xr.DataArray(
        np.linspace(35.0, 45.0, 32), dims=("lat",), attrs={"units": "degrees_north"}
    )
    lon = xr.DataArray(
        np.linspace(-30.0, -10.0, 64), dims=("lon",), attrs={"units": "degrees_east"}
    )
    ref_arr = rng.standard_normal((lat.size, lon.size))
    pred_arr = ref_arr + 0.1 * rng.standard_normal((lat.size, lon.size))
    ds_pred = xr.Dataset(
        {"ssh": (("lat", "lon"), pred_arr)}, coords={"lat": lat, "lon": lon}
    )
    ds_ref = xr.Dataset(
        {"ssh": (("lat", "lon"), ref_arr)}, coords={"lat": lat, "lon": lon}
    )
    return ds_pred, ds_ref


# ---------- core behaviour ------------------------------------------------


def test_two_tone_band_isolation(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    out = band_limited_rmse(
        ds_pred,
        ds_ref,
        variable="phi",
        bands={"low": (0.0, 0.10), "high": (0.20, 0.40)},
        dims=("x",),
    )
    assert "band" in out.dims
    assert out.sizes["band"] == 2
    rmse_low = float(out["phi"].sel(band="low").values)
    rmse_high = float(out["phi"].sel(band="high").values)
    # Error is a pure 0.30 cycle/m tone with unit amplitude → RMS ~ 1/sqrt(2).
    assert rmse_low < 1e-6
    assert rmse_high == pytest.approx(1.0 / np.sqrt(2.0), abs=5e-3)


def test_band_low_high_coords_attached(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    out = band_limited_rmse(
        ds_pred,
        ds_ref,
        variable="phi",
        bands={"a": (0.0, 0.1), "b": (0.2, 0.4)},
        dims=("x",),
    )
    assert list(out.coords["band_low"].values) == [0.0, 0.2]
    assert list(out.coords["band_high"].values) == [0.1, 0.4]


# ---------- Nyquist handling ----------------------------------------------


def test_band_above_nyquist_warns_and_returns_nan(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    with pytest.warns(UserWarning, match="Nyquist"):
        out = band_limited_rmse(
            ds_pred,
            ds_ref,
            variable="phi",
            bands={"sane": (0.0, 0.1), "wild": (10.0, 20.0)},
            dims=("x",),
        )
    assert np.isfinite(float(out["phi"].sel(band="sane").values))
    assert np.isnan(float(out["phi"].sel(band="wild").values))


# ---------- 2-D lat/lon spatial bands -------------------------------------


def test_latlon_dims_with_degrees_units(latlon_pair) -> None:
    ds_pred, ds_ref = latlon_pair
    out = band_limited_rmse(
        ds_pred,
        ds_ref,
        variable="ssh",
        bands={
            "basin": (0.0, 1.0 / 1000.0),  # > 1000 km
            "meso": (1.0 / 500.0, 1.0 / 50.0),  # 50–500 km
        },
        dims=("lat", "lon"),
    )
    assert "band" in out.dims
    vals = out["ssh"].values
    assert np.all(np.isfinite(vals))


def test_coord_spacing_override_bypasses_units(two_tone) -> None:
    # Strip units; expect ValueError without override.
    ds_pred, ds_ref = two_tone
    ds_pred = ds_pred.copy()
    ds_ref = ds_ref.copy()
    ds_pred["x"].attrs.pop("units", None)
    ds_ref["x"].attrs.pop("units", None)
    with pytest.raises(ValueError, match="units"):
        band_limited_rmse(
            ds_pred,
            ds_ref,
            variable="phi",
            bands={"a": (0.0, 0.1)},
            dims=("x",),
        )
    out = band_limited_rmse(
        ds_pred,
        ds_ref,
        variable="phi",
        bands={"a": (0.0, 0.1)},
        dims=("x",),
        coord_spacing={"x": 1.0},
    )
    assert np.isfinite(float(out["phi"].sel(band="a").values))


# ---------- Operator interface --------------------------------------------


def test_frequency_band_skill_default_metric_matches_rmse(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    op = FrequencyBandSkill(
        variable="phi",
        dims=("x",),
        bands={"low": (0.0, 0.1), "high": (0.2, 0.4)},
    )
    out = op(ds_pred, ds_ref)
    expected = band_limited_rmse(
        ds_pred,
        ds_ref,
        variable="phi",
        bands={"low": (0.0, 0.1), "high": (0.2, 0.4)},
        dims=("x",),
    )
    xr.testing.assert_allclose(out, expected)


def test_band_limited_rmse_subclass_is_frequency_band_skill() -> None:
    op = BandLimitedRMSE("phi", ("x",), {"a": (0.0, 0.1)})
    assert isinstance(op, FrequencyBandSkill)


def test_inner_metric_must_be_operator() -> None:
    with pytest.raises(TypeError, match="Operator"):
        FrequencyBandSkill(
            variable="phi",
            dims=("x",),
            bands={"a": (0.0, 0.1)},
            metric=lambda p, _r: p,  # type: ignore[arg-type]  # bare callable rejected
        )


def test_get_config_round_trips() -> None:
    op = FrequencyBandSkill(
        variable="phi",
        dims=("x",),
        bands={"low": (0.0, 0.1), "high": (0.2, 0.4)},
        metric=RMSE("phi", dims=("x",)),
    )
    cfg = op.get_config()
    payload = json.dumps(cfg)
    round_trip = json.loads(payload)
    assert round_trip["variable"] == "phi"
    assert round_trip["dims"] == ["x"]
    assert round_trip["bands"]["low"] == [0.0, 0.1]
    assert round_trip["metric"]["class"] == "RMSE"


# ---------- band validation -----------------------------------------------


def test_invalid_band_raises() -> None:
    with pytest.raises(ValueError, match="0 <= low < high"):
        FrequencyBandSkill("phi", ("x",), {"bad": (0.5, 0.5)})
    with pytest.raises(ValueError, match="non-empty"):
        FrequencyBandSkill("phi", ("x",), {})


def test_default_metric_is_serialised_in_get_config() -> None:
    op = FrequencyBandSkill("phi", ("x",), {"a": (0.0, 0.1)})
    cfg = op.get_config()
    assert cfg["metric"]["class"] == "RMSE"
    assert json.loads(json.dumps(cfg))["metric"]["class"] == "RMSE"


def test_irregular_coord_raises(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    # Stretch the x-coord so np.diff is non-uniform.
    bad_x = np.cumsum(np.linspace(0.5, 1.5, ds_pred.sizes["x"]))
    ds_pred = ds_pred.assign_coords(x=("x", bad_x))
    ds_ref = ds_ref.assign_coords(x=("x", bad_x))
    ds_pred["x"].attrs["units"] = "m"
    ds_ref["x"].attrs["units"] = "m"
    with pytest.raises(ValueError, match="not uniformly spaced"):
        band_limited_rmse(
            ds_pred,
            ds_ref,
            variable="phi",
            bands={"a": (0.0, 0.1)},
            dims=("x",),
        )


def test_pred_ref_must_share_dim_coords(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    # Shift reference x by 0.5 so the coord arrays disagree.
    ds_ref = ds_ref.assign_coords(x=("x", ds_ref["x"].values + 0.5))
    ds_ref["x"].attrs["units"] = "m"
    with pytest.raises(ValueError, match="prediction and reference disagree"):
        band_limited_rmse(
            ds_pred,
            ds_ref,
            variable="phi",
            bands={"a": (0.0, 0.1)},
            dims=("x",),
        )


def test_bandpass_axis_order_independent_of_dims_order() -> None:
    # Data is stored as (lon, lat); caller passes dims=('lat','lon') —
    # i.e. the non-data axis order. The mask reshape must still align
    # with the data's actual axis positions or the band-pass produces
    # numerically wrong scores without raising.
    n = 64
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, n))
    coords = {
        "lon": xr.DataArray(
            np.arange(n, dtype=float), dims=("lon",), attrs={"units": "m"}
        ),
        "lat": xr.DataArray(
            np.arange(n, dtype=float), dims=("lat",), attrs={"units": "m"}
        ),
    }
    ds_lonfirst = xr.Dataset({"phi": (("lon", "lat"), arr)}, coords=coords)
    ds_latfirst = xr.Dataset({"phi": (("lat", "lon"), arr.T)}, coords=coords)

    bands = {"low": (0.0, 0.05), "high": (0.05, 0.5)}
    out_lonfirst = band_limited_rmse(
        ds_lonfirst.copy(deep=True),
        ds_lonfirst,
        variable="phi",
        bands=bands,
        dims=("lat", "lon"),
    )
    out_latfirst = band_limited_rmse(
        ds_latfirst.copy(deep=True),
        ds_latfirst,
        variable="phi",
        bands=bands,
        dims=("lat", "lon"),
    )
    # pred==ref in both cases → both must be all-zero RMSE per band.
    np.testing.assert_allclose(out_lonfirst["phi"].values, 0.0, atol=1e-12)
    np.testing.assert_allclose(out_latfirst["phi"].values, 0.0, atol=1e-12)


def test_evaluate_by_frequency_band_passes_through_extra_metric(two_tone) -> None:
    ds_pred, ds_ref = two_tone
    out = evaluate_by_frequency_band(
        ds_pred,
        ds_ref,
        variable="phi",
        bands={"a": (0.0, 0.1)},
        dims=("x",),
        metric=RMSE("phi", dims=("x",)),
    )
    assert "band" in out.dims
