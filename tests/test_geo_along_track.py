"""Tests for along-track wavelength-domain filters."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pyproj import Geod

from xrtoolz.geo import BandpassWavelength, bandpass_wavelength, median_dx_km
from xrtoolz.geo._src import along_track


def _track_dataset(n: int = 256) -> xr.Dataset:
    x = np.arange(n)
    signal = np.sin(2 * np.pi * x / 20) + 0.2 * np.sin(2 * np.pi * x / 4)
    return xr.Dataset(
        {
            "sla": (("num_lines",), signal),
            "label": (("num_lines",), np.array(["track"] * n)),
            "static": ((), 1.0),
        },
        coords={
            "num_lines": x,
            "lon": (("num_lines",), -45.0 + 0.1 * x),
            "lat": (("num_lines",), np.full(n, 10.0)),
        },
    )


def test_median_dx_km_matches_pyproj_segment():
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 0.0, 0.0])
    expected = np.median(Geod(ellps="WGS84").line_lengths(lon, lat)) / 1000.0
    assert median_dx_km(lon, lat) == pytest.approx(expected, rel=1e-3)


def test_bandpass_wavelength_translates_cutoffs(monkeypatch):
    ds = _track_dataset()
    calls: dict[str, object] = {}

    def fake_fir_filter(da, **kwargs):
        calls.update(kwargs)
        return da

    monkeypatch.setattr(along_track, "fir_filter", fake_fir_filter)

    out = along_track.bandpass_wavelength(
        ds["sla"],
        dim="num_lines",
        lambda_min_km=20.0,
        lambda_max_km=100.0,
        spacing_km=5.0,
    )

    assert isinstance(out, xr.DataArray)
    assert calls["btype"] == "bandpass"
    assert calls["cutoff"] == pytest.approx((0.1, 0.5))


def test_bandpass_wavelength_lambda_min_only_is_low_pass(monkeypatch):
    """``lambda_min_km`` alone keeps wavelengths >= the bound, i.e.
    cuts short wavelengths — that's a low-pass on the frequency axis."""
    ds = _track_dataset()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        along_track, "fir_filter", lambda da, **k: (calls.update(k), da)[1]
    )
    along_track.bandpass_wavelength(
        ds["sla"], dim="num_lines", lambda_min_km=20.0, spacing_km=5.0
    )
    assert calls["btype"] == "low"
    assert calls["cutoff"] == pytest.approx(0.5)


def test_bandpass_wavelength_lambda_max_only_is_high_pass(monkeypatch):
    """``lambda_max_km`` alone keeps wavelengths <= the bound, i.e.
    cuts long wavelengths — that's a high-pass on the frequency axis."""
    ds = _track_dataset()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        along_track, "fir_filter", lambda da, **k: (calls.update(k), da)[1]
    )
    along_track.bandpass_wavelength(
        ds["sla"], dim="num_lines", lambda_max_km=100.0, spacing_km=5.0
    )
    assert calls["btype"] == "high"
    assert calls["cutoff"] == pytest.approx(0.1)


def test_bandpass_wavelength_raises_on_multidim_lon_lat_without_spacing():
    """Multi-track / 2-D coords would mix endpoints in the median, so
    spacing inference must refuse and ask for ``spacing_km`` explicitly.

    Exercised at the Operator boundary now — the operator looks up the
    lon/lat DataArrays from the Dataset before calling the primitive.
    """
    n = 16
    ds = xr.Dataset(
        {"sla": (("track", "num_lines"), np.zeros((2, n)))},
        coords={
            "track": [0, 1],
            "num_lines": np.arange(n),
            "lon": (("track", "num_lines"), np.tile(np.arange(n) * 0.05, (2, 1))),
            "lat": (("track", "num_lines"), np.zeros((2, n))),
        },
    )
    with pytest.raises(ValueError, match="not 1-D along"):
        BandpassWavelength(dim="num_lines", lambda_min_km=20.0)(ds)


def test_bandpass_wavelength_operator_raises_on_misspelled_dim():
    """Regression: a misspelled ``dim`` used to silently pass every
    variable through after the PR β primitive flip. With PR γ the
    primitive is DataArray-only, but the Operator still enforces the
    Dataset dim check so callers get a clear failure rather than a
    no-op pass-through."""
    ds = _track_dataset()
    with pytest.raises(ValueError, match="not in Dataset dims"):
        BandpassWavelength(
            dim="num_linez",  # typo
            lambda_min_km=20.0,
            spacing_km=5.0,
        )(ds)


def test_bandpass_wavelength_primitive_raises_on_misspelled_dim():
    """The DataArray-only primitive also raises on a missing ``dim``."""
    ds = _track_dataset()
    with pytest.raises(ValueError, match="not in DataArray dims"):
        bandpass_wavelength(
            ds["sla"],
            dim="num_linez",  # typo
            lambda_min_km=20.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_raises_below_nyquist():
    ds = _track_dataset()
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_wavelength(
            ds["sla"],
            dim="num_lines",
            lambda_min_km=8.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_validates_bounds():
    ds = _track_dataset()
    with pytest.raises(ValueError, match="at least one"):
        bandpass_wavelength(ds["sla"], dim="num_lines", spacing_km=5.0)
    with pytest.raises(ValueError, match="lambda_min_km must be <"):
        bandpass_wavelength(
            ds["sla"],
            dim="num_lines",
            lambda_min_km=100.0,
            lambda_max_km=20.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_operator_passes_through_non_numeric_and_no_dim():
    """The Dataset pass-through / per-variable loop now lives on the
    Operator wrapper. Non-numeric variables and variables that don't
    carry the filter dim pass through unchanged."""
    ds = _track_dataset()
    out = BandpassWavelength(
        dim="num_lines",
        lambda_min_km=20.0,
        lambda_max_km=100.0,
        spacing_km=5.0,
        num_taps=15,
    )(ds)

    np.testing.assert_array_equal(out["label"].values, ds["label"].values)
    assert float(out["static"]) == 1.0
    assert out["sla"].shape == ds["sla"].shape


def test_bandpass_wavelength_primitive_infers_spacing_from_dataarrays():
    """Regression: the DataArray-positional primitive accepts lon/lat
    DataArrays for spacing inference (previously it received variable-name
    strings)."""
    n = 64
    x = np.arange(n)
    da = xr.DataArray(
        np.sin(2 * np.pi * x / 16),
        dims=("num_lines",),
        coords={
            "num_lines": x,
            "lon": (("num_lines",), 0.05 * x),
            "lat": (("num_lines",), np.zeros(n)),
        },
        name="sla",
    )
    out = bandpass_wavelength(
        da,
        dim="num_lines",
        lambda_min_km=20.0,
        lon=da.coords["lon"],
        lat=da.coords["lat"],
        num_taps=15,
    )
    assert out.shape == da.shape


def test_bandpass_wavelength_operator_config_round_trip():
    op = BandpassWavelength(
        dim="num_lines",
        lambda_min_km=20.0,
        lambda_max_km=100.0,
        spacing_km=5.0,
        method="kaiser",
        num_taps=15,
        attenuation_db=60.0,
    )

    cfg = op.get_config()
    assert cfg == {
        "dim": "num_lines",
        "lambda_min_km": 20.0,
        "lambda_max_km": 100.0,
        "spacing_km": 5.0,
        "method": "kaiser",
        "num_taps": 15,
        "attenuation_db": 60.0,
        "lon": "lon",
        "lat": "lat",
    }
    assert BandpassWavelength(**cfg).get_config() == cfg


def test_bandpass_wavelength_operator_default_lon_lat_lookup():
    """The ``BandpassWavelength`` operator resolves ``lon`` / ``lat``
    variable names from the Dataset before calling the primitive."""
    n = 64
    x = np.arange(n)
    ds = xr.Dataset(
        {"sla": (("num_lines",), np.sin(2 * np.pi * x / 16))},
        coords={
            "num_lines": x,
            "lon": (("num_lines",), 0.05 * x),
            "lat": (("num_lines",), np.zeros(n)),
        },
    )
    out = BandpassWavelength(
        dim="num_lines",
        lambda_min_km=20.0,
        num_taps=15,
    )(ds)
    assert out["sla"].shape == ds["sla"].shape
