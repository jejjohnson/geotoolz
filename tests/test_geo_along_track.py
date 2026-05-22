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
        ds,
        dim="num_lines",
        lambda_min_km=20.0,
        lambda_max_km=100.0,
        spacing_km=5.0,
    )

    assert isinstance(out, type(ds))
    assert calls["btype"] == "bandpass"
    assert calls["cutoff"] == pytest.approx((0.1, 0.5))


def test_bandpass_wavelength_lambda_min_only_is_low_pass(monkeypatch):
    """``lambda_min_km`` alone keeps wavelengths >= the bound, i.e.
    cuts short wavelengths — that's a low-pass on the frequency axis."""
    ds = _track_dataset()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        along_track, "fir_filter", lambda ds, **k: (calls.update(k), ds)[1]
    )
    along_track.bandpass_wavelength(
        ds, dim="num_lines", lambda_min_km=20.0, spacing_km=5.0
    )
    assert calls["btype"] == "low"
    assert calls["cutoff"] == pytest.approx(0.5)


def test_bandpass_wavelength_lambda_max_only_is_high_pass(monkeypatch):
    """``lambda_max_km`` alone keeps wavelengths <= the bound, i.e.
    cuts long wavelengths — that's a high-pass on the frequency axis."""
    ds = _track_dataset()
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        along_track, "fir_filter", lambda ds, **k: (calls.update(k), ds)[1]
    )
    along_track.bandpass_wavelength(
        ds, dim="num_lines", lambda_max_km=100.0, spacing_km=5.0
    )
    assert calls["btype"] == "high"
    assert calls["cutoff"] == pytest.approx(0.1)


def test_bandpass_wavelength_raises_on_multidim_lon_lat_without_spacing():
    """Multi-track / 2-D coords would mix endpoints in the median, so
    spacing inference must refuse and ask for ``spacing_km`` explicitly."""
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
        bandpass_wavelength(ds, dim="num_lines", lambda_min_km=20.0)


def test_bandpass_wavelength_raises_on_misspelled_dim_with_explicit_spacing():
    """Regression: a misspelled ``dim`` used to silently pass every variable
    through after the PR β primitive flip (every var lacked the bad dim so
    the loop continued); now it raises like the original Dataset-flavoured
    ``fir_filter`` did."""
    ds = _track_dataset()
    with pytest.raises(ValueError, match="not in Dataset dims"):
        bandpass_wavelength(
            ds,
            dim="num_linez",  # typo
            lambda_min_km=20.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_raises_below_nyquist():
    ds = _track_dataset()
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_wavelength(
            ds,
            dim="num_lines",
            lambda_min_km=8.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_validates_bounds():
    ds = _track_dataset()
    with pytest.raises(ValueError, match="at least one"):
        bandpass_wavelength(ds, dim="num_lines", spacing_km=5.0)
    with pytest.raises(ValueError, match="lambda_min_km must be <"):
        bandpass_wavelength(
            ds,
            dim="num_lines",
            lambda_min_km=100.0,
            lambda_max_km=20.0,
            spacing_km=5.0,
        )


def test_bandpass_wavelength_passes_through_non_numeric_and_no_dim():
    ds = _track_dataset()
    out = bandpass_wavelength(
        ds,
        dim="num_lines",
        lambda_min_km=20.0,
        lambda_max_km=100.0,
        spacing_km=5.0,
        num_taps=15,
    )

    np.testing.assert_array_equal(out["label"].values, ds["label"].values)
    assert float(out["static"]) == 1.0
    assert out["sla"].shape == ds["sla"].shape


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


def test_bandpass_wavelength_default_lon_lat_lookup():
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
    out = bandpass_wavelength(
        ds,
        dim="num_lines",
        lambda_min_km=20.0,
        num_taps=15,
    )
    assert out["sla"].shape == ds["sla"].shape
