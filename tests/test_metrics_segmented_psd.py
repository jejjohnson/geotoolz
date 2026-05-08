"""Tests for gap-tolerant segmented along-track PSD metrics."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.metrics import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
    resolved_scale,
    segment_signal,
    segmented_coherence,
    segmented_psd,
)


def test_segment_signal_counts_windows_and_respects_gaps() -> None:
    x = np.arange(10, dtype=float)
    segments = segment_signal(x, npt=4, overlap=0.5)
    assert segments.shape == ((10 - 4) // 2 + 1, 4)

    gap_segments = segment_signal(x, npt=4, overlap=0.5, gap_indices=[4])
    np.testing.assert_array_equal(gap_segments, np.array([[0, 1, 2, 3], [5, 6, 7, 8]]))

    dropped = segment_signal(
        x,
        npt=4,
        overlap=0.5,
        gap_indices=[4],
        min_segment_length=6,
    )
    assert dropped.shape == (0, 4)


def test_segmented_psd_detects_tone_and_preserves_variance() -> None:
    npt = 128
    fs = 1.0
    x = np.arange(512, dtype=float)
    freq = 8.0 / npt
    tone = np.sin(2.0 * np.pi * freq * x)

    wavenumber, psd = segmented_psd(tone, fs=fs, npt=npt, overlap=0.5)
    df = wavenumber[1] - wavenumber[0]
    assert abs(wavenumber[np.argmax(psd)] - freq) <= df

    _, boxcar_psd = segmented_psd(tone, fs=fs, npt=npt, overlap=0.5, window="boxcar")
    segmented_var = np.mean(np.var(segment_signal(tone, npt=npt, overlap=0.5), axis=1))
    assert np.sum(boxcar_psd) * df == pytest.approx(segmented_var, rel=0.05)


def test_segmented_coherence_identical_and_independent_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048)
    y = rng.standard_normal(2048)
    npt = 128

    _, identical = segmented_coherence(x, x, fs=1.0, npt=npt, overlap=0.5)
    assert np.nanmean(identical[1:]) == pytest.approx(1.0)

    _, independent = segmented_coherence(x, y, fs=1.0, npt=npt, overlap=0.5)
    n_segments = segment_signal(x, npt=npt, overlap=0.5).shape[0]
    assert np.nanmean(independent[1:]) == pytest.approx(
        1.0 / n_segments, rel=2.0, abs=0.05
    )


def _track_dataset(n: int = 192) -> xr.Dataset:
    rng = np.random.default_rng(4)
    dim = "num_lines"
    ref = rng.standard_normal(n)
    pred = 0.8 * ref
    lon = np.linspace(-20.0, 20.0, n)
    lat = np.linspace(-5.0, 5.0, n)
    seconds = np.arange(n, dtype="timedelta64[s]")
    seconds[96:] += np.timedelta64(10, "s")
    time = np.datetime64("2020-01-01") + seconds
    return xr.Dataset(
        {
            "ssh_ref": (dim, ref),
            "ssh_pred": (dim, pred),
            "lon": (dim, lon),
            "lat": (dim, lat),
            "time": (dim, time),
        }
    )


def _is_near_zero_meridian(lon: float) -> bool:
    return min(lon, 360.0 - lon) < 0.1


def test_along_track_psd_score_detects_time_gaps() -> None:
    ds = _track_dataset()
    out = along_track_psd_score(
        ds,
        var_ref="ssh_ref",
        var_pred="ssh_pred",
        npt=64,
        overlap=0.5,
        spacing_km=7.0,
    )
    no_gap = along_track_psd_score(
        ds,
        var_ref="ssh_ref",
        var_pred="ssh_pred",
        npt=64,
        overlap=0.5,
        max_gap=np.timedelta64(1, "D"),
        spacing_km=7.0,
    )

    assert out.sizes["segment"] == 4
    assert no_gap.sizes["segment"] == 5


def test_along_track_psd_score_bounds() -> None:
    ds = _track_dataset()
    out = along_track_psd_score(
        ds,
        var_ref="ssh_ref",
        var_pred="ssh_pred",
        npt=64,
        overlap=0.5,
        spacing_km=7.0,
    )
    assert float(out["psd_score"].min()) > 0.0
    assert float(out["psd_score"].max()) < 1.0
    assert {"psd_ref", "psd_pred", "psd_err", "psd_score", "coherence"} <= set(
        out.data_vars
    )


def test_along_track_segment_longitude_uses_circular_mean() -> None:
    npt = 64
    dim = "num_lines"
    x = np.arange(npt, dtype=float)
    lon = np.concatenate(
        [np.linspace(358.0, 359.9, npt // 2), np.linspace(0.1, 2.0, npt // 2)]
    )
    ds = xr.Dataset(
        {
            "ssh_ref": (dim, np.sin(2 * np.pi * x / npt)),
            "ssh_pred": (dim, 0.9 * np.sin(2 * np.pi * x / npt)),
            "lon": (dim, lon),
            "lat": (dim, np.zeros(npt)),
            "time": (dim, np.datetime64("2020-01-01") + x.astype("timedelta64[s]")),
        }
    )

    out = along_track_psd_score(
        ds,
        var_ref="ssh_ref",
        var_pred="ssh_pred",
        npt=npt,
        spacing_km=7.0,
    )
    segment_lon = float(out["segment_lon"].item())
    assert _is_near_zero_meridian(segment_lon)


def test_resolved_scale_returns_exact_crossing_wavelength() -> None:
    score = xr.DataArray(
        [0.2, 0.5, 0.8],
        dims=("wavenumber",),
        coords={"wavenumber": [0.01, 0.02, 0.04]},
    )
    assert resolved_scale(score, frequency="wavenumber", level=0.5) == pytest.approx(
        50.0
    )


def test_psd_score_by_region_applies_min_segments_threshold() -> None:
    ds_segments = xr.Dataset(
        {
            "psd_ref": (("segment", "wavenumber"), np.ones((3, 2))),
            "psd_pred": (("segment", "wavenumber"), np.ones((3, 2)) * 0.8),
            "psd_err": (("segment", "wavenumber"), np.ones((3, 2)) * 0.25),
            "coherence": (("segment", "wavenumber"), np.ones((3, 2))),
        },
        coords={
            "segment_lon": ("segment", [359.0, 1.0, 30.0]),
            "segment_lat": ("segment", [0.0, 0.0, 0.0]),
            "wavenumber": ("wavenumber", [0.0, 0.1]),
            "wavelength": ("wavenumber", [np.inf, 10.0]),
        },
    )

    out = psd_score_by_region(
        ds_segments,
        lat_centers=[0.0],
        lon_centers=[0.0, 30.0],
        delta_lat=2.0,
        delta_lon=4.0,
        min_segments=2,
    )
    assert np.all(np.isfinite(out["psd_score"].sel(lat=0.0, lon=0.0)))
    assert np.all(np.isnan(out["psd_score"].sel(lat=0.0, lon=30.0)))


def test_psd_score_by_region_accepts_lon_centers_in_minus180_180() -> None:
    """Callers can pass ``lon_centers`` in either [-180, 180] or
    [0, 360] convention — they are normalized to [0, 360) so the
    circular-distance computation is correct either way."""
    ds_segments = xr.Dataset(
        {
            "psd_ref": (("segment", "wavenumber"), np.ones((2, 2))),
            "psd_pred": (("segment", "wavenumber"), np.ones((2, 2)) * 0.8),
            "psd_err": (("segment", "wavenumber"), np.ones((2, 2)) * 0.25),
        },
        coords={
            "segment_lon": ("segment", [358.0, 2.0]),
            "segment_lat": ("segment", [0.0, 0.0]),
            "wavenumber": ("wavenumber", [0.0, 0.1]),
        },
    )
    out = psd_score_by_region(
        ds_segments,
        lat_centers=[0.0],
        lon_centers=[-1.0],
        delta_lat=2.0,
        delta_lon=10.0,
        min_segments=2,
    )
    # Both segments fall within ±5° of the prime meridian once the
    # negative center is normalized to 359°.
    assert np.all(np.isfinite(out["psd_score"].sel(lat=0.0)))


def test_psd_score_by_region_omits_psd_score_when_inputs_missing() -> None:
    """``psd_score`` is only well-defined when both ``psd_ref`` and
    ``psd_err`` are present; otherwise the function must not raise."""
    ds_segments = xr.Dataset(
        {"psd_pred": (("segment", "wavenumber"), np.ones((2, 2)))},
        coords={
            "segment_lon": ("segment", [0.0, 0.0]),
            "segment_lat": ("segment", [0.0, 0.0]),
            "wavenumber": ("wavenumber", [0.0, 0.1]),
        },
    )
    out = psd_score_by_region(
        ds_segments,
        lat_centers=[0.0],
        lon_centers=[0.0],
        delta_lat=2.0,
        delta_lon=2.0,
        min_segments=1,
    )
    assert "psd_score" not in out.data_vars


def test_segmented_psd_score_operator_config_round_trips() -> None:
    op = SegmentedPSDScore(
        var_ref="ssh_ref",
        var_pred="ssh_pred",
        npt=64,
        overlap=0.25,
        spacing_km=7.0,
        window="boxcar",
    )
    cfg = op.get_config()
    assert SegmentedPSDScore(**cfg).get_config() == cfg
