"""Tests for thin composite metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.metrics import (
    nrmse,
    psd_score_spacetime,
    resolved_scale_2d,
    rmse_skill_scores,
)


def test_nrmse_and_rmse_skill_scores_match_manual_formula() -> None:
    ref = xr.Dataset(
        {
            "ssh": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[2.0, 3.0], [4.0, 5.0]],
                    ]
                ),
            )
        }
    )
    pred = xr.Dataset(
        {
            "ssh": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[2.0, 3.0], [4.0, 5.0]],
                        [[2.0, 3.0], [4.0, 7.0]],
                    ]
                ),
            )
        }
    )

    expected_rmse_t = np.array(
        [
            1.0
            - np.sqrt(np.mean(np.ones((2, 2)) ** 2))
            / np.sqrt(np.mean(ref["ssh"][0].values ** 2)),
            1.0
            - np.sqrt(np.mean(np.array([[0.0, 0.0], [0.0, 2.0]]) ** 2))
            / np.sqrt(np.mean(ref["ssh"][1].values ** 2)),
        ]
    )
    expected_rmse_xy = np.sqrt(
        np.mean(
            np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 2.0]],
                ]
            )
            ** 2,
            axis=0,
        )
    )
    expected_leaderboard = 1.0 - 1.0 / np.sqrt(np.mean(ref["ssh"].values ** 2))

    np.testing.assert_allclose(
        nrmse(pred["ssh"], ref["ssh"], dim=("lat", "lon")).values,
        expected_rmse_t,
    )

    scores = rmse_skill_scores(pred["ssh"], ref["ssh"])
    np.testing.assert_allclose(scores["rmse_t"].values, expected_rmse_t)
    np.testing.assert_allclose(scores["rmse_xy"].values, expected_rmse_xy)
    assert float(scores["leaderboard_rmse"]) == pytest.approx(expected_leaderboard)
    assert float(scores["error_stability"]) == pytest.approx(
        np.std(expected_rmse_t, ddof=0)
    )


def test_rmse_skill_scores_identical_fields_have_perfect_skill() -> None:
    ref = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), np.arange(12.0).reshape(3, 2, 2))}
    )
    scores = rmse_skill_scores(ref["ssh"], ref["ssh"])

    np.testing.assert_allclose(scores["rmse_t"].values, np.ones(3))
    np.testing.assert_allclose(scores["rmse_xy"].values, np.zeros((2, 2)))
    assert float(scores["leaderboard_rmse"]) == pytest.approx(1.0)
    assert float(scores["error_stability"]) == pytest.approx(0.0)


def test_resolved_scale_2d_returns_expected_wavelength_bounds() -> None:
    score = xr.DataArray(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        dims=("freq_time", "freq_lon"),
        coords={
            "freq_time": [10.0, 20.0, 40.0, 80.0],
            "freq_lon": [1.0, 2.0, 4.0, 8.0],
        },
    )

    summary = resolved_scale_2d(score, level=0.5)

    assert summary == pytest.approx(
        {
            "lambda_space_min": 1.0 / 6.0,
            "lambda_time_min": 1.0 / 60.0,
            "lambda_space_max": 1.0 / 1.5,
            "lambda_time_max": 1.0 / 15.0,
        }
    )


def test_resolved_scale_2d_handles_disconnected_segments_and_no_contour() -> None:
    disconnected = xr.DataArray(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        dims=("freq_time", "freq_lon"),
        coords={
            "freq_time": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
            "freq_lon": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        },
    )

    summary = resolved_scale_2d(disconnected, level=0.5)
    assert summary["lambda_space_max"] == pytest.approx(1.0 / 1.5)
    assert summary["lambda_space_min"] == pytest.approx(1.0 / 48.0)
    assert summary["lambda_time_max"] == pytest.approx(1.0 / 15.0)
    assert summary["lambda_time_min"] == pytest.approx(1.0 / 480.0)

    no_contour = resolved_scale_2d(
        xr.DataArray(
            np.ones((3, 3)),
            dims=("freq_time", "freq_lon"),
            coords={"freq_time": [1.0, 2.0, 3.0], "freq_lon": [1.0, 2.0, 3.0]},
        ),
        level=0.5,
    )
    assert all(np.isnan(value) for value in no_contour.values())


def test_psd_score_spacetime_returns_positive_frequency_score_and_summary() -> None:
    rng = np.random.default_rng(0)
    time = np.arange(64, dtype=float)
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.arange(48, dtype=float)
    ref_field = rng.standard_normal((time.size, lat.size, lon.size))
    smoothed = (
        ref_field
        + np.roll(ref_field, 1, axis=0)
        + np.roll(ref_field, -1, axis=0)
        + np.roll(ref_field, 1, axis=2)
        + np.roll(ref_field, -1, axis=2)
    ) / 5.0
    pred_field = 0.5 * ref_field + 0.5 * smoothed

    ref = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), ref_field)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    pred = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), pred_field)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    score, summary = psd_score_spacetime(pred["ssh"], ref["ssh"])

    assert score["score"].dims == ("freq_lon", "freq_time")
    assert np.all(score["freq_lon"].values > 0.0)
    assert np.all(score["freq_time"].values > 0.0)
    assert 0.0 < float(score["score"].min()) < 1.0
    assert 0.0 < float(score["score"].max()) < 1.0
    assert set(summary) == {
        "lambda_space_min",
        "lambda_time_min",
        "lambda_space_max",
        "lambda_time_max",
    }
    assert all(np.isfinite(value) for value in summary.values())


def test_rmse_skill_scores_rejects_time_dim_in_space_dims() -> None:
    ds = xr.Dataset(
        {"ssh": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
        coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
    )
    with pytest.raises(ValueError, match="time_dim"):
        rmse_skill_scores(ds["ssh"], ds["ssh"], space_dims=("time", "lat"))


def test_psd_score_spacetime_rejects_isotropic_kwarg() -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"ssh": (("time", "lon"), rng.standard_normal((16, 16)))},
        coords={"time": np.arange(16.0), "lon": np.arange(16.0)},
    )
    with pytest.raises(ValueError, match="isotropic"):
        psd_score_spacetime(ds["ssh"], ds["ssh"], isotropic=True)
