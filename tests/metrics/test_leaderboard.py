"""Tests for :func:`xrtoolz.metrics.rank_methods` (leaderboard ranking)."""

from __future__ import annotations

import pytest
import xarray as xr

from xrtoolz.metrics import rank_methods


def _scores_ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "rmse": ("method", [0.20, 0.10, 0.30]),
            "nrmse": ("method", [0.40, 0.30, 0.50]),
            "corr": ("method", [0.85, 0.95, 0.70]),
        },
        coords={"method": ["B", "A", "C"]},
    )


def test_rank_methods_sorts_ascending_by_rmse():
    df = rank_methods(_scores_ds(), by="rmse")
    assert list(df.index) == ["A", "B", "C"]
    assert list(df["rmse"]) == [0.10, 0.20, 0.30]


def test_rank_methods_descending_for_higher_is_better():
    df = rank_methods(_scores_ds(), by="corr", ascending=False)
    assert list(df.index) == ["A", "B", "C"]


def test_rank_methods_include_subsets_columns():
    df = rank_methods(_scores_ds(), by="rmse", include=("rmse", "corr"))
    assert list(df.columns) == ["rmse", "corr"]


def test_rank_methods_unknown_metric_raises():
    with pytest.raises(ValueError, match="not in data_vars"):
        rank_methods(_scores_ds(), by="bogus")


def test_rank_methods_rejects_forgotten_region_dim():
    """If the dataset has a (region, method) layout but the caller forgets
    ``region_dim``, we must not silently mix rows across regions."""
    ds = xr.Dataset(
        {
            "rmse": (
                ("region", "method"),
                [[0.1, 0.3, 0.2], [0.4, 0.2, 0.3]],
            ),
        },
        coords={"region": ["NA", "GS"], "method": ["A", "B", "C"]},
    )
    with pytest.raises(ValueError, match="multi-valued extra index"):
        rank_methods(ds, by="rmse")  # forgot region_dim


def test_rank_methods_squeezes_singleton_extra_dim():
    """Singleton extras (e.g. one region) are still allowed without
    region_dim — they're silently squeezed out."""
    ds = xr.Dataset(
        {"rmse": (("region", "method"), [[0.1, 0.3, 0.2]])},
        coords={"region": ["NA"], "method": ["A", "B", "C"]},
    )
    df = rank_methods(ds, by="rmse")
    assert list(df.index) == ["A", "C", "B"]


def test_rank_methods_per_region():
    ds = xr.Dataset(
        {
            "rmse": (
                ("region", "method"),
                [[0.1, 0.3, 0.2], [0.4, 0.2, 0.3]],
            ),
        },
        coords={"region": ["NA", "GS"], "method": ["A", "B", "C"]},
    )
    df = rank_methods(ds, by="rmse", region_dim="region")
    # Per-region sort by ascending rmse.
    assert list(df.loc["NA"].index) == ["A", "C", "B"]
    assert list(df.loc["GS"].index) == ["B", "C", "A"]
