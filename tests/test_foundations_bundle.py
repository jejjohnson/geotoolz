"""Tests for the Foundations bundle (#119/#123/#124/#126/#127)."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xr_toolz.metrics import find_intercept_2D, rank_methods
from xr_toolz.viz import shared_norm
from xr_toolz.viz.validation import (
    PSDIsotropicScorePanel,
    PSDSpaceTimeScorePanel,
    method_palette,
)


# ---------- #119 method_palette ------------------------------------------


def test_method_palette_deterministic_order():
    p1 = method_palette(["DUACS", "MIOST", "BFNQG", "4DVarNet"])
    p2 = method_palette(["4DVarNet", "BFNQG", "DUACS", "MIOST"])
    assert p1 == p2  # input order doesn't change output


def test_method_palette_dedupes():
    p = method_palette(["a", "a", "b"])
    assert set(p.keys()) == {"a", "b"}


def test_method_palette_cycles_when_more_names_than_colours():
    p = method_palette([f"m{i}" for i in range(12)], cycle=("red", "blue"))
    assert set(p.values()) == {"red", "blue"}


def test_method_palette_empty_cycle_raises():
    with pytest.raises(ValueError, match="cycle"):
        method_palette(["a"], cycle=())


# ---------- #123 shared_norm ---------------------------------------------


def test_shared_norm_quantile_default():
    a = xr.DataArray(np.linspace(0, 100, 101))
    b = xr.DataArray(np.linspace(0, 200, 101))
    vmin, vmax = shared_norm(a, b)
    assert vmin == pytest.approx(2.02, abs=0.1)
    assert vmax == pytest.approx(191.96, abs=0.1)


def test_shared_norm_full_range_when_q_none():
    a = xr.DataArray(np.array([-3.0, 0.0, 5.0]))
    assert shared_norm(a, q=None) == (-3.0, 5.0)


def test_shared_norm_symmetric_returns_balanced():
    a = xr.DataArray(np.array([-2.0, 0.5, 3.0]))
    vmin, vmax = shared_norm(a, q=None, symmetric=True)
    assert vmin == -3.0
    assert vmax == 3.0


def test_shared_norm_dataset_single_var():
    ds = xr.Dataset({"x": ("t", np.array([0.0, 1.0, 2.0]))})
    assert shared_norm(ds, q=None) == (0.0, 2.0)


def test_shared_norm_multi_var_dataset_raises():
    ds = xr.Dataset({"x": ("t", [0, 1]), "y": ("t", [2, 3])})
    with pytest.raises(ValueError, match="exactly one data variable"):
        shared_norm(ds)


def test_shared_norm_ignores_nan():
    a = xr.DataArray(np.array([np.nan, 0.0, 10.0]))
    assert shared_norm(a, q=None) == (0.0, 10.0)


def test_shared_norm_no_inputs_raises():
    with pytest.raises(ValueError, match="at least one"):
        shared_norm()


def test_shared_norm_accepts_dask_backed_arrays():
    """Dask-backed inputs should work (laziness preserved up to the
    final scalar quantile)."""
    da_module = pytest.importorskip("dask.array")
    a = xr.DataArray(da_module.from_array(np.linspace(0.0, 1.0, 100), chunks=25))
    b = xr.DataArray(da_module.from_array(np.linspace(0.0, 2.0, 100), chunks=25))
    vmin, vmax = shared_norm(a, b, q=None)
    assert vmin == pytest.approx(0.0)
    assert vmax == pytest.approx(2.0)


# ---------- #124 clip toggle on score panels -----------------------------


def _signed_iso_score() -> xr.DataArray:
    f = np.linspace(0.001, 0.5, 50)
    score = 1.0 - 6.0 * f  # crosses 0 around f=1/6
    return xr.DataArray(score, coords={"freq_r": f}, dims=("freq_r",), name="score")


def test_iso_score_panel_clip_default_pins_ylim():
    fig = PSDIsotropicScorePanel()(_signed_iso_score())
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.get_ylim(), (0.0, 1.0))
    plt.close(fig)


def test_iso_score_panel_unclipped_shows_negative_values():
    fig = PSDIsotropicScorePanel(clip=False)(_signed_iso_score())
    ax = fig.axes[0]
    ymin, _ = ax.get_ylim()
    assert ymin < 0.0
    plt.close(fig)


def test_iso_score_panel_explicit_ylim_overrides():
    fig = PSDIsotropicScorePanel(clip=False, ylim=(-2.0, 1.5))(_signed_iso_score())
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.get_ylim(), (-2.0, 1.5))
    plt.close(fig)


def test_iso_score_panel_ylim_wrong_length_raises():
    with pytest.raises(ValueError, match="2-tuple"):
        PSDIsotropicScorePanel(ylim=(0.0, 0.5, 1.0))


def test_iso_score_panel_ylim_inverted_raises():
    with pytest.raises(ValueError, match="ymin <= ymax"):
        PSDIsotropicScorePanel(ylim=(1.0, 0.0))


def test_space_time_score_panel_clip_toggle_returns_figure():
    fl = np.linspace(0.001, 0.5, 16)
    ft = np.linspace(0.001, 0.4, 12)
    FL, FT = np.meshgrid(fl, ft)
    score = 1.0 - 5.0 * (FL + FT)  # spans signed range
    da = xr.DataArray(
        score,
        coords={"freq_time": ft, "freq_lon": fl},
        dims=("freq_time", "freq_lon"),
        name="score",
    )
    fig_clipped = PSDSpaceTimeScorePanel()(da)
    fig_signed = PSDSpaceTimeScorePanel(clip=False)(da)
    assert fig_clipped is not None and fig_signed is not None
    plt.close(fig_clipped)
    plt.close(fig_signed)


# ---------- #126 rank_methods --------------------------------------------


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


# ---------- #127 find_intercept_2D ---------------------------------------


def test_find_intercept_2D_recovers_analytic_boundary():
    n = 64
    fl = np.linspace(0.0, 1.0, n)
    ft = np.linspace(0.0, 1.0, n)
    FL, FT = np.meshgrid(fl, ft)
    # Linear monotone score: 1 - (fl + ft); score=0.5 along fl+ft=0.5
    score = 1.0 - (FL + FT)
    da = xr.DataArray(
        score,
        coords={"freq_time": ft, "freq_lon": fl},
        dims=("freq_time", "freq_lon"),
    )
    segments = find_intercept_2D(da, level=0.5)
    assert len(segments) >= 1
    seg = segments[0]
    assert "axis" in seg.dims
    sx = seg.sel(axis="freq_lon").values
    ty = seg.sel(axis="freq_time").values
    # Analytic boundary: sx + ty = 0.5
    np.testing.assert_allclose(sx + ty, 0.5, atol=2.0 / n)


def test_find_intercept_2D_rejects_non_2d():
    da = xr.DataArray([1.0, 2.0, 3.0], dims=("freq_lon",))
    with pytest.raises(ValueError, match="2-D"):
        find_intercept_2D(da)


def test_find_intercept_2D_missing_dim_raises():
    da = xr.DataArray(np.zeros((4, 4)), dims=("a", "b"))
    with pytest.raises(ValueError, match="must have dims"):
        find_intercept_2D(da)
