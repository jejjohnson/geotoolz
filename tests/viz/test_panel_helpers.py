"""Tests for viz styling helpers and score-panel clipping.

Covers ``method_palette``, ``shared_norm``, and the ``clip``/``ylim``
behaviour of the PSD score panels.
"""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xrtoolz.viz import shared_norm
from xrtoolz.viz.validation import (
    PSDIsotropicScorePanel,
    PSDSpaceTimeScorePanel,
    method_palette,
)


# ---------- method_palette ------------------------------------------------


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


# ---------- shared_norm ---------------------------------------------------


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


# ---------- clip toggle on score panels -----------------------------------


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
