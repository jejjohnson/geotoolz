"""Tests for the spatial-panel bundle (#116/#128 + Variable.cmap)."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xr_toolz.types._src.variable import (
    ADT,
    CHL,
    ICE_CONC,
    MSL,
    REGISTRY,
    SO,
    SSH,
    SST,
    T2M,
    TOTAL_CLOUD_COVER,
    TP,
    UO,
    VISIBILITY,
    WAVE_FROM_DIRECTION,
    WIND_SPEED,
    Variable,
)
from xr_toolz.viz import PRESETS, cmap_for, make_axes
from xr_toolz.viz.validation import SpatialMapPanel


# ---------- Variable.cmap registry -----------------------------------------


@pytest.mark.parametrize(
    "var, expected",
    [
        (SSH, "RdBu_r"),
        (ADT, "RdBu_r"),
        (UO, "RdBu_r"),
        (SST, "RdYlBu_r"),
        (T2M, "RdYlBu_r"),
        (SO, "viridis"),
        (CHL, "viridis"),
        (ICE_CONC, "Blues"),
        (TP, "Blues"),
        (MSL, "viridis"),
        (TOTAL_CLOUD_COVER, "gray_r"),
        (WIND_SPEED, "magma"),
        (WAVE_FROM_DIRECTION, "twilight"),
        (VISIBILITY, "cividis"),
    ],
)
def test_variable_registry_has_cmap(var, expected):
    assert var.cmap == expected


def test_all_registry_entries_have_cmap():
    missing = [v.name for v in REGISTRY.values() if v.cmap is None]
    assert missing == []


def test_user_constructed_variable_cmap_default_none():
    v = Variable(name="foo")
    assert v.cmap is None


# ---------- cmap_for ------------------------------------------------------


def test_cmap_for_known_name():
    assert cmap_for("ssh") == "RdBu_r"
    assert cmap_for("sst") == "RdYlBu_r"


def test_cmap_for_case_insensitive():
    assert cmap_for("SST") == "RdYlBu_r"


def test_cmap_for_variable_instance():
    assert cmap_for(SSH) == "RdBu_r"


def test_cmap_for_unknown_returns_default():
    assert cmap_for("not_a_var") == "viridis"
    assert cmap_for("not_a_var", default="plasma") == "plasma"


def test_cmap_for_none_returns_default():
    assert cmap_for(None) == "viridis"


def test_cmap_for_variable_without_cmap_falls_back():
    v = Variable(name="custom")
    assert cmap_for(v, default="cividis") == "cividis"


# ---------- PRESETS / make_axes -------------------------------------------


def test_presets_keys_match_acceptance():
    assert set(PRESETS) == {
        "global",
        "north_atlantic",
        "gulf_stream",
        "kuroshio",
        "mediterranean",
    }


def test_make_axes_no_projection_returns_plain_axes():
    fig, ax = make_axes(projection=None)
    assert not hasattr(ax, "set_extent")
    plt.close(fig)


def test_make_axes_preset_returns_geoaxes_with_extent():
    fig, ax = make_axes(projection="gulf_stream")
    assert hasattr(ax, "set_extent")
    # PlateCarree presets keep extent in lon/lat units.
    extent = ax.get_extent(crs=ccrs.PlateCarree())
    assert pytest.approx(extent, abs=1.0) == (-80, -50, 30, 45)
    plt.close(fig)


def test_make_axes_unknown_projection_raises():
    with pytest.raises(ValueError, match="Unknown projection"):
        make_axes(projection="atlantis")


# ---------- SpatialMapPanel ----------------------------------------------


def _ssh_snapshot(with_time: bool = True) -> xr.DataArray:
    rng = np.random.default_rng(0)
    lon = np.linspace(-80.0, -50.0, 30)
    lat = np.linspace(30.0, 45.0, 20)
    if with_time:
        time = np.arange(3)
        data = rng.standard_normal((3, 20, 30))
        return xr.DataArray(
            data,
            coords={"time": time, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            name="ssh",
        )
    data = rng.standard_normal((20, 30))
    return xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="ssh"
    )


def test_panel_plain_axes_returns_figure():
    fig = SpatialMapPanel(var="ssh")(_ssh_snapshot())
    assert fig is not None
    plt.close(fig)


def test_panel_auto_cmap_from_registry():
    panel = SpatialMapPanel(var="ssh")
    fig = panel(_ssh_snapshot())
    # The pcolormesh is the first QuadMesh on the axes.
    qm = next(c for c in fig.axes[0].collections)
    assert qm.cmap.name == "RdBu_r"
    plt.close(fig)


def test_panel_explicit_cmap_overrides_registry():
    fig = SpatialMapPanel(var="ssh", cmap="plasma")(_ssh_snapshot())
    qm = next(c for c in fig.axes[0].collections)
    assert qm.cmap.name == "plasma"
    plt.close(fig)


def test_panel_handles_no_time_dim():
    fig = SpatialMapPanel(var="ssh")(_ssh_snapshot(with_time=False))
    assert fig is not None
    plt.close(fig)


def test_panel_dataset_input_auto_picks_var():
    da = _ssh_snapshot()
    fig = SpatialMapPanel()(da.to_dataset())  # var=None auto-picks
    assert fig is not None
    plt.close(fig)


def test_panel_gulf_stream_preset_applies_extent():
    fig = SpatialMapPanel(var="ssh", projection="gulf_stream")(_ssh_snapshot())
    ax = fig.axes[0]
    assert hasattr(ax, "set_extent")
    extent = ax.get_extent(crs=ccrs.PlateCarree())
    assert pytest.approx(extent, abs=1.0) == (-80, -50, 30, 45)
    plt.close(fig)


def test_panel_savefig_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "snapshot.png"
    fig = SpatialMapPanel(var="ssh", savefig=out)(_ssh_snapshot())
    assert out.exists()
    plt.close(fig)


def test_panel_get_config_round_trip():
    panel = SpatialMapPanel(var="sst", projection="north_atlantic", cbar_label="K")
    cfg = panel.get_config()
    assert cfg["var"] == "sst"
    assert cfg["projection"] == "north_atlantic"
    assert cfg["cbar_label"] == "K"
