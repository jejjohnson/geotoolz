"""HovmollerPanel — time × spatial-axis section (#120)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from xrtoolz.viz.validation import FacetPanel, HovmollerPanel


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _field(positive: bool = False) -> xr.DataArray:
    t = np.arange(12.0)
    lat = np.linspace(-30, 30, 7)
    lon = np.linspace(0, 350, 10)
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(12, 7, 10))
    if positive:
        vals = np.abs(vals) + 0.1
    return xr.DataArray(
        vals,
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": lat, "lon": lon},
        name="sst",
        attrs={"units": "K"},
    )


def _mesh(fig):
    return fig.axes[0].collections[0]


def test_axis_labels_include_time_and_keep_dim() -> None:
    fig = HovmollerPanel()(_field())
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("time", "lat")
    fig = HovmollerPanel(keep_dim="lon", time_axis="y")(_field())
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("lon", "time")


def test_section_is_mean_over_the_other_dims() -> None:
    da = _field()
    da[0, 0, 0] = np.nan  # skipped, not propagated
    fig = HovmollerPanel(keep_dim="lat")(da)
    values = np.ma.getdata(_mesh(fig).get_array()).reshape(7, 12)
    np.testing.assert_allclose(values, da.mean("lon", skipna=True).values.T)


def test_dataset_input_with_extra_dims_and_colorbar_units() -> None:
    ds = _field().expand_dims(depth=[0.0, 10.0]).to_dataset()
    fig = HovmollerPanel(variable="sst", keep_dim="lon")(ds)
    assert _mesh(fig).get_array().size == 12 * 10
    assert fig.axes[1].get_ylabel() == "K"
    assert fig.axes[0].get_title() == "" and fig.texts[0].get_text() == "sst"


def test_log_norm_masks_non_positive_values() -> None:
    fig = HovmollerPanel(norm="log")(_field(positive=True))
    assert isinstance(_mesh(fig).norm, mcolors.LogNorm)
    da = _field(positive=True)
    da[:, 3, :] = -1.0
    arr = _mesh(HovmollerPanel(norm="log")(da)).get_array()
    assert np.ma.getmaskarray(arr).reshape(7, 12)[3].all()


def test_linear_limits_and_cmap() -> None:
    mesh = _mesh(HovmollerPanel(cmap="viridis", vmin=-1, vmax=1)(_field()))
    assert mesh.get_clim() == (-1.0, 1.0)
    assert mesh.get_cmap().name == "viridis"


def test_cartesian_keep_dim() -> None:
    x = np.linspace(0, 1e6, 8)
    da = xr.DataArray(
        np.zeros((5, 6, 8)),
        dims=("time", "y", "x"),
        coords={"time": np.arange(5), "y": np.arange(6.0), "x": x},
    )
    fig = HovmollerPanel(keep_dim="x")(da)
    assert fig.axes[0].get_ylabel() == "x"


def test_errors() -> None:
    with pytest.raises(ValueError, match="norm"):
        HovmollerPanel(norm="symlog")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="time_axis"):
        HovmollerPanel(time_axis="z")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="missing dim"):
        HovmollerPanel(keep_dim="depth")(_field())


def test_composes_with_facet_panel() -> None:
    ds = xr.concat([_field(), _field() + 1], dim="experiment").assign_coords(
        experiment=["a", "b"]
    )
    fig = FacetPanel(HovmollerPanel(), facet_dim="experiment")(ds)
    assert sum(ax.get_ylabel() == "lat" for ax in fig.axes) == 2


def test_config_round_trip() -> None:
    panel = HovmollerPanel(variable="sst", keep_dim="lon", norm="log", vmin=0.1)
    assert HovmollerPanel(**panel.get_config()).get_config() == panel.get_config()
