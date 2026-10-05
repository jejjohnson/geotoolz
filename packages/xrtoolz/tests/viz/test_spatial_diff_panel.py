"""SpatialDiffPanel — (pred, ref, pred − ref) triptych (#117)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from xrtoolz.viz.validation import (
    CartesianMapPanel,
    SpatialDiffPanel,
    SpatialMapPanel,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _pair(nt: int = 2) -> tuple[xr.Dataset, xr.Dataset]:
    lat = np.linspace(30, 40, 11)
    lon = np.linspace(-70, -60, 13)
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(nt, 11, 13))
    pred = ref + 0.3 + 0.1 * rng.normal(size=ref.shape)
    coords = {"time": np.arange(nt), "lat": lat, "lon": lon}
    dims = ("time", "lat", "lon")
    return (
        xr.Dataset({"ssh": (dims, pred)}, coords=coords),
        xr.Dataset({"ssh": (dims, ref)}, coords=coords),
    )


def _meshes(fig):
    return [ax.collections[0] for ax in fig.axes[:3]]


def test_three_cells_with_titles() -> None:
    fig = SpatialDiffPanel(variable="ssh")(*_pair())
    assert [ax.get_title() for ax in fig.axes[:3]] == [
        "prediction",
        "reference",
        "prediction - reference",
    ]


def test_pred_and_ref_share_limits_and_diff_is_symmetric() -> None:
    pred_m, ref_m, diff_m = _meshes(SpatialDiffPanel(variable="ssh")(*_pair()))
    assert pred_m.get_clim() == ref_m.get_clim()
    lo, hi = diff_m.get_clim()
    assert lo == pytest.approx(-hi)
    assert diff_m.get_cmap().name == "RdBu_r"


def test_diff_cell_holds_pred_minus_ref_at_time_index() -> None:
    pred, ref = _pair()
    fig = SpatialDiffPanel(variable="ssh", time_index=1)(pred, ref)
    values = np.ma.getdata(_meshes(fig)[2].get_array()).reshape(11, 13)
    np.testing.assert_allclose(values, (pred - ref)["ssh"].values[1])


def test_explicit_limits_and_asymmetric_diff() -> None:
    pred_m, _, diff_m = _meshes(
        SpatialDiffPanel(variable="ssh", vmin=-2.0, vmax=2.0, diff_symmetric=False)(
            *_pair()
        )
    )
    assert pred_m.get_clim() == (-2.0, 2.0)
    lo, hi = diff_m.get_clim()
    assert lo != pytest.approx(-hi)


def test_inner_panel_attributes_are_restored() -> None:
    inner = SpatialMapPanel(variable="ssh", cmap="viridis")
    SpatialDiffPanel(inner, field_cmap="magma")(*_pair())
    assert (inner.cmap, inner.vmin, inner.vmax) == ("viridis", None, None)


def test_cartesian_inner_panel() -> None:
    x = np.linspace(0.0, 5e5, 16)
    ref = xr.DataArray(
        np.random.default_rng(1).normal(size=(16, 16)),
        dims=("y", "x"),
        coords={"x": x, "y": x},
    )
    fig = SpatialDiffPanel(CartesianMapPanel())(ref * 1.1, ref)
    assert [ax.get_xlabel() for ax in fig.axes[:3]] == ["x [km]"] * 3
    pred_m, ref_m, diff_m = _meshes(fig)
    assert pred_m.get_clim() == ref_m.get_clim()
    lo, hi = diff_m.get_clim()
    assert lo == pytest.approx(-hi)


def test_mismatched_grids_raise() -> None:
    pred, ref = _pair()
    with pytest.raises(ValueError, match="same grid"):
        SpatialDiffPanel(variable="ssh")(pred, ref.isel(lat=slice(1, None)))


def test_rejects_multi_input_inner_panel() -> None:
    with pytest.raises(TypeError, match="cannot be wrapped"):
        SpatialDiffPanel(SpatialDiffPanel())


def test_config_describes_inner_panel() -> None:
    config = SpatialDiffPanel(variable="ssh", names=("a", "b")).get_config()
    assert config["panel"] == "SpatialMapPanel"
    assert config["names"] == ["a", "b"]
    assert config["diff_cmap"] == "RdBu_r"


def test_projected_inner_panel_builds_geoaxes() -> None:
    cgeo = pytest.importorskip("cartopy.mpl.geoaxes")
    inner = SpatialMapPanel(variable="ssh", projection="PlateCarree", gridlines=False)
    fig = SpatialDiffPanel(inner)(*_pair())
    assert all(isinstance(ax, cgeo.GeoAxes) for ax in fig.axes[:3])
