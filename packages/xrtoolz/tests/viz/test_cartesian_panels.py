"""CartesianMapPanel + EnergyTimeSeriesPanel (#325)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from xrtoolz.viz.validation import (
    AnimatePanel,
    CartesianMapPanel,
    EnergyTimeSeriesPanel,
    FacetPanel,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _model_run(nt: int = 3, n: int = 16) -> xr.Dataset:
    x = np.linspace(0.0, 1e6, n)
    t = np.arange(nt)
    rng = np.random.default_rng(0)
    zeta = rng.normal(size=(nt, n, n)) + 3.0  # off-centre so symmetry is visible
    mask = np.ones((n, n), dtype=bool)
    mask[:, 0] = False
    ds = xr.Dataset(
        {
            "zeta": (("time", "y", "x"), zeta, {"long_name": "ζ", "units": "s-1"}),
            "psi": (("time", "y", "x"), rng.normal(size=(nt, n, n))),
            "mask": (("y", "x"), mask),
        },
        coords={"time": t, "x": x, "y": x},
    )
    return ds


def _mesh(fig):
    return fig.axes[0].collections[0]


class TestCartesianMapPanel:
    def test_axes_in_km_with_equal_aspect(self) -> None:
        fig = CartesianMapPanel(variable="zeta")(_model_run())
        ax = fig.axes[0]
        assert ax.get_xlabel() == "x [km]"
        assert ax.get_ylabel() == "y [km]"
        assert ax.get_xlim()[1] == pytest.approx(1000.0, rel=0.05)
        assert ax.get_aspect() == 1.0

    def test_metres_option(self) -> None:
        fig = CartesianMapPanel(variable="zeta", units="m")(_model_run())
        assert fig.axes[0].get_xlabel() == "x [m]"
        assert fig.axes[0].get_xlim()[1] == pytest.approx(1e6, rel=0.05)

    def test_default_norm_is_symmetric_about_zero(self) -> None:
        mesh = _mesh(CartesianMapPanel(variable="zeta")(_model_run()))
        vmin, vmax = mesh.get_clim()
        assert vmin == pytest.approx(-vmax)
        assert vmax > 3.0

    def test_q99_norm_is_not_centred(self) -> None:
        mesh = _mesh(CartesianMapPanel(variable="zeta", norm="q99")(_model_run()))
        vmin, vmax = mesh.get_clim()
        assert vmin > 0.0
        assert vmax > vmin

    def test_explicit_limits(self) -> None:
        mesh = _mesh(CartesianMapPanel(variable="zeta", norm=(-1, 2))(_model_run()))
        assert mesh.get_clim() == (-1.0, 2.0)

    def test_mask_blanks_invalid_cells(self) -> None:
        mesh = _mesh(CartesianMapPanel(variable="zeta")(_model_run()))
        values = np.ma.getdata(mesh.get_array()).reshape(16, 16)
        assert np.isnan(values[:, 0]).all()
        assert np.isfinite(values[:, 1:]).all()

    def test_selects_time_index_and_labels_colorbar(self) -> None:
        ds = _model_run()
        fig = CartesianMapPanel(variable="zeta", time_index=2, norm=None)(ds)
        values = np.ma.getdata(_mesh(fig).get_array()).reshape(16, 16)
        np.testing.assert_allclose(values[:, 1:], ds["zeta"].values[2][:, 1:])
        assert fig.axes[1].get_ylabel() == "ζ [s-1]"

    def test_contour_overlay(self) -> None:
        fig = CartesianMapPanel(variable="zeta", contour="psi", contour_levels=5)(
            _model_run()
        )
        assert len(fig.axes[0].collections) > 1

    def test_rejects_non_2d_field(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            CartesianMapPanel(variable="zeta", time_index=None)(_model_run())

    def test_rejects_bad_options(self) -> None:
        with pytest.raises(ValueError, match="units"):
            CartesianMapPanel(units="miles")  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="norm"):
            CartesianMapPanel(norm="q95")  # ty: ignore[invalid-argument-type]

    def test_config_round_trip(self) -> None:
        panel = CartesianMapPanel(variable="zeta", norm=(-1.0, 1.0), units="m")
        config = panel.get_config()
        rebuilt = CartesianMapPanel(**config)
        assert rebuilt.get_config() == config
        assert rebuilt.norm == (-1.0, 1.0)

    def test_composes_with_animate_and_facet(self) -> None:
        ds = _model_run()
        ani = AnimatePanel(CartesianMapPanel(variable="zeta"), frame_dim="time")(ds)
        assert isinstance(ani, FuncAnimation)
        for index in range(ds.sizes["time"]):
            ani._func(index)

        fig = FacetPanel(
            CartesianMapPanel(variable="zeta", time_index=None), facet_dim="time"
        )(ds)
        assert sum(ax.get_xlabel() == "x [km]" for ax in fig.axes) == 3


class TestEnergyTimeSeriesPanel:
    def _invariants(self) -> xr.Dataset:
        t = np.linspace(0.0, 10.0, 11)
        return xr.Dataset(
            {
                "kinetic_energy": ("time", 4.0 * np.exp(-0.01 * t), {"units": "J"}),
                "enstrophy": ("time", 2.0 + 0.1 * t),
            },
            coords={"time": t},
        )

    def test_one_line_per_variable(self) -> None:
        fig = EnergyTimeSeriesPanel(["kinetic_energy", "enstrophy"])(self._invariants())
        ax = fig.axes[0]
        assert [line.get_label() for line in ax.get_lines()] == [
            "kinetic_energy",
            "enstrophy",
        ]
        assert ax.get_xlabel() == "time"

    def test_relative_normalises_by_first_sample(self) -> None:
        ds = self._invariants()
        fig = EnergyTimeSeriesPanel(["kinetic_energy", "enstrophy"], relative=True)(ds)
        lines = fig.axes[0].get_lines()
        for line, name in zip(lines, ["kinetic_energy", "enstrophy"], strict=False):
            expected = ds[name].values / ds[name].values[0]
            np.testing.assert_allclose(line.get_ydata(), expected)
            assert line.get_ydata()[0] == 1.0

    def test_log_scale_and_single_variable_units(self) -> None:
        fig = EnergyTimeSeriesPanel("kinetic_energy", log=True)(self._invariants())
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        assert ax.get_ylabel() == "J"

    def test_relative_rejects_zero_start(self) -> None:
        ds = self._invariants().assign(mass=("time", np.arange(11.0)))
        with pytest.raises(ValueError, match="starts at"):
            EnergyTimeSeriesPanel(["mass"], relative=True)(ds)

    def test_rejects_missing_or_non_1d(self) -> None:
        with pytest.raises(ValueError, match="not in the input"):
            EnergyTimeSeriesPanel(["nope"])(self._invariants())
        with pytest.raises(ValueError, match="1-D"):
            EnergyTimeSeriesPanel(["zeta"])(_model_run())

    def test_config_round_trip(self) -> None:
        panel = EnergyTimeSeriesPanel(["a", "b"], relative=True, time="step")
        assert EnergyTimeSeriesPanel(**panel.get_config()).get_config() == (
            panel.get_config()
        )
