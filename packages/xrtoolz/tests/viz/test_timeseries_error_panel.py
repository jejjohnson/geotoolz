"""TimeSeriesErrorPanel — multi-method scalar metrics over time (#118)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from xrtoolz.viz.validation import FacetPanel, TimeSeriesErrorPanel, method_palette


METHODS = ["duacs", "miost", "4dvarnet"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _scores() -> xr.Dataset:
    rng = np.random.default_rng(0)
    dims = ("method", "time")
    return xr.Dataset(
        {
            "rmse": (dims, rng.random((3, 20))),
            "nrmse": (dims, rng.random((3, 20))),
            "corr": (dims, rng.random((3, 20))),
        },
        coords={"method": METHODS, "time": np.arange(20)},
    )


def _labels(ax) -> list[str]:
    return [line.get_label() for line in ax.get_lines()]


def test_one_subplot_per_metric_one_line_per_method() -> None:
    fig = TimeSeriesErrorPanel()(_scores())
    assert len(fig.axes) == 3
    for ax, metric in zip(fig.axes, ["rmse", "nrmse", "corr"], strict=True):
        assert _labels(ax) == METHODS
        assert ax.get_ylabel() == metric
    assert fig.axes[-1].get_xlabel() == "time"


def test_single_legend_avoids_collisions() -> None:
    fig = TimeSeriesErrorPanel()(_scores())
    legends = [ax.get_legend() for ax in fig.axes]
    assert legends[0] is not None
    assert all(legend is None for legend in legends[1:])
    assert [t.get_text() for t in legends[0].get_texts()] == METHODS


def test_metrics_subset_is_respected() -> None:
    fig = TimeSeriesErrorPanel(metrics=["rmse"], ylabel="RMSE [m]")(_scores())
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "RMSE [m]"
    np.testing.assert_allclose(
        fig.axes[0].get_lines()[1].get_ydata(), _scores()["rmse"].values[1]
    )


def test_variables_are_methods_layout() -> None:
    ds = xr.Dataset(
        {
            m: ("time", np.random.default_rng(i).random(10))
            for i, m in enumerate(METHODS)
        },
        coords={"time": np.arange(10)},
    )
    fig = TimeSeriesErrorPanel(method_dim=None, metrics=["duacs", "miost"])(ds)
    assert len(fig.axes) == 1
    assert _labels(fig.axes[0]) == ["duacs", "miost"]


def test_colours_follow_method_palette() -> None:
    fig = TimeSeriesErrorPanel(metrics=["rmse"])(_scores())
    expected = method_palette(METHODS)
    for line in fig.axes[0].get_lines():
        assert mcolors.same_color(line.get_color(), expected[line.get_label()])


def test_explicit_and_named_palettes() -> None:
    fig = TimeSeriesErrorPanel(metrics=["rmse"], palette={"duacs": "black"})(_scores())
    colours = {line.get_label(): line.get_color() for line in fig.axes[0].get_lines()}
    assert mcolors.same_color(colours["duacs"], "black")
    fig = TimeSeriesErrorPanel(metrics=["rmse"], palette="viridis")(_scores())
    assert len({line.get_color() for line in fig.axes[0].get_lines()}) == 3


def test_dataarray_input() -> None:
    fig = TimeSeriesErrorPanel()(_scores()["rmse"])
    assert _labels(fig.axes[0]) == METHODS


def test_errors() -> None:
    with pytest.raises(ValueError, match="not in the input"):
        TimeSeriesErrorPanel(metrics=["bias"])(_scores())
    with pytest.raises(ValueError, match="method_dim=None"):
        TimeSeriesErrorPanel()(_scores().isel(method=0, drop=True))
    with pytest.raises(ValueError, match="1-D"):
        TimeSeriesErrorPanel()(_scores().expand_dims(region=["a", "b"]))


def test_composes_into_a_single_axes() -> None:
    ds = _scores().expand_dims(region=["north", "south"])
    fig = FacetPanel(TimeSeriesErrorPanel(metrics=["rmse"]), facet_dim="region")(ds)
    assert sum(_labels(ax) == METHODS for ax in fig.axes) == 2
    with pytest.raises(ValueError, match="one was given"):
        FacetPanel(TimeSeriesErrorPanel(), facet_dim="region")(ds)


def test_config_round_trip() -> None:
    panel = TimeSeriesErrorPanel(
        metrics=["rmse"], palette={"a": "red"}, method_dim=None
    )
    assert TimeSeriesErrorPanel(**panel.get_config()).get_config() == panel.get_config()
