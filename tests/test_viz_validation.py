"""V6 validation-panel tests — V6.1 base + V6.2/3/4 panels."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")  # headless — must run before pyplot import

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Graph, Input, Sequential
from xr_toolz.viz.validation import (
    EulerianLagrangianPanel,
    EventVerificationPanel,
    LeadTimeSkillPanel,
    ProcessBudgetPanel,
    ScaleSkillPanel,
    SpectralSkillPanel,
)
from xr_toolz.viz.validation._src.base import _ValidationPanel


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- V6.1 base ----------------------------------------------------------


def test_validation_panel_returns_figure():
    class _Demo(_ValidationPanel):
        def _build(self, fig, axes, x):
            axes.plot(np.arange(len(x)), x)

    fig = _Demo()(np.array([1.0, 2.0, 3.0]))
    assert isinstance(fig, mpl_figure.Figure)


def test_validation_panel_get_config_round_trips_kwargs():
    class _Demo(_ValidationPanel):
        def _build(self, fig, axes, x):
            return None

    cfg = _Demo(figsize=(6, 4), title="hello", style=None).get_config()
    assert cfg["figsize"] == [6, 4]
    assert cfg["title"] == "hello"


def test_validation_panel_does_not_mutate_input_dataset():
    da = xr.DataArray(np.arange(5.0), dims=("lead_time",), name="rmse")
    ds = da.to_dataset()
    ds_before = ds.copy(deep=True)
    LeadTimeSkillPanel()(ds)
    xr.testing.assert_identical(ds, ds_before)


def test_validation_panel_in_sequential_at_last_step_returns_figure():
    da = xr.DataArray(np.arange(5.0), dims=("lead_time",))
    pipeline = Sequential([LeadTimeSkillPanel()])
    out = pipeline(da)
    assert isinstance(out, mpl_figure.Figure)


def test_validation_panel_non_terminal_in_sequential_fails_downstream():
    """A panel placed mid-pipeline must fail downstream — a Figure has no
    Dataset / DataArray methods, so the next op's natural error surfaces.

    Sequential itself is intentionally generic (it pipes any callable),
    so the contract is documented rather than runtime-enforced; this
    test pins the failure mode so a future Sequential rewrite that
    silently accepts a Figure does not regress D10.
    """

    class _NeedsDataArray:
        def __call__(self, x):
            return x.where(x > 0)  # Figure has no .where()

    da = xr.DataArray(np.arange(5.0), dims=("lead_time",))
    pipeline = Sequential([LeadTimeSkillPanel(), _NeedsDataArray()])
    with pytest.raises(AttributeError):
        pipeline(da)


def test_validation_panel_in_graph_emits_figure_alongside_score():
    da = xr.DataArray(np.arange(4.0), dims=("lead_time",), name="rmse")
    inp = Input(name="skill")
    fig_node = LeadTimeSkillPanel()(inp)
    g = Graph(inputs={"skill": inp}, outputs={"figure": fig_node, "raw": inp})
    out = g(skill=da)
    assert isinstance(out["figure"], mpl_figure.Figure)
    xr.testing.assert_identical(out["raw"], da)


# ---- V6.2 V1 panels -----------------------------------------------------


def test_lead_time_skill_panel_renders_dataarray():
    da = xr.DataArray(
        np.array([0.1, 0.2, 0.3, 0.4]),
        coords={"lead_time": np.arange(4)},
        dims=("lead_time",),
        name="rmse",
    )
    fig = LeadTimeSkillPanel()(da)
    assert isinstance(fig, mpl_figure.Figure)


def test_lead_time_skill_panel_renders_dataset_with_legend():
    da_a = xr.DataArray(
        np.arange(4.0),
        coords={"lead_time": np.arange(4)},
        dims=("lead_time",),
        name="rmse",
    )
    da_b = da_a.copy(data=np.arange(4.0) * 1.5).rename("mae")
    ds = xr.Dataset({"rmse": da_a, "mae": da_b})
    fig = LeadTimeSkillPanel()(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_scale_skill_panel_default_metric_picks_first():
    ds = xr.Dataset(
        {
            "rmse": ("region", [0.5, 0.7, 0.6]),
            "mae": ("region", [0.4, 0.5, 0.45]),
        },
        coords={"region": ["GS", "MD", "AC"]},
    )
    fig = ScaleSkillPanel()(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_scale_skill_panel_explicit_metric():
    ds = xr.Dataset(
        {"rmse": ("region", [0.5, 0.7]), "mae": ("region", [0.4, 0.5])},
        coords={"region": ["A", "B"]},
    )
    fig = ScaleSkillPanel(metric="mae")(ds)
    assert isinstance(fig, mpl_figure.Figure)


def test_spectral_skill_panel_uses_log_x_by_default():
    da = xr.DataArray(
        np.linspace(0.5, 1.0, 20),
        coords={"freq": np.logspace(-3, 0, 20)},
        dims=("freq",),
        name="psd_score",
    )
    fig = SpectralSkillPanel()(da)
    assert isinstance(fig, mpl_figure.Figure)


def test_spectral_skill_panel_log_y_toggle():
    da = xr.DataArray(
        np.linspace(0.5, 1.0, 5),
        coords={"freq": np.logspace(-2, 0, 5)},
        dims=("freq",),
    )
    fig = SpectralSkillPanel(log_y=True)(da)
    assert isinstance(fig, mpl_figure.Figure)


# ---- V6.3 V3 + V4 panels ------------------------------------------------


def _trajectory_fixture(n_traj=3, n_t=10):
    rng = np.random.default_rng(0)
    times = np.arange(n_t)
    lons = np.cumsum(rng.normal(0.0, 0.3, size=(n_traj, n_t)), axis=1) - 10.0
    lats = np.cumsum(rng.normal(0.0, 0.2, size=(n_traj, n_t)), axis=1) + 35.0
    return xr.Dataset(
        {
            "lon": (("trajectory", "time"), lons),
            "lat": (("trajectory", "time"), lats),
        },
        coords={"trajectory": np.arange(n_traj), "time": times},
    )


def _eulerian_fixture():
    lon = np.linspace(-15, -5, 21)
    lat = np.linspace(30, 40, 21)
    LON, LAT = np.meshgrid(lon, lat, indexing="xy")
    field = 0.1 * np.sin(np.deg2rad(LON)) * np.cos(np.deg2rad(LAT))
    return xr.Dataset(
        {"ssh": (("lat", "lon"), field)},
        coords={"lat": lat, "lon": lon},
    )


def test_eulerian_lagrangian_panel_renders():
    panel = EulerianLagrangianPanel(eulerian_var="ssh")
    fig = panel(_eulerian_fixture(), _trajectory_fixture())
    assert isinstance(fig, mpl_figure.Figure)


def test_eulerian_lagrangian_panel_handles_eulerian_with_time():
    eul = _eulerian_fixture()
    eul = eul.expand_dims(time=np.arange(2))
    panel = EulerianLagrangianPanel()
    fig = panel(eul, _trajectory_fixture())
    assert isinstance(fig, mpl_figure.Figure)


def test_process_budget_panel_residual_only():
    rng = np.random.default_rng(1)
    time = np.arange(20)
    res = xr.DataArray(
        rng.normal(0.0, 0.01, size=(20, 5, 5)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.arange(5), "lon": np.arange(5)},
    )
    fig = ProcessBudgetPanel()(res)
    assert isinstance(fig, mpl_figure.Figure)


def test_process_budget_panel_with_components():
    rng = np.random.default_rng(2)
    time = np.arange(15)
    base = rng.normal(0.0, 0.05, size=(15, 4, 4))
    coords = {"time": time, "lat": np.arange(4), "lon": np.arange(4)}
    dims = ("time", "lat", "lon")
    components = {
        "tendency": xr.DataArray(base, dims=dims, coords=coords),
        "advection": xr.DataArray(-base * 0.5, dims=dims, coords=coords),
        "source": xr.DataArray(np.zeros_like(base), dims=dims, coords=coords),
    }
    residual = xr.DataArray(base * 0.5, dims=dims, coords=coords)
    fig = ProcessBudgetPanel()(residual, components)
    assert isinstance(fig, mpl_figure.Figure)


# ---- V6.4 V5 panel ------------------------------------------------------


def _objects_fixture(label_grid):
    lon = np.linspace(-10.0, 0.0, label_grid.shape[1])
    lat = np.linspace(30.0, 40.0, label_grid.shape[0])
    return xr.Dataset(
        {"label": (("lat", "lon"), label_grid)},
        coords={"lat": lat, "lon": lon},
    )


def test_event_verification_panel_renders_basic():
    pred = np.zeros((10, 12), dtype=int)
    pred[2:5, 2:5] = 1
    pred[6:8, 7:10] = 2
    ref = np.zeros((10, 12), dtype=int)
    ref[2:5, 2:5] = 10
    ref[7:9, 1:3] = 20
    matches = {
        "hits": [(1, 10)],
        "false_alarms": [2],
        "misses": [20],
    }
    scores = {"POD": 0.5, "FAR": 0.5, "CSI": 0.33, "IoU": 0.4}
    fig = EventVerificationPanel(use_cartopy=False)(
        _objects_fixture(pred), _objects_fixture(ref), matches, scores
    )
    assert isinstance(fig, mpl_figure.Figure)


def test_event_verification_panel_handles_no_hits():
    pred = np.zeros((6, 6), dtype=int)
    pred[1:3, 1:3] = 1
    ref = np.zeros((6, 6), dtype=int)
    ref[3:5, 3:5] = 7
    matches = {"hits": [], "false_alarms": [1], "misses": [7]}
    scores = {"POD": 0.0, "FAR": 1.0, "CSI": 0.0, "IoU": 0.0}
    fig = EventVerificationPanel(use_cartopy=False)(
        _objects_fixture(pred), _objects_fixture(ref), matches, scores
    )
    assert isinstance(fig, mpl_figure.Figure)


# ---- viz package surface ------------------------------------------------


def test_viz_validation_top_level_imports():
    import xr_toolz.viz.validation as vv

    for name in (
        "LeadTimeSkillPanel",
        "ScaleSkillPanel",
        "SpectralSkillPanel",
        "EulerianLagrangianPanel",
        "ProcessBudgetPanel",
        "EventVerificationPanel",
    ):
        assert hasattr(vv, name)
