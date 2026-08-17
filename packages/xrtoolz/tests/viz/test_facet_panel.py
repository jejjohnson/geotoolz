from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

from xrtoolz.viz.validation import (
    FacetPanel,
    PairwiseComparePanel,
    SpatialMapPanel,
    seasonal_groupby,
)


def _faceted(n: int, dim: str = "experiment") -> xr.Dataset:
    values = [f"exp{i}" for i in range(n)]
    return xr.DataArray(
        np.random.default_rng(0).random((n, 5, 6)),
        dims=(dim, "lat", "lon"),
        coords={
            dim: values,
            "lat": np.linspace(-5, 5, 5),
            "lon": np.linspace(-5, 5, 6),
        },
    ).to_dataset(name="ssh")


def _record_calls() -> tuple[list[tuple], object]:
    """A callable inner panel that logs each (ds, ax) it receives."""
    calls: list[tuple] = []

    def panel(ds, ax) -> None:
        calls.append((ds, ax))
        ax.plot([0, 1], [0, 1])

    return calls, panel


def _main_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_four_facets_default_to_a_two_by_two_grid() -> None:
    calls, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment")(_faceted(4))

    assert len(calls) == 4
    assert {ax.get_subplotspec().get_gridspec().get_geometry() for ax in fig.axes} == {
        (2, 2)
    }


def test_explicit_nrows_and_ncols_are_honoured() -> None:
    _, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment", nrows=1, ncols=4)(_faceted(4))

    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 4)


def test_five_facets_hide_the_trailing_cell() -> None:
    _, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment")(_faceted(5))

    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)
    assert sum(ax.get_visible() for ax in fig.axes) == 5
    assert len(fig.axes) == 6


def test_layout_too_small_for_the_facet_count_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match=r"cannot hold 4 facets"):
        FacetPanel(panel, facet_dim="experiment", nrows=1, ncols=2)(_faceted(4))


def test_missing_facet_dim_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match=r"facet_dim 'nope' is not a dimension"):
        FacetPanel(panel, facet_dim="nope")(_faceted(3))


def test_titles_default_to_the_coord_value() -> None:
    _, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment")(_faceted(4))

    assert [ax.get_title() for ax in fig.axes[:4]] == ["exp0", "exp1", "exp2", "exp3"]


def test_title_format_receives_value_and_index() -> None:
    _, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment", title_format="{value} ({index})")(
        _faceted(2)
    )

    assert [ax.get_title() for ax in fig.axes[:2]] == ["exp0 (0)", "exp1 (1)"]


def test_each_slice_is_passed_to_its_own_axes() -> None:
    calls, panel = _record_calls()
    ds = _faceted(3)

    fig = FacetPanel(panel, facet_dim="experiment")(ds)

    assert [ax for _, ax in calls] == fig.axes[:3]
    for index, (received, _) in enumerate(calls):
        xr.testing.assert_identical(received, ds.isel(experiment=index))


def test_validation_panel_inner_renders_per_slice() -> None:
    fig = FacetPanel(SpatialMapPanel(variable="ssh"), facet_dim="experiment")(
        _faceted(4)
    )

    # SpatialMapPanel draws its own colorbar per cell.
    assert len(_main_axes(fig)) == 4


def test_projection_is_forwarded_from_the_inner_panel() -> None:
    import cartopy.mpl.geoaxes as cgeo

    fig = FacetPanel(
        SpatialMapPanel(variable="ssh", projection="north_atlantic"),
        facet_dim="experiment",
    )(_faceted(2))

    assert all(isinstance(ax, cgeo.GeoAxes) for ax in _main_axes(fig))


def test_explicit_subplot_kw_overrides_the_inner_projection() -> None:
    import cartopy.crs as ccrs

    fig = FacetPanel(
        SpatialMapPanel(variable="ssh", projection="north_atlantic"),
        facet_dim="experiment",
        subplot_kw={"projection": ccrs.Mollweide()},
    )(_faceted(2))

    assert all(isinstance(ax.projection, ccrs.Mollweide) for ax in _main_axes(fig))


def test_empty_subplot_kw_forces_plain_axes_for_a_plain_panel() -> None:
    import cartopy.mpl.geoaxes as cgeo

    fig = FacetPanel(
        SpatialMapPanel(variable="ssh"), facet_dim="experiment", subplot_kw={}
    )(_faceted(2))

    assert not any(isinstance(ax, cgeo.GeoAxes) for ax in _main_axes(fig))


def test_sharebar_replaces_per_cell_colorbars_with_one() -> None:
    fig = FacetPanel(
        SpatialMapPanel(variable="ssh"), facet_dim="experiment", sharebar=True
    )(_faceted(4))

    # 4 cells + exactly one shared colorbar, not one bar per cell.
    assert len(fig.axes) == 5
    assert len(_main_axes(fig)) == 4


def test_sharebar_without_a_mappable_warns_and_falls_back() -> None:
    _, panel = _record_calls()

    with pytest.warns(UserWarning, match="no mappable"):
        fig = FacetPanel(panel, facet_dim="experiment", sharebar=True)(_faceted(3))

    assert len(fig.axes) == 4  # 2x2 grid, no colorbar added


def test_config_recurses_into_a_validation_panel_inner() -> None:
    panel = FacetPanel(SpatialMapPanel(variable="ssh"), facet_dim="experiment", ncols=2)

    config = panel.get_config()

    assert config["panel"] == "SpatialMapPanel"
    assert config["panel_kind"] == "validation_panel"
    assert config["panel_config"]["variable"] == "ssh"
    assert config["facet_dim"] == "experiment"
    assert json.loads(json.dumps(config)) == config


def test_config_flags_a_callable_inner_as_non_roundtrippable() -> None:
    _, panel = _record_calls()

    config = FacetPanel(panel, facet_dim="experiment").get_config()

    assert config["panel_kind"] == "callable"
    assert config["panel_config"] is None
    assert json.loads(json.dumps(config)) == config


def test_facet_dim_without_coords_falls_back_to_positional_labels() -> None:
    _, panel = _record_calls()
    ds = xr.DataArray(np.zeros((3, 4)), dims=("run", "x")).to_dataset(name="score")

    fig = FacetPanel(panel, facet_dim="run")(ds)

    assert [ax.get_title() for ax in fig.axes[:3]] == ["0", "1", "2"]


def test_savefig_is_inherited_from_the_panel_base(tmp_path) -> None:
    _, panel = _record_calls()
    out = tmp_path / "nested" / "facet.png"

    FacetPanel(panel, facet_dim="experiment", savefig=out)(_faceted(2))

    assert out.exists()


def test_seasonal_groupby_reduces_time_to_four_seasons() -> None:
    import pandas as pd

    ds = xr.DataArray(
        np.arange(365.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=365)},
    ).to_dataset(name="sst")

    seasonal = seasonal_groupby(ds)

    assert seasonal.sizes["season"] == 4
    assert set(seasonal["season"].values) == {"DJF", "MAM", "JJA", "SON"}


def test_seasonal_groupby_honours_the_reduction() -> None:
    import pandas as pd

    ds = xr.DataArray(
        np.ones(365),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=365)},
    ).to_dataset(name="sst")

    assert float(seasonal_groupby(ds, reduction="sum")["sst"].sum()) == 365.0


def test_seasonal_groupby_rejects_an_unknown_reduction() -> None:
    import pandas as pd

    ds = xr.DataArray(
        np.ones(10),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=10)},
    ).to_dataset(name="sst")

    with pytest.raises(ValueError, match="unknown reduction"):
        seasonal_groupby(ds, reduction="bogus")


def test_seasonal_groupby_feeds_straight_into_a_facet_mosaic() -> None:
    import pandas as pd

    _, panel = _record_calls()
    ds = xr.DataArray(
        np.arange(365.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=365)},
    ).to_dataset(name="sst")

    fig = FacetPanel(panel, facet_dim="season", ncols=2)(seasonal_groupby(ds))

    assert len(fig.axes) == 4
    assert set(ax.get_title() for ax in fig.axes) == {"DJF", "MAM", "JJA", "SON"}


def test_empty_facet_dim_raises_informatively() -> None:
    _, panel = _record_calls()
    empty = _faceted(3).isel(experiment=slice(0, 0))

    with pytest.raises(ValueError, match=r"facet_dim 'experiment' is empty"):
        FacetPanel(panel, facet_dim="experiment")(empty)


def test_dataset_may_be_supplied_as_a_keyword() -> None:
    calls, panel = _record_calls()
    ds = _faceted(3)

    fig = FacetPanel(panel, facet_dim="experiment")(ds=ds)

    assert len(calls) == 3
    assert len(fig.axes) == 4


def test_calling_without_any_input_raises_typeerror() -> None:
    _, panel = _record_calls()

    with pytest.raises(TypeError, match="requires an input dataset"):
        FacetPanel(panel, facet_dim="experiment")()


def test_preset_extent_is_applied_to_wrapper_built_axes() -> None:
    from xrtoolz.viz._src.projections import PRESETS

    fig = FacetPanel(
        SpatialMapPanel(variable="ssh", projection="north_atlantic"),
        facet_dim="experiment",
    )(_faceted(2))

    expected = PRESETS["north_atlantic"]["extent"]
    for ax in _main_axes(fig):
        assert ax.get_extent(crs=ax.projection.as_geodetic()) == pytest.approx(
            expected, abs=1.0
        )


def test_sharex_is_wired_across_cells() -> None:
    _, panel = _record_calls()

    fig = FacetPanel(panel, facet_dim="experiment", sharex=True)(_faceted(4))

    first = fig.axes[0]
    assert all(ax in first.get_shared_x_axes().get_siblings(first) for ax in fig.axes)


def test_sharing_survives_a_subdivided_facet_grid() -> None:
    ds = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 4)),
        dims=("scale", "method", "x"),
        coords={"scale": ["large", "small"], "method": ["a", "b"]},
    ).to_dataset(name="rmse")
    _, inner = _record_calls()

    fig = FacetPanel(PairwiseComparePanel(inner), facet_dim="scale", sharex=True)(ds)

    first = fig.axes[0]
    siblings = first.get_shared_x_axes().get_siblings(first)
    assert all(ax in siblings for ax in fig.axes)


def test_seasons_come_back_in_chronological_order() -> None:
    import pandas as pd

    ds = xr.DataArray(
        np.arange(365.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=365)},
    ).to_dataset(name="sst")

    # Grouping on the string season coord sorts lexicographically, which
    # would put JJA before MAM in the mosaic.
    assert list(seasonal_groupby(ds)["season"].values) == ["DJF", "MAM", "JJA", "SON"]


def test_partial_year_keeps_only_the_seasons_present() -> None:
    import pandas as pd

    ds = xr.DataArray(
        np.arange(120.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=120)},
    ).to_dataset(name="sst")

    assert list(seasonal_groupby(ds)["season"].values) == ["DJF", "MAM"]


def test_sharebar_keeps_per_cell_bars_when_no_mappable_exists() -> None:
    calls: list = []

    def bar_drawing_panel(ds, ax) -> None:
        # Draws a colorbar from a standalone mappable, leaving nothing
        # attached to the main axes for the shared bar to reuse.
        import matplotlib.cm as mpl_cm
        import matplotlib.colors as mpl_colors

        mappable = mpl_cm.ScalarMappable(norm=mpl_colors.Normalize(0, 1))
        ax.figure.colorbar(mappable, ax=ax)
        calls.append(ds)

    with pytest.warns(UserWarning, match="no mappable"):
        fig = FacetPanel(bar_drawing_panel, facet_dim="experiment", sharebar=True)(
            _faceted(2)
        )

    # The warning promises the panel's own colorbars survive, so they must.
    assert len([ax for ax in fig.axes if ax.get_label() == "<colorbar>"]) == 2


def test_sharebar_puts_every_facet_on_one_scale() -> None:
    import matplotlib.cm as mpl_cm

    # Facet 0 spans [1, 1], facet 1 spans [100, 100]; autoscaled
    # independently the shared bar would describe only one of them.
    ds = (
        xr.concat(
            [
                xr.DataArray(
                    np.full((5, 6), value),
                    dims=("lat", "lon"),
                    coords={
                        "lat": np.linspace(-5, 5, 5),
                        "lon": np.linspace(-5, 5, 6),
                    },
                )
                for value in (1.0, 100.0)
            ],
            dim="experiment",
        )
        .assign_coords(experiment=["a", "b"])
        .to_dataset(name="ssh")
    )

    fig = FacetPanel(
        SpatialMapPanel(variable="ssh"), facet_dim="experiment", sharebar=True
    )(ds)

    clims = {
        artist.get_clim()
        for ax in _main_axes(fig)
        for artist in ax.collections
        if isinstance(artist, mpl_cm.ScalarMappable)
    }
    assert len(clims) == 1
    low, high = clims.pop()
    assert low <= 1.0 and high >= 100.0


@pytest.mark.parametrize(
    "panel_cls", ["EulerianLagrangianPanel", "EventVerificationPanel"]
)
def test_multi_input_panels_are_rejected_at_construction(panel_cls: str) -> None:
    import xrtoolz.viz.validation as validation

    inner = getattr(validation, panel_cls)()

    with pytest.raises(TypeError, match="cannot be wrapped"):
        FacetPanel(inner, facet_dim="experiment")
