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
)


def _paired(size: int = 2, names: list[str] | None = None) -> xr.Dataset:
    coords = {
        "method": names or [f"m{i}" for i in range(size)],
        "lat": np.linspace(-5, 5, 5),
        "lon": np.linspace(-5, 5, 6),
    }
    return xr.DataArray(
        np.arange(float(size * 30)).reshape(size, 5, 6) + 1.0,
        dims=("method", "lat", "lon"),
        coords=coords,
    ).to_dataset(name="rmse")


def _record_calls() -> tuple[list[tuple], object]:
    calls: list[tuple] = []

    def panel(ds, ax) -> None:
        calls.append((ds, ax))
        ax.plot([0, 1], [0, 1])

    return calls, panel


def _main_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_renders_three_cells_by_default() -> None:
    calls, panel = _record_calls()

    fig = PairwiseComparePanel(panel)(_paired(names=["duacs", "miost"]))

    assert len(calls) == 3
    assert [ax.get_title() for ax in fig.axes] == [
        "duacs",
        "miost",
        "duacs → miost: Δ%",
    ]


def test_diff_none_renders_only_two_cells() -> None:
    calls, panel = _record_calls()

    fig = PairwiseComparePanel(panel, diff="none")(_paired())

    assert len(calls) == 2
    assert len(fig.axes) == 2


def test_absolute_diff_is_study_minus_ref() -> None:
    calls, panel = _record_calls()
    ds = _paired()

    PairwiseComparePanel(panel, diff="absolute")(ds)

    expected = ds.isel(method=1) - ds.isel(method=0)
    xr.testing.assert_allclose(calls[2][0], expected)


def test_relative_diff_is_a_percentage_of_the_reference() -> None:
    calls, panel = _record_calls()
    ds = _paired()

    PairwiseComparePanel(panel, diff="relative")(ds)

    ref, study = ds.isel(method=0), ds.isel(method=1)
    xr.testing.assert_allclose(calls[2][0], 100.0 * (study - ref) / ref)


def test_relative_diff_guards_a_zero_reference() -> None:
    calls, panel = _record_calls()
    ds = _paired()
    ds["rmse"][0, 0, 0] = 0.0

    PairwiseComparePanel(panel, diff="relative")(ds)

    # 0 reference has no meaningful percentage change — NaN, not inf.
    assert np.isnan(calls[2][0]["rmse"].values[0, 0])
    assert np.isfinite(calls[2][0]["rmse"].values[1:]).all()


def test_column_layout_stacks_the_cells() -> None:
    _, panel = _record_calls()

    fig = PairwiseComparePanel(panel, layout="col")(_paired())

    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (3, 1)


def test_diff_label_overrides_the_default() -> None:
    _, panel = _record_calls()

    fig = PairwiseComparePanel(panel, diff_label="bias")(_paired(names=["a", "b"]))

    assert fig.axes[2].get_title() == "a → b: bias"


def test_wrong_method_count_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match=r"exactly 2 entries along 'method'"):
        PairwiseComparePanel(panel)(_paired(size=3))


def test_missing_method_dim_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match=r"method_dim 'nope' is not a dimension"):
        PairwiseComparePanel(panel, method_dim="nope")(_paired())


def test_unknown_diff_mode_raises_at_construction() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="unknown diff mode"):
        PairwiseComparePanel(panel, diff="bogus")


def test_unknown_layout_raises_at_construction() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="unknown layout"):
        PairwiseComparePanel(panel, layout="diagonal")


def test_diff_kwargs_apply_only_to_the_diff_cell() -> None:
    inner = SpatialMapPanel(variable="rmse", cmap="viridis")
    seen: list[str | None] = []

    original = inner._build

    def spy(fig, ax, ds):
        seen.append(inner.cmap)
        return original(fig, ax, ds)

    inner._build = spy  # type: ignore[method-assign]

    PairwiseComparePanel(inner, diff_kwargs={"cmap": "coolwarm"})(_paired())

    assert seen == ["viridis", "viridis", "coolwarm"]
    # The override is scoped to the diff render, not left behind.
    assert inner.cmap == "viridis"


def test_diff_kwargs_on_a_callable_inner_warns() -> None:
    _, panel = _record_calls()

    with pytest.warns(UserWarning, match="no attributes to override"):
        PairwiseComparePanel(panel, diff_kwargs={"cmap": "coolwarm"})(_paired())


def test_projection_is_forwarded_from_the_inner_panel() -> None:
    import cartopy.mpl.geoaxes as cgeo

    fig = PairwiseComparePanel(
        SpatialMapPanel(variable="rmse", projection="north_atlantic")
    )(_paired())

    assert all(isinstance(ax, cgeo.GeoAxes) for ax in _main_axes(fig))


def test_method_dim_without_coords_uses_positional_names() -> None:
    _, panel = _record_calls()
    ds = xr.DataArray(np.ones((2, 4)), dims=("method", "x")).to_dataset(name="rmse")

    fig = PairwiseComparePanel(panel)(ds)

    assert [ax.get_title() for ax in fig.axes[:2]] == ["reference", "study"]


def test_config_recurses_and_round_trips_as_json() -> None:
    panel = PairwiseComparePanel(
        SpatialMapPanel(variable="rmse"), diff="absolute", layout="col"
    )

    config = panel.get_config()

    assert config["panel"] == "SpatialMapPanel"
    assert config["panel_kind"] == "validation_panel"
    assert config["diff"] == "absolute"
    assert config["layout"] == "col"
    assert json.loads(json.dumps(config)) == config


def test_composes_under_facet_into_a_six_panel_mosaic() -> None:
    ds = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 5, 6)),
        dims=("scale", "method", "lat", "lon"),
        coords={
            "scale": ["large", "small"],
            "method": ["duacs", "miost"],
            "lat": np.linspace(-5, 5, 5),
            "lon": np.linspace(-5, 5, 6),
        },
    ).to_dataset(name="rmse")

    fig = FacetPanel(
        PairwiseComparePanel(SpatialMapPanel(variable="rmse")), facet_dim="scale"
    )(ds)

    cells = _main_axes(fig)
    assert len(cells) == 6
    # The facet label is prefixed onto each row's leading cell rather than
    # overwriting the inner panel's own method titles.
    assert cells[0].get_title() == "large\nduacs"
    assert cells[3].get_title() == "small\nduacs"
    assert cells[2].get_title() == "duacs → miost: Δ%"


def test_nested_projection_reaches_the_facet_grid() -> None:
    import cartopy.mpl.geoaxes as cgeo

    ds = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 5, 6)),
        dims=("scale", "method", "lat", "lon"),
        coords={
            "scale": ["large", "small"],
            "method": ["duacs", "miost"],
            "lat": np.linspace(-5, 5, 5),
            "lon": np.linspace(-5, 5, 6),
        },
    ).to_dataset(name="rmse")

    # The projection lives two levels down, on the SpatialMapPanel.
    fig = FacetPanel(
        PairwiseComparePanel(
            SpatialMapPanel(variable="rmse", projection="north_atlantic")
        ),
        facet_dim="scale",
    )(ds)

    assert all(isinstance(ax, cgeo.GeoAxes) for ax in _main_axes(fig))


def test_preset_extent_is_applied_to_the_triptych() -> None:
    from xrtoolz.viz._src.projections import PRESETS

    fig = PairwiseComparePanel(
        SpatialMapPanel(variable="rmse", projection="north_atlantic")
    )(_paired())

    expected = PRESETS["north_atlantic"]["extent"]
    for ax in _main_axes(fig):
        assert ax.get_extent(crs=ax.projection.as_geodetic()) == pytest.approx(
            expected, abs=1.0
        )


def test_nested_sharebar_finds_a_mappable_in_the_subdivided_cell() -> None:
    ds = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 5, 6)),
        dims=("scale", "method", "lat", "lon"),
        coords={
            "scale": ["large", "small"],
            "method": ["duacs", "miost"],
            "lat": np.linspace(-5, 5, 5),
            "lon": np.linspace(-5, 5, 6),
        },
    ).to_dataset(name="rmse")

    fig = FacetPanel(
        PairwiseComparePanel(SpatialMapPanel(variable="rmse")),
        facet_dim="scale",
        sharebar=True,
    )(ds)

    colorbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    # A subdivided cell hands over an axes array; without searching its
    # sub-axes the shared bar would find no mappable and vanish.
    assert len(colorbars) == 1
    assert len(_main_axes(fig)) == 6


def test_nested_preset_extent_reaches_subdivided_cells() -> None:
    from xrtoolz.viz._src.projections import PRESETS

    ds = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 5, 6)),
        dims=("scale", "method", "lat", "lon"),
        coords={
            "scale": ["large", "small"],
            "method": ["duacs", "miost"],
            "lat": np.linspace(-5, 5, 5),
            "lon": np.linspace(-5, 5, 6),
        },
    ).to_dataset(name="rmse")

    fig = FacetPanel(
        PairwiseComparePanel(
            SpatialMapPanel(variable="rmse", projection="north_atlantic")
        ),
        facet_dim="scale",
    )(ds)

    # Each facet cell holds its own axes array, so a single ravel would
    # hand set_extent an array instead of a GeoAxes and silently skip it.
    expected = PRESETS["north_atlantic"]["extent"]
    for ax in _main_axes(fig):
        assert ax.get_extent(crs=ax.projection.as_geodetic()) == pytest.approx(
            expected, abs=1.0
        )


def test_multi_input_panels_are_rejected_at_construction() -> None:
    from xrtoolz.viz.validation import EventVerificationPanel

    with pytest.raises(TypeError, match="cannot be wrapped"):
        PairwiseComparePanel(EventVerificationPanel())
