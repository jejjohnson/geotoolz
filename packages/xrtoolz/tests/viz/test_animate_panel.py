from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest
import xarray as xr


matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation

from xrtoolz.viz.validation import (
    AnimatePanel,
    FacetPanel,
    PairwiseComparePanel,
    SpatialMapPanel,
    save_animation,
)


def _frames(n: int = 4, extra: dict | None = None) -> xr.Dataset:
    dims = ("time", *(extra or {}), "lat", "lon")
    shape = (n, *(len(v) for v in (extra or {}).values()), 5, 6)
    coords = {
        "time": np.arange(n),
        **(extra or {}),
        "lat": np.linspace(-5, 5, 5),
        "lon": np.linspace(-5, 5, 6),
    }
    return xr.DataArray(
        np.random.default_rng(0).random(shape), dims=dims, coords=coords
    ).to_dataset(name="ssh")


def _record_calls() -> tuple[list[tuple], object]:
    calls: list[tuple] = []

    def panel(ds, ax) -> None:
        calls.append((ds, ax))
        ax.plot([0, 1], [0, 1])

    return calls, panel


def _run_all_frames(ani: FuncAnimation, n: int) -> None:
    for index in range(n):
        ani._func(index)


def test_returns_a_funcanimation_over_every_frame() -> None:
    _, panel = _record_calls()

    ani = AnimatePanel(panel)(_frames(4))

    assert isinstance(ani, FuncAnimation)
    assert ani._save_count == 4


def test_each_frame_receives_its_own_slice() -> None:
    calls, panel = _record_calls()
    ds = _frames(3)

    _run_all_frames(AnimatePanel(panel)(ds), 3)

    assert len(calls) == 3
    for index, (received, _) in enumerate(calls):
        xr.testing.assert_identical(received, ds.isel(time=index))


def test_frames_reuse_a_single_axes() -> None:
    calls, panel = _record_calls()

    _run_all_frames(AnimatePanel(panel)(_frames(5)), 5)

    assert len({id(ax) for _, ax in calls}) == 1


def test_colorbar_axes_do_not_accumulate_across_frames() -> None:
    # SpatialMapPanel calls fig.colorbar inside _build, so without
    # reclaiming them a long animation grows one colorbar per frame.
    ani = AnimatePanel(SpatialMapPanel(variable="ssh"))(_frames(8))

    _run_all_frames(ani, 8)

    assert len(ani._fig.axes) == 2  # the map plus exactly one colorbar


def test_title_defaults_to_the_frame_coord_value() -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(3))

    ani._func(2)

    assert ani._fig.axes[0].get_title() == "2"


def test_title_format_receives_value_and_index() -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel, title_format="t={value} (#{index})")(_frames(3))

    ani._func(1)

    assert ani._fig.axes[0].get_title() == "t=1 (#1)"


def test_interval_derives_from_fps() -> None:
    _, panel = _record_calls()

    ani = AnimatePanel(panel, fps=25)(_frames(2))

    assert ani._interval == pytest.approx(40.0)


def test_explicit_interval_wins_over_fps() -> None:
    _, panel = _record_calls()

    ani = AnimatePanel(panel, fps=25, interval_ms=200)(_frames(2))

    assert ani._interval == pytest.approx(200.0)


def test_pixel_dimensions_override_figsize() -> None:
    _, panel = _record_calls()

    ani = AnimatePanel(panel, pixelwidth=800, pixelheight=400, dpi=100)(_frames(2))

    assert tuple(ani._fig.get_size_inches()) == pytest.approx((8.0, 4.0))


def test_non_positive_fps_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="fps must be positive"):
        AnimatePanel(panel, fps=0)


def test_missing_frame_dim_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match=r"frame_dim 'nope' is not a dimension"):
        AnimatePanel(panel, frame_dim="nope")(_frames(2))


def test_empty_frame_dim_raises() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="no frames to animate"):
        AnimatePanel(panel)(_frames(4).isel(time=slice(0, 0)))


def test_preview_renders_one_frame() -> None:
    calls, panel = _record_calls()
    ds = _frames(5)

    fig, _ = AnimatePanel(panel).preview(ds, frame_index=3)

    assert len(calls) == 1
    xr.testing.assert_identical(calls[0][0], ds.isel(time=3))
    assert fig.axes[0].get_title() == "3"


def test_preview_rejects_an_out_of_range_frame() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="out of range"):
        AnimatePanel(panel).preview(_frames(3), frame_index=7)


def test_projection_is_forwarded_from_the_inner_panel() -> None:
    import cartopy.mpl.geoaxes as cgeo

    fig, _ = AnimatePanel(
        SpatialMapPanel(variable="ssh", projection="north_atlantic")
    ).preview(_frames(2))

    assert isinstance(fig.axes[0], cgeo.GeoAxes)


def test_config_round_trips_as_json() -> None:
    config = AnimatePanel(SpatialMapPanel(variable="ssh"), fps=12).get_config()

    assert config["panel"] == "SpatialMapPanel"
    assert config["fps"] == 12
    assert config["panel_kind"] == "validation_panel"
    assert json.loads(json.dumps(config)) == config


def test_repr_names_the_inner_panel() -> None:
    text = repr(AnimatePanel(SpatialMapPanel(variable="ssh"), frame_dim="lead"))

    assert "AnimatePanel" in text
    assert "SpatialMapPanel" in text
    assert "lead" in text


def test_wraps_a_facet_panel_and_labels_frames_on_the_figure() -> None:
    ds = _frames(3, extra={"experiment": ["a", "b", "c", "d"]})

    ani = AnimatePanel(
        FacetPanel(SpatialMapPanel(variable="ssh"), facet_dim="experiment")
    )(ds)
    _run_all_frames(ani, 3)

    cells = [ax for ax in ani._fig.axes if ax.get_label() != "<colorbar>"]
    assert len(cells) == 4
    # Per-cell titles belong to the facet; the frame label goes on the figure.
    assert ani._fig._suptitle.get_text() == "2"
    assert [ax.get_title() for ax in cells] == ["a", "b", "c", "d"]


def test_wraps_a_pairwise_compare_panel() -> None:
    ds = _frames(3, extra={"method": ["duacs", "miost"]})

    ani = AnimatePanel(PairwiseComparePanel(SpatialMapPanel(variable="ssh")))(ds)
    _run_all_frames(ani, 3)

    cells = [ax for ax in ani._fig.axes if ax.get_label() != "<colorbar>"]
    assert len(cells) == 3
    assert cells[2].get_title() == "duacs → miost: Δ%"


def test_save_animation_writes_a_gif(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(3))
    out = tmp_path / "nested" / "movie.gif"

    written = save_animation(ani, out, fps=5)

    assert written == out
    assert out.stat().st_size > 0


def test_save_animation_writes_self_contained_html(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(2))
    out = tmp_path / "movie.html"

    save_animation(ani, out)

    assert "<script" in out.read_text(encoding="utf-8")


def test_save_animation_refuses_to_clobber_by_default(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(2))
    out = tmp_path / "movie.gif"
    out.write_text("existing")

    with pytest.raises(FileExistsError, match="overwrite_existing=True"):
        save_animation(ani, out)


def test_save_animation_overwrites_when_asked(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(2))
    out = tmp_path / "movie.gif"
    out.write_text("existing")

    save_animation(ani, out, overwrite_existing=True)

    assert out.read_bytes()[:3] == b"GIF"


def test_save_animation_rejects_an_unknown_extension(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel)(_frames(2))

    with pytest.raises(ValueError, match="unsupported animation format"):
        save_animation(ani, tmp_path / "movie.avi")


def test_blit_true_is_rejected() -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="blit=True is not supported"):
        AnimatePanel(panel, blit=True)


@pytest.mark.parametrize("kwargs", [{"pixelwidth": 800}, {"pixelheight": 400}])
def test_incomplete_pixel_size_is_rejected(kwargs: dict) -> None:
    _, panel = _record_calls()

    with pytest.raises(ValueError, match="must be given together"):
        AnimatePanel(panel, **kwargs)


def test_explicit_figsize_is_applied_to_a_composite_inner() -> None:
    # FacetPanel sizes itself from figsize_per_panel and never reads a
    # `figsize` attribute, so the request has to land on the figure.
    ds = _frames(2, extra={"experiment": ["a", "b", "c", "d"]})

    ani = AnimatePanel(
        FacetPanel(SpatialMapPanel(variable="ssh"), facet_dim="experiment"),
        figsize=(8, 6),
    )(ds)

    assert tuple(ani._fig.get_size_inches()) == pytest.approx((8.0, 6.0))


def test_composite_inner_sizes_itself_when_figsize_is_unset() -> None:
    ds = _frames(2, extra={"method": ["duacs", "miost"]})

    ani = AnimatePanel(PairwiseComparePanel(SpatialMapPanel(variable="ssh")))(ds)

    # PairwiseComparePanel's own 3 x (5, 4) layout, not a generic default.
    assert tuple(ani._fig.get_size_inches()) == pytest.approx((15.0, 4.0))


def test_save_animation_defaults_to_the_panels_configured_fps(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel, fps=5)(_frames(2))
    out = tmp_path / "movie.html"

    save_animation(ani, out)

    # to_jshtml embeds the frame interval; 5 fps is 200 ms, not 24 fps.
    assert "200" in out.read_text(encoding="utf-8")


def test_explicit_save_fps_overrides_the_panel(tmp_path) -> None:
    _, panel = _record_calls()
    ani = AnimatePanel(panel, fps=5)(_frames(2))
    out = tmp_path / "movie.html"

    save_animation(ani, out, fps=20)

    assert "50" in out.read_text(encoding="utf-8")


def test_animation_carries_its_configured_fps() -> None:
    _, panel = _record_calls()

    ani = AnimatePanel(panel, fps=7)(_frames(2))

    assert ani.xrtoolz_fps == 7


def test_subplot_kw_does_not_flatten_a_composite_layout() -> None:
    import cartopy.crs as ccrs

    ds = _frames(2, extra={"experiment": ["a", "b", "c", "d"]})

    # An explicit subplot_kw must not bypass FacetPanel's grid builder,
    # or the frame would render into a single Axes and index past it.
    ani = AnimatePanel(
        FacetPanel(SpatialMapPanel(variable="ssh"), facet_dim="experiment"),
        subplot_kw={"projection": ccrs.Mollweide()},
    )(ds)
    _run_all_frames(ani, 2)

    cells = [ax for ax in ani._fig.axes if ax.get_label() != "<colorbar>"]
    assert len(cells) == 4
    assert all(isinstance(ax.projection, ccrs.Mollweide) for ax in cells)


def test_subplot_kw_preserves_the_pairwise_triptych() -> None:
    import cartopy.crs as ccrs

    ds = _frames(2, extra={"method": ["duacs", "miost"]})

    fig, _ = AnimatePanel(
        PairwiseComparePanel(SpatialMapPanel(variable="ssh")),
        subplot_kw={"projection": ccrs.Mollweide()},
    ).preview(ds)

    assert len([ax for ax in fig.axes if ax.get_label() != "<colorbar>"]) == 3


def test_multi_input_panels_are_rejected_at_construction() -> None:
    from xrtoolz.viz.validation import EulerianLagrangianPanel

    with pytest.raises(TypeError, match="cannot be wrapped"):
        AnimatePanel(EulerianLagrangianPanel())
