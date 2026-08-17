"""Shared plumbing for the composable panel wrappers.

:class:`~xrtoolz.viz.validation.FacetPanel`,
:class:`~xrtoolz.viz.validation.PairwiseComparePanel` and
:class:`~xrtoolz.viz.validation.AnimatePanel` all wrap an *inner* panel
and render it into axes they own. The mechanics they share live here:
dispatching to the inner renderer, forwarding cartopy projections,
recovering a mappable for shared colorbars, and keeping the figure's
axes list from growing when a panel is rendered repeatedly.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.cm as mpl_cm
import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel


#: An inner panel is either a ``_ValidationPanel`` or a bare
#: ``(ds, ax) -> Any`` callable.
InnerPanel = _ValidationPanel | Callable[[xr.Dataset, Any], Any]


def _innermost_projection(panel: InnerPanel) -> Any:
    """Find the projection spec owned anywhere down the wrapper chain.

    Wrappers hold their inner panel on ``.panel``, so
    ``FacetPanel(PairwiseComparePanel(SpatialMapPanel(projection=...)))``
    hides the projection two levels down. Looking only at the immediate
    inner panel would build plain axes and then let the spatial panel
    call ``coastlines()`` on them.
    """
    seen: set[int] = set()
    current: Any = panel
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        projection = getattr(current, "projection", None)
        if projection is not None:
            return projection
        current = getattr(current, "panel", None)
    return None


def _resolve_subplot_kw(
    panel: InnerPanel, override: dict[str, Any] | None
) -> dict[str, Any]:
    """Build ``subplot_kw`` for a grid hosting ``panel``.

    Panels that need cartopy axes advertise it through a ``projection``
    attribute (the convention :class:`SpatialMapPanel` established), and
    the search walks nested wrappers to find it. An explicit ``override``
    always wins — including an empty dict, which deliberately forces
    plain matplotlib axes.
    """
    if override is not None:
        return dict(override)
    projection = _innermost_projection(panel)
    if projection is None:
        return {}
    from xrtoolz.viz._src.projections import _resolve_projection

    crs = _resolve_projection(projection)
    return {} if crs is None else {"projection": crs}


def _apply_preset_extent(panel: InnerPanel, axes: Any) -> None:
    """Zoom wrapper-created GeoAxes to a named preset's extent.

    :meth:`SpatialMapPanel._make_fig_axes` calls ``set_extent`` for a
    preset such as ``"north_atlantic"``; axes that a wrapper builds
    itself never went through that path, so without this they render the
    projection's whole default domain instead of the requested region.
    """
    projection = _innermost_projection(panel)
    if not isinstance(projection, str):
        return
    from xrtoolz.viz._src.projections import PRESETS

    preset = PRESETS.get(projection)
    if preset is None or preset["extent"] is None:
        return

    import cartopy.crs as ccrs

    for ax in _flatten_axes(axes):
        if hasattr(ax, "set_extent"):
            ax.set_extent(preset["extent"], crs=ccrs.PlateCarree())


def _flatten_axes(axes: Any) -> list[Any]:
    """Flatten an Axes, an array of them, or an array of arrays of them.

    A subdivided :class:`FacetPanel` grid nests one axes array per cell,
    so a single ``np.ravel`` yields arrays rather than Axes.
    """
    if isinstance(axes, np.ndarray):
        out: list[Any] = []
        for item in axes.ravel().tolist():
            out.extend(_flatten_axes(item))
        return out
    return [axes]


def _require_single_input_panel(panel: InnerPanel) -> None:
    """Reject inner panels the wrappers cannot drive.

    The wrappers hand an inner panel exactly one sliced object and one
    Axes. Panels such as ``EulerianLagrangianPanel`` (eulerian +
    trajectories) and ``EventVerificationPanel`` (four inputs across an
    axes pair) need more of both, and would otherwise fail deep inside
    rendering with a bare ``TypeError``.
    """
    if not isinstance(panel, _ValidationPanel):
        return
    layout = getattr(panel, "_default_axes_layout", (1, 1))
    parameters = list(inspect.signature(panel._build).parameters.values())
    # Drop `fig` and `axes`; what remains is the panel's data arity.
    data_params = [
        param
        for param in parameters[2:]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        and param.default is param.empty
    ]
    if len(data_params) > 1 or tuple(layout) != (1, 1):
        names = [param.name for param in data_params]
        raise TypeError(
            f"{type(panel).__name__} cannot be wrapped: the composable "
            "wrappers render one sliced input into one Axes, but this panel "
            f"takes {len(data_params)} inputs {names} across a "
            f"{tuple(layout)} axes layout. Only single-input, single-axes "
            "panels compose."
        )


def _render_into(panel: InnerPanel, fig: mpl_figure.Figure, ax: Any, ds: Any) -> Any:
    """Render ``ds`` into ``ax`` using whichever inner-panel flavour."""
    if isinstance(panel, _ValidationPanel):
        return panel._build(fig, ax, ds)
    return panel(ds, ax)


def _find_mappable(ax: Any, returned: Any = None) -> Any | None:
    """Recover a colorbar-able mappable produced by an inner panel.

    Prefers whatever ``_build`` returned, so a panel can name its own
    mappable explicitly. None of the shipped panels return one today —
    they call ``fig.colorbar`` internally — so fall back to scavenging
    the most recent artist off the axes, which is what actually makes
    ``sharebar`` work with the current panel suite.
    """
    if isinstance(returned, mpl_cm.ScalarMappable):
        return returned
    for artists in (getattr(ax, "collections", []), getattr(ax, "images", [])):
        for artist in reversed(list(artists)):
            if isinstance(artist, mpl_cm.ScalarMappable):
                return artist
    return None


def _drop_axes_added_since(fig: mpl_figure.Figure, keep: frozenset[Any]) -> None:
    """Remove figure axes created since ``keep`` was snapshotted.

    Panels such as :class:`SpatialMapPanel` call ``fig.colorbar`` inside
    ``_build``, which appends a new Axes to the figure every single call.
    Re-rendering into the same axes — every animation frame, or every
    cell when a shared colorbar supersedes the per-cell ones — would
    otherwise pile up Axes without bound.
    """
    for axis in [a for a in fig.axes if a not in keep]:
        axis.remove()


@contextlib.contextmanager
def _temporary_attrs(obj: Any, overrides: dict[str, Any] | None) -> Iterator[None]:
    """Temporarily set attributes on ``obj``, restoring them after.

    Used instead of the clone-via-``get_config`` round-trip the design
    sketches proposed: several panels stringify un-reconstructable state
    in ``get_config`` (``SpatialMapPanel`` emits ``repr(crs)`` for a
    cartopy instance), so rebuilding through ``__init__`` would fail on
    exactly the panels most likely to be wrapped.
    """
    if not overrides:
        yield
        return
    missing = object()
    previous = {name: getattr(obj, name, missing) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, value)


def _inner_cell_grid(panel: InnerPanel) -> tuple[int, int] | None:
    """Return the ``(nrows, ncols)`` a panel needs, if more than one Axes.

    Most panels draw into a single Axes and report ``None``. Wrappers
    that render several — :class:`PairwiseComparePanel`'s ref/study/diff
    triptych — advertise their shape through ``_cell_grid`` so an outer
    grid can hand them a subdivided cell instead of a bare Axes.
    """
    cell_grid = getattr(panel, "_cell_grid", None)
    if cell_grid is None:
        return None
    shape = cell_grid()
    return None if shape == (1, 1) else shape


def _build_axes_for(
    panel: InnerPanel,
    ds: Any,
    *,
    figsize: tuple[float, float] | None,
    subplot_kw: dict[str, Any] | None,
) -> tuple[mpl_figure.Figure, Any]:
    """Create the figure and axes ``panel`` needs to render ``ds``.

    Panels whose layout depends on the data — :class:`FacetPanel` sizes
    its grid from the faceted dimension — expose ``_make_fig_axes_for``.
    That hook is what lets a wrapper host another wrapper, e.g.
    ``AnimatePanel(FacetPanel(...))``, where the axes must exist before
    the first frame is drawn.

    ``figsize=None`` lets the panel size itself, which is what composite
    inners want — they compute a total from ``figsize_per_panel``. An
    explicit size is stamped onto the finished figure instead of being
    poked into a ``figsize`` attribute the composites never read.
    """
    import matplotlib.pyplot as plt

    if isinstance(panel, _ValidationPanel):
        builder = getattr(panel, "_make_fig_axes_for", None)
        overrides: dict[str, Any] = {}
        if figsize is not None:
            overrides["figsize"] = figsize
        if subplot_kw is not None:
            # Push the override *through* the composite rather than around
            # it: bypassing its builder would yield a single Axes, and its
            # `_build` would then index cells that do not exist.
            overrides["subplot_kw"] = subplot_kw
        with _temporary_attrs(panel, overrides or None):
            fig, axes = builder(ds) if builder is not None else panel._make_fig_axes()
    else:
        resolved = _resolve_subplot_kw(panel, subplot_kw)
        fig, axes = plt.subplots(figsize=figsize or (8, 6), subplot_kw=resolved or None)
        _apply_preset_extent(panel, axes)
    if figsize is not None:
        fig.set_size_inches(*figsize)
    return fig, axes


def _clear_axes(axes: Any) -> None:
    """Clear one Axes or a whole array of them."""
    for ax in np.ravel(np.asarray(axes, dtype=object)).tolist():
        ax.clear()


def _inner_config(panel: InnerPanel) -> dict[str, Any]:
    """Describe the inner panel for a wrapper's ``get_config``.

    Introspection only. A wrapper holds a *live* panel object, and no
    wrapper constructor accepts this dict back — ``FacetPanel(**config)``
    would pass the class-name string where a renderer belongs. The
    ``panel_kind`` key says which flavour was wrapped so callers can tell
    a named panel from an anonymous callable without pretending the
    representation replays.
    """
    if isinstance(panel, _ValidationPanel):
        return {
            "panel": type(panel).__name__,
            "panel_kind": "validation_panel",
            "panel_config": panel.get_config(),
        }
    return {
        "panel": getattr(panel, "__name__", "<callable>"),
        "panel_kind": "callable",
        "panel_config": None,
    }


__all__ = [
    "InnerPanel",
    "_apply_preset_extent",
    "_build_axes_for",
    "_clear_axes",
    "_drop_axes_added_since",
    "_find_mappable",
    "_flatten_axes",
    "_inner_cell_grid",
    "_inner_config",
    "_innermost_projection",
    "_render_into",
    "_require_single_input_panel",
    "_resolve_subplot_kw",
    "_temporary_attrs",
]
