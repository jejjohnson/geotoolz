"""ODC-2.3 — generic N-up faceting over any single-axes panel.

:class:`FacetPanel` turns any existing panel (or plain
``(ds, ax) -> Any`` callable) into a grid, one cell per value along a
categorical dimension. The upstream "seasonal PSD-score mosaic" is not a
distinct panel type under this scheme, just
``FacetPanel(PSDSpaceTimeScorePanel(...), facet_dim="season")`` — and the
same machinery serves ``experiment``, ``method``, ``region`` or
``lead_time`` without further code.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _NullContext, _ValidationPanel
from xrtoolz.viz.validation._src.composition import (
    InnerPanel,
    _apply_preset_extent,
    _drop_axes_added_since,
    _find_mappable,
    _flatten_axes,
    _inner_cell_grid,
    _inner_config,
    _render_into,
    _require_single_input_panel,
    _resolve_subplot_kw,
)


_SEASON_ORDER: tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")


class FacetPanel(_ValidationPanel):
    """Render any single-axes panel faceted across a categorical dim.

    Args:
        panel: The panel to repeat. Either a ``_ValidationPanel``
            subclass or a callable ``(ds, ax) -> Any``. Panels that own
            the whole figure (seaborn ``JointGrid``, ``pairplot``) and
            non-matplotlib backends are out of scope.
        facet_dim: Dimension to facet over. One cell per value.
        ncols: Columns in the grid. ``None`` derives a near-square
            layout, ``ceil(sqrt(n))``.
        nrows: Rows in the grid. ``None`` derives it from ``ncols``.
        sharex: Share the x axis across cells. Default ``True``.
        sharey: Share the y axis across cells. Default ``True``.
        sharebar: Replace the inner panel's per-cell colorbars with a
            single shared one spanning the grid. Requires that the inner
            panel produce a mappable; when none is found a warning is
            emitted and per-cell behaviour is left untouched. Default
            ``False``.
        figsize_per_panel: Per-cell ``(width, height)`` in inches, so the
            figure grows with the facet count (xarray ``FacetGrid``
            convention). Default ``(5, 4)``.
        title_format: Per-cell title template, formatted with ``value``
            (the coord value) and ``index``. Default ``"{value}"``.
        subplot_kw: Forwarded to :func:`matplotlib.pyplot.subplots`,
            overriding the projection inferred from the inner panel. Use
            it to pick a *different* projection: stripping one from a
            panel that draws in projected coordinates (``{}`` against a
            ``SpatialMapPanel(projection=...)``) leaves the panel
            transforming onto axes that cannot accept it, and cartopy
            raises.

    Returns:
        A Figure with one cell per ``facet_dim`` value; trailing unused
        cells are hidden.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import FacetPanel
        >>> ds = xr.DataArray(
        ...     np.arange(12.0).reshape(4, 3),
        ...     dims=("run", "x"),
        ...     coords={"run": ["a", "b", "c", "d"]},
        ... ).to_dataset(name="score")
        >>> def line(sub, ax):
        ...     ax.plot(sub["score"].values)
        >>> fig = FacetPanel(line, facet_dim="run").__call__(ds)
        >>> len(fig.axes)
        4

    """

    def __init__(
        self,
        panel: InnerPanel,
        *,
        facet_dim: str,
        ncols: int | None = None,
        nrows: int | None = None,
        sharex: bool = True,
        sharey: bool = True,
        sharebar: bool = False,
        figsize_per_panel: tuple[float, float] = (5, 4),
        title_format: str = "{value}",
        subplot_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        _require_single_input_panel(panel)
        super().__init__(**kw)
        self.panel = panel
        self.facet_dim = facet_dim
        self.ncols = ncols
        self.nrows = nrows
        self.sharex = bool(sharex)
        self.sharey = bool(sharey)
        self.sharebar = bool(sharebar)
        self.figsize_per_panel = tuple(figsize_per_panel)
        self.title_format = title_format
        self.subplot_kw = subplot_kw

    def _default_title(self) -> str:
        return ""

    def _layout(self, n: int) -> tuple[int, int]:
        """Resolve ``(nrows, ncols)`` for ``n`` cells."""
        if n == 0:
            # ceil(sqrt(0)) is 0, and the division below would then raise
            # ZeroDivisionError. An empty dim is a legitimate result of
            # selection, so say what happened.
            raise ValueError(
                f"facet_dim {self.facet_dim!r} is empty; there is nothing to "
                "facet over."
            )
        if self.nrows is not None and self.ncols is not None:
            nrows, ncols = int(self.nrows), int(self.ncols)
            if nrows * ncols < n:
                raise ValueError(
                    f"nrows={nrows} x ncols={ncols} cannot hold {n} facets of "
                    f"{self.facet_dim!r}."
                )
            return nrows, ncols
        if self.ncols is not None:
            ncols = int(self.ncols)
            return math.ceil(n / ncols), ncols
        if self.nrows is not None:
            nrows = int(self.nrows)
            return nrows, math.ceil(n / nrows)
        ncols = math.ceil(math.sqrt(n))
        return math.ceil(n / ncols), ncols

    def _make_fig_axes_for(
        self, ds: xr.Dataset | xr.DataArray
    ) -> tuple[mpl_figure.Figure, Any]:
        """Build the grid sized for ``ds``'s facet dimension.

        The hook wrappers use when they must create axes before seeing
        data — see :func:`composition._build_axes_for`.
        """
        return self._grid(int(ds.sizes[self.facet_dim]))

    def _grid(self, n: int) -> tuple[mpl_figure.Figure, Any]:
        import matplotlib.pyplot as plt

        nrows, ncols = self._layout(n)
        width, height = self.figsize_per_panel
        subplot_kw = _resolve_subplot_kw(self.panel, self.subplot_kw) or None
        inner_shape = _inner_cell_grid(self.panel)

        if inner_shape is None:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=(ncols * width, nrows * height),
                subplot_kw=subplot_kw,
                squeeze=False,
            )
            _apply_preset_extent(self.panel, axes)
            return fig, axes

        # The inner panel wants several axes per facet, so each outer cell
        # becomes its own sub-grid. `axes` is then an object array whose
        # entries are per-cell axes arrays, which `_render_into` forwards
        # to the inner panel untouched.
        inner_rows, inner_cols = inner_shape
        fig = plt.figure(
            figsize=(ncols * width * inner_cols, nrows * height * inner_rows)
        )
        outer = fig.add_gridspec(nrows, ncols)
        cells = np.empty((nrows, ncols), dtype=object)
        # Axis sharing has to be wired by hand here: `add_subplot` cannot
        # join groups after the fact the way `plt.subplots(sharex=...)`
        # does. Cartopy GeoAxes reject shared axes outright, so projected
        # grids keep independent limits.
        inner_sharex = bool(getattr(self.panel, "sharex", True))
        inner_sharey = bool(getattr(self.panel, "sharey", True))
        shareable = not (subplot_kw or {}).get("projection")
        anchor: Any = None
        for row in range(nrows):
            for col in range(ncols):
                sub = outer[row, col].subgridspec(inner_rows, inner_cols)
                axes_in_cell: list[Any] = []
                lead: Any = None
                for r in range(inner_rows):
                    for c in range(inner_cols):
                        share: dict[str, Any] = {}
                        if shareable:
                            # Within the cell, follow the inner panel's own
                            # sharing; across cells, follow this panel's.
                            if lead is not None:
                                if inner_sharex:
                                    share["sharex"] = lead
                                if inner_sharey:
                                    share["sharey"] = lead
                            elif anchor is not None:
                                if self.sharex:
                                    share["sharex"] = anchor
                                if self.sharey:
                                    share["sharey"] = anchor
                        ax = fig.add_subplot(sub[r, c], **(subplot_kw or {}), **share)
                        if lead is None:
                            lead = ax
                        if anchor is None:
                            anchor = ax
                        axes_in_cell.append(ax)
                cells[row, col] = np.array(axes_in_cell, dtype=object).reshape(
                    inner_rows, inner_cols
                )
        _apply_preset_extent(self.panel, cells)
        return fig, cells

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, ds: xr.Dataset | xr.DataArray
    ) -> None:
        if self.facet_dim not in ds.dims:
            raise ValueError(
                f"facet_dim {self.facet_dim!r} is not a dimension of the input; "
                f"got dims {tuple(ds.dims)}."
            )
        n = int(ds.sizes[self.facet_dim])
        flat = list(np.ravel(axes))
        values = (
            np.asarray(ds[self.facet_dim].values)
            if self.facet_dim in ds.coords
            else np.arange(n)
        )

        mappables: list[Any] = []
        # Snapshot before any cell renders: reclaiming the panel's own
        # colorbars is only correct once a shared bar is guaranteed, so
        # the removal is deferred until after the loop.
        baseline = frozenset(fig.axes)
        for index in range(n):
            ax = flat[index]
            slice_ = ds.isel({self.facet_dim: index})
            returned = _render_into(self.panel, fig, ax, slice_)
            label = self.title_format.format(value=values[index], index=index)
            if self.sharebar:
                for sub_ax in _flatten_axes(ax):
                    found = _find_mappable(sub_ax, returned)
                    if found is not None:
                        mappables.append(found)
            if isinstance(ax, np.ndarray):
                # A subdivided cell: the inner panel titled each of its own
                # axes, so prefix the facet label onto the first rather than
                # overwriting what it wrote.
                lead = _flatten_axes(ax)[0]
                existing = lead.get_title()
                lead.set_title(f"{label}\n{existing}" if existing else label)
            else:
                ax.set_title(label)

        for ax in flat[n:]:
            for sub in _flatten_axes(ax):
                sub.set_visible(False)

        if self.sharebar:
            if not mappables:
                warnings.warn(
                    "sharebar=True but the inner panel produced no mappable; "
                    "falling back to whatever colorbars the panel draws itself.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # Every facet autoscales independently unless the inner panel
                # pins vmin/vmax, so one bar drawn from one mappable would
                # misdescribe the others. Put them all on a common scale.
                limits = [m.get_clim() for m in mappables]
                lows = [low for low, _ in limits if low is not None]
                highs = [high for _, high in limits if high is not None]
                if lows and highs:
                    shared = (min(lows), max(highs))
                    for m in mappables:
                        m.set_clim(*shared)
                _drop_axes_added_since(fig, baseline)
                fig.colorbar(
                    mappables[0],
                    ax=[sub for cell in flat[:n] for sub in _flatten_axes(cell)],
                )

    def _apply(self, *args: Any, **kwargs: Any) -> mpl_figure.Figure:
        # Overridden rather than using `_make_fig_axes`: the grid shape
        # depends on the data, which the base hook never sees.
        import matplotlib.pyplot as plt

        # Operator.__call__ forwards eager keyword arguments straight to
        # `_apply`, so `panel(ds=...)` must work as it does for the
        # sibling panels that take their data positionally.
        if args:
            ds = args[0]
        elif "ds" in kwargs:
            ds = kwargs.pop("ds")
            args = (ds,)
        else:
            raise TypeError(
                f"{type(self).__name__} requires an input dataset, given "
                "positionally or as `ds=`."
            )
        if self.facet_dim not in ds.dims:
            raise ValueError(
                f"facet_dim {self.facet_dim!r} is not a dimension of the input; "
                f"got dims {tuple(ds.dims)}."
            )
        ctx = (
            plt.style.context(self.style) if self.style is not None else _NullContext()
        )
        with ctx:
            fig, axes = self._make_fig_axes_for(ds)
            self._build(fig, axes, *args, **kwargs)
            title = self.title if self.title is not None else self._default_title()
            with warnings.catch_warnings():
                # Colorbar / cartopy axes are legitimately not tight_layout
                # compatible; the layout is still the best available.
                warnings.filterwarnings("ignore", message=".*tight_layout.*")
                fig.tight_layout()
            if title:
                fig.suptitle(title)
        self._maybe_save(fig)
        self._maybe_show(fig)
        return fig

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **_inner_config(self.panel),
            "facet_dim": self.facet_dim,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "sharex": self.sharex,
            "sharey": self.sharey,
            "sharebar": self.sharebar,
            "figsize_per_panel": list(self.figsize_per_panel),
            "title_format": self.title_format,
        }


def seasonal_groupby(
    ds: xr.Dataset | xr.DataArray,
    *,
    time: str = "time",
    reduction: str = "mean",
) -> xr.Dataset | xr.DataArray:
    """Reduce a continuous-time object to a four-cell ``season`` dim.

    A one-liner over ``ds.groupby(f"{time}.season")`` so the seasonal
    mosaic is a straight feed into ``FacetPanel(facet_dim="season")``.

    Args:
        ds: Input carrying a datetime-like ``time`` coordinate.
        time: Name of that coordinate. Default ``"time"``.
        reduction: Any groupby reduction name — ``"mean"``, ``"sum"``,
            ``"median"``, … Default ``"mean"``.

    Returns:
        The reduced object with a ``season`` dimension holding
        ``DJF``/``MAM``/``JJA``/``SON`` in chronological order. Grouping
        on the string ``time.season`` coordinate sorts lexicographically
        (``DJF``, ``JJA``, ``MAM``, ``SON``), which would put summer
        before spring in a mosaic, so the result is reindexed.

    Raises:
        ValueError: If ``reduction`` is not a groupby method.

    Example:
        >>> import numpy as np, pandas as pd, xarray as xr
        >>> from xrtoolz.viz.validation import seasonal_groupby
        >>> ds = xr.DataArray(
        ...     np.arange(365.0),
        ...     dims="time",
        ...     coords={"time": pd.date_range("2020-01-01", periods=365)},
        ... ).to_dataset(name="sst")
        >>> int(seasonal_groupby(ds).sizes["season"])
        4

    """
    grouped = ds.groupby(f"{time}.season")
    if not hasattr(grouped, reduction):
        raise ValueError(
            f"unknown reduction {reduction!r}; expected a groupby method such "
            "as 'mean', 'sum' or 'median'."
        )
    reduced = getattr(grouped, reduction)()
    present = [s for s in _SEASON_ORDER if s in set(reduced["season"].values)]
    return reduced.reindex(season=present)


__all__ = ["FacetPanel", "seasonal_groupby"]
