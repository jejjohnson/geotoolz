"""ODC-3.4 — paired study-vs-reference comparison with a diff cell.

:class:`PairwiseComparePanel` renders the ``(ref, study, diff)`` triptych
that every intercomparison figure rebuilds by hand. Composed under
:class:`~xrtoolz.viz.validation.FacetPanel` it reproduces the upstream
six-panel "scale x (ref, study, diff)" mosaics for free.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel
from xrtoolz.viz.validation._src.composition import (
    InnerPanel,
    _apply_preset_extent,
    _inner_config,
    _render_into,
    _require_single_input_panel,
    _resolve_subplot_kw,
    _temporary_attrs,
)


DiffMode = Literal["relative", "absolute", "none"]

_DIFF_LABELS: dict[str, str] = {"absolute": "Δ", "relative": "Δ%"}


class PairwiseComparePanel(_ValidationPanel):
    """Side-by-side study vs reference, with an optional diff cell.

    The input carries ``method_dim`` of size 2: position 0 is the
    reference / baseline, position 1 the study / candidate.

    Args:
        panel: Inner panel to render each cell with — a
            ``_ValidationPanel`` or a ``(ds, ax) -> Any`` callable, the
            same boundary :class:`FacetPanel` accepts.
        method_dim: Dimension holding the two methods. Default
            ``"method"``.
        diff: ``"relative"`` renders ``100 * (study - ref) / ref``,
            ``"absolute"`` renders ``study - ref``, ``"none"`` omits the
            third cell entirely. Default ``"relative"``.
        diff_kwargs: Attribute overrides applied to the inner panel for
            the diff cell only — typically ``{"cmap": "coolwarm",
            "vmin": -20, "vmax": 20}``, since a signed difference wants a
            divergent scale. Ignored with a warning for callable inners,
            which have no attributes to override.
        diff_label: Title for the diff cell. ``None`` uses ``"Δ"`` or
            ``"Δ%"`` per ``diff``.
        layout: ``"row"`` lays the cells out horizontally, ``"col"``
            vertically. Default ``"row"``.
        sharex: Share the x axis across cells. Default ``True``.
        sharey: Share the y axis across cells. Default ``True``.
        figsize_per_panel: Per-cell ``(width, height)`` in inches.
        subplot_kw: Forwarded to :func:`matplotlib.pyplot.subplots`,
            overriding the inner panel's projection.

    Returns:
        A Figure of three cells (two when ``diff="none"``).

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import PairwiseComparePanel
        >>> ds = xr.DataArray(
        ...     np.arange(8.0).reshape(2, 4) + 1.0,
        ...     dims=("method", "x"),
        ...     coords={"method": ["duacs", "miost"]},
        ... ).to_dataset(name="rmse")
        >>> def line(sub, ax):
        ...     ax.plot(sub["rmse"].values)
        >>> fig = PairwiseComparePanel(line).__call__(ds)
        >>> len(fig.axes)
        3

    """

    def __init__(
        self,
        panel: InnerPanel,
        *,
        method_dim: str = "method",
        diff: DiffMode = "relative",
        diff_kwargs: dict[str, Any] | None = None,
        diff_label: str | None = None,
        layout: Literal["row", "col"] = "row",
        sharex: bool = True,
        sharey: bool = True,
        figsize_per_panel: tuple[float, float] = (5, 4),
        subplot_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        if diff not in ("relative", "absolute", "none"):
            raise ValueError(
                f"unknown diff mode {diff!r}; expected 'relative', 'absolute' "
                "or 'none'."
            )
        if layout not in ("row", "col"):
            raise ValueError(f"unknown layout {layout!r}; expected 'row' or 'col'.")
        _require_single_input_panel(panel)
        super().__init__(**kw)
        self.panel = panel
        self.method_dim = method_dim
        self.diff = diff
        self.diff_kwargs = dict(diff_kwargs) if diff_kwargs else None
        self.diff_label = diff_label
        self.layout = layout
        self.sharex = bool(sharex)
        self.sharey = bool(sharey)
        self.figsize_per_panel = tuple(figsize_per_panel)
        self.subplot_kw = subplot_kw

    def _default_title(self) -> str:
        return ""

    @property
    def _n_cells(self) -> int:
        return 2 if self.diff == "none" else 3

    def _cell_grid(self) -> tuple[int, int]:
        """Axes shape this panel needs — see ``composition._inner_cell_grid``.

        Lets an outer :class:`FacetPanel` subdivide each facet cell into
        a triptych rather than handing over a single Axes.
        """
        n = self._n_cells
        return (1, n) if self.layout == "row" else (n, 1)

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        import matplotlib.pyplot as plt

        n = self._n_cells
        width, height = self.figsize_per_panel
        nrows, ncols = (1, n) if self.layout == "row" else (n, 1)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex=self.sharex,
            sharey=self.sharey,
            figsize=(ncols * width, nrows * height),
            subplot_kw=_resolve_subplot_kw(self.panel, self.subplot_kw) or None,
            squeeze=False,
        )
        _apply_preset_extent(self.panel, axes)
        return fig, axes

    def _compute_diff(self, study: Any, ref: Any) -> Any:
        if self.diff == "absolute":
            return study - ref
        # Guard the zero-denominator case rather than emitting inf: a
        # reference of exactly 0 has no meaningful percentage change.
        return xr.where(ref != 0, 100.0 * (study - ref) / ref, np.nan)

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, ds: xr.Dataset | xr.DataArray
    ) -> None:
        if self.method_dim not in ds.dims:
            raise ValueError(
                f"method_dim {self.method_dim!r} is not a dimension of the "
                f"input; got dims {tuple(ds.dims)}."
            )
        size = int(ds.sizes[self.method_dim])
        if size != 2:
            raise ValueError(
                f"PairwiseComparePanel needs exactly 2 entries along "
                f"{self.method_dim!r} (position 0 = reference, 1 = study); "
                f"got {size}."
            )

        flat = list(np.ravel(axes))
        ref = ds.isel({self.method_dim: 0})
        study = ds.isel({self.method_dim: 1})
        if self.method_dim in ds.coords:
            names = [str(v) for v in np.asarray(ds[self.method_dim].values)]
        else:
            names = ["reference", "study"]

        for ax, cell, name in zip(flat, (ref, study), names, strict=False):
            _render_into(self.panel, fig, ax, cell)
            ax.set_title(name)

        if self.diff == "none":
            return

        if self.diff_kwargs and not isinstance(self.panel, _ValidationPanel):
            warnings.warn(
                "diff_kwargs is ignored for callable inner panels — there are "
                "no attributes to override.",
                UserWarning,
                stacklevel=2,
            )
            overrides = None
        else:
            overrides = self.diff_kwargs

        with _temporary_attrs(self.panel, overrides):
            _render_into(self.panel, fig, flat[2], self._compute_diff(study, ref))
        label = self.diff_label or _DIFF_LABELS[self.diff]
        flat[2].set_title(f"{names[0]} → {names[1]}: {label}")

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **_inner_config(self.panel),
            "method_dim": self.method_dim,
            "diff": self.diff,
            "diff_kwargs": self.diff_kwargs,
            "diff_label": self.diff_label,
            "layout": self.layout,
            "sharex": self.sharex,
            "sharey": self.sharey,
            "figsize_per_panel": list(self.figsize_per_panel),
        }


__all__ = ["DiffMode", "PairwiseComparePanel"]
