"""Two-input ``(pred, ref, pred − ref)`` snapshot comparison.

:class:`SpatialDiffPanel` is the canonical three-up "prediction /
reference / difference" figure from twin and hindcast comparisons. Unlike
:class:`PairwiseComparePanel`, which takes one input stacked along a
method dim and lets each cell scale itself, it takes the two fields as
separate inputs and fixes the colour scales: one shared scale on the
prediction and reference cells, and a zero-centred divergent scale on the
difference.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz._src.norm import shared_norm
from xrtoolz.viz.validation._src.base import _ValidationPanel
from xrtoolz.viz.validation._src.composition import (
    _apply_preset_extent,
    _inner_config,
    _render_into,
    _require_single_input_panel,
    _resolve_subplot_kw,
    _temporary_attrs,
)
from xrtoolz.viz.validation._src.spatial import SpatialMapPanel


def _scale_overrides(
    panel: _ValidationPanel, cmap: str | None, limits: tuple[float, float] | None
) -> dict[str, Any]:
    """Attribute overrides that pin ``panel``'s colormap and colour limits.

    Map panels spell their limits differently: :class:`CartesianMapPanel`
    takes a ``norm`` tuple, :class:`SpatialMapPanel` takes ``vmin`` /
    ``vmax``.
    """
    out: dict[str, Any] = {} if cmap is None else {"cmap": cmap}
    if hasattr(panel, "norm"):
        out["norm"] = limits
    else:
        out["vmin"], out["vmax"] = limits if limits is not None else (None, None)
    return out


class SpatialDiffPanel(_ValidationPanel):
    """Three-axis ``(pred, ref, pred − ref)`` snapshot panel.

    Called with two inputs, ``panel(pred, ref)``, each a Dataset or
    DataArray on the same grid.

    Args:
        panel: Map panel used to draw each cell. ``None`` (default) uses
            ``SpatialMapPanel(variable=variable)``; pass a configured
            :class:`SpatialMapPanel` (e.g. with a ``projection``) or a
            :class:`CartesianMapPanel` for model output in metres.
        variable: Data-variable name when the inputs are Datasets.
            ``None`` picks the first ``data_var``.
        time_index: When the inputs have a ``time`` dim, ``isel`` this
            index before plotting. ``None`` skips the selection.
        field_cmap: Colormap for the prediction / reference cells.
            ``None`` keeps the inner panel's own (registry) colormap.
        diff_cmap: Colormap for the difference cell. Default ``"RdBu_r"``.
        vmin: Lower colour limit shared by prediction and reference.
            ``None`` uses the 2/98 % quantiles of both fields together.
        vmax: Upper colour limit, as ``vmin``.
        diff_symmetric: Centre the difference scale on zero at its
            2/98 % quantiles. ``False`` lets the inner panel scale it.
            Default ``True``.
        names: Cell titles for the two inputs. Default
            ``("prediction", "reference")``.
        figsize_per_panel: Per-cell ``(width, height)`` in inches.

    Returns:
        A Figure with three map cells.

    Raises:
        ValueError: The two inputs are not on the same grid.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import SpatialDiffPanel
        >>> coords = {"lat": np.linspace(30, 40, 6), "lon": np.linspace(-70, -60, 8)}
        >>> ref = xr.DataArray(np.ones((6, 8)), dims=("lat", "lon"), coords=coords)
        >>> fig = SpatialDiffPanel()(ref + 0.1, ref)
        >>> [ax.get_title() for ax in fig.axes[:3]]
        ['prediction', 'reference', 'prediction - reference']

    """

    def __init__(
        self,
        panel: _ValidationPanel | None = None,
        *,
        variable: str | None = None,
        time_index: int | None = 0,
        field_cmap: str | None = None,
        diff_cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        diff_symmetric: bool = True,
        names: tuple[str, str] = ("prediction", "reference"),
        figsize_per_panel: tuple[float, float] = (5, 4),
        **kw: Any,
    ) -> None:
        inner = panel if panel is not None else SpatialMapPanel(variable=variable)
        _require_single_input_panel(inner)
        super().__init__(**kw)
        self.panel = inner
        self.variable = variable
        self.time_index = time_index
        self.field_cmap = field_cmap
        self.diff_cmap = diff_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.diff_symmetric = bool(diff_symmetric)
        self.names = (str(names[0]), str(names[1]))
        self.figsize_per_panel = tuple(figsize_per_panel)

    def _default_title(self) -> str:
        return ""

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        import matplotlib.pyplot as plt

        width, height = self.figsize_per_panel
        fig, axes = plt.subplots(
            1,
            3,
            sharex=True,
            sharey=True,
            figsize=(3 * width, height),
            subplot_kw=_resolve_subplot_kw(self.panel, None) or None,
            squeeze=False,
        )
        _apply_preset_extent(self.panel, axes)
        return fig, axes

    def _field(self, obj: xr.Dataset | xr.DataArray) -> xr.DataArray:
        if isinstance(obj, xr.Dataset):
            obj = obj[self.variable or next(iter(obj.data_vars))]
        if self.time_index is not None and "time" in obj.dims:
            obj = obj.isel(time=self.time_index)
        return obj

    def _field_limits(
        self, pred: xr.DataArray, ref: xr.DataArray
    ) -> tuple[float, float] | None:
        lo, hi = shared_norm(pred, ref)
        lo = lo if self.vmin is None else self.vmin
        hi = hi if self.vmax is None else self.vmax
        return None if np.isnan(lo) or np.isnan(hi) else (float(lo), float(hi))

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        pred: xr.Dataset | xr.DataArray,
        ref: xr.Dataset | xr.DataArray,
    ) -> None:
        try:
            p, r = xr.align(self._field(pred), self._field(ref), join="exact")
        except ValueError as exc:
            raise ValueError(
                "SpatialDiffPanel needs pred and ref on the same grid; regrid "
                f"one onto the other first ({exc})."
            ) from exc
        diff = p - r
        ax_pred, ax_ref, ax_diff = np.ravel(axes)

        field = _scale_overrides(self.panel, self.field_cmap, self._field_limits(p, r))
        with _temporary_attrs(self.panel, field):
            for ax, cell, name in (
                (ax_pred, p, self.names[0]),
                (ax_ref, r, self.names[1]),
            ):
                _render_into(self.panel, fig, ax, cell)
                ax.set_title(name)

        diff_limits = None
        if self.diff_symmetric:
            lo, hi = shared_norm(diff, symmetric=True)
            diff_limits = None if np.isnan(hi) else (lo, hi)
        with _temporary_attrs(
            self.panel, _scale_overrides(self.panel, self.diff_cmap, diff_limits)
        ):
            _render_into(self.panel, fig, ax_diff, diff)
        ax_diff.set_title(f"{self.names[0]} - {self.names[1]}")

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **_inner_config(self.panel),
            "variable": self.variable,
            "time_index": self.time_index,
            "field_cmap": self.field_cmap,
            "diff_cmap": self.diff_cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "diff_symmetric": self.diff_symmetric,
            "names": list(self.names),
            "figsize_per_panel": list(self.figsize_per_panel),
        }


__all__ = ["SpatialDiffPanel"]
