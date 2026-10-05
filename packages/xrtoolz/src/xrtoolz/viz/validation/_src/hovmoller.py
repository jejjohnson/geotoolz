"""Hovmöller diagram — time × one spatial axis.

:class:`HovmollerPanel` averages a field over every dim except time and
one kept spatial axis, then draws the ``(time, keep_dim)`` section, so
propagation speeds, persistent anomalies and seasonal cycles read
straight off the figure.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz._src.cmaps import cmap_for
from xrtoolz.viz.validation._src.base import _ValidationPanel


class HovmollerPanel(_ValidationPanel):
    """Time × spatial-axis cross-section of a field.

    Every dim other than ``time_dim`` and ``keep_dim`` is averaged out
    (NaN-skipping, unweighted — area-weight beforehand if averaging over
    latitude matters).

    Args:
        variable: Data-variable name when input is a Dataset. ``None``
            (default) auto-picks the first ``data_var``.
        time_dim: Time dimension. Default ``"time"``.
        keep_dim: Spatial dimension to keep, e.g. ``"lat"`` (average over
            ``lon``), ``"lon"``, or ``"x"`` / ``"y"`` for Cartesian model
            output. Default ``"lat"``.
        time_axis: Put time on the ``"x"`` (default) or ``"y"`` axis.
        cmap: Matplotlib colormap. ``None`` (default) resolves from the
            variable registry, falling back to ``"RdBu_r"``.
        norm: ``"linear"`` (default) or ``"log"``. The log scale masks
            non-positive values.
        vmin: Optional lower colour limit.
        vmax: Optional upper colour limit.

    Returns:
        A :class:`matplotlib.figure.Figure` with one section plot.

    Raises:
        ValueError: An unknown ``norm`` / ``time_axis``, or the input lacks
            ``time_dim`` or ``keep_dim``.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import HovmollerPanel
        >>> t, lon = np.arange(30.0), np.linspace(0.0, 350.0, 36)
        >>> wave = np.sin(np.deg2rad(lon[None, :] - 5.0 * t[:, None]))
        >>> da = xr.DataArray(
        ...     wave[:, None, :] * np.ones((1, 4, 1)),
        ...     dims=("time", "lat", "lon"),
        ...     coords={"time": t, "lat": np.arange(4.0), "lon": lon},
        ... )
        >>> fig = HovmollerPanel(keep_dim="lon", time_axis="y")(da)
        >>> fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()
        ('lon', 'time')

    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        variable: str | None = None,
        time_dim: str = "time",
        keep_dim: str = "lat",
        time_axis: Literal["x", "y"] = "x",
        cmap: str | None = None,
        norm: Literal["linear", "log"] = "linear",
        vmin: float | None = None,
        vmax: float | None = None,
        **kw: Any,
    ) -> None:
        if norm not in ("linear", "log"):
            raise ValueError(f"norm={norm!r} must be 'linear' or 'log'.")
        if time_axis not in ("x", "y"):
            raise ValueError(f"time_axis={time_axis!r} must be 'x' or 'y'.")
        super().__init__(**kw)
        self.variable = variable
        self.time_dim = time_dim
        self.keep_dim = keep_dim
        self.time_axis = time_axis
        self.cmap = cmap
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax

    def _default_title(self) -> str:
        return self.variable or "Hovmöller"

    def _section(self, obj: xr.DataArray | xr.Dataset) -> xr.DataArray:
        if isinstance(obj, xr.Dataset):
            obj = obj[self.variable or next(iter(obj.data_vars))]
        missing = [d for d in (self.time_dim, self.keep_dim) if d not in obj.dims]
        if missing:
            raise ValueError(
                f"HovmollerPanel input is missing dim(s) {missing}; got {obj.dims}."
            )
        other = [d for d in obj.dims if d not in (self.time_dim, self.keep_dim)]
        if other:
            obj = obj.mean(dim=other, skipna=True)
        return obj

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        field: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        section = self._section(field)
        # pcolormesh(X, Y, C) wants C shaped (len(Y), len(X)).
        x_dim, y_dim = (
            (self.time_dim, self.keep_dim)
            if self.time_axis == "x"
            else (self.keep_dim, self.time_dim)
        )
        vals = np.asarray(section.transpose(y_dim, x_dim).values)
        cmap = self.cmap if self.cmap is not None else cmap_for(self.variable, "RdBu_r")
        pcm_kw: dict[str, Any] = {"cmap": cmap, "shading": "auto"}
        if self.norm == "log":
            vals = np.ma.masked_less_equal(vals, 0.0)
            pcm_kw["norm"] = mcolors.LogNorm(vmin=self.vmin, vmax=self.vmax)
        else:
            pcm_kw["vmin"], pcm_kw["vmax"] = self.vmin, self.vmax
        im = ax.pcolormesh(section[x_dim].values, section[y_dim].values, vals, **pcm_kw)
        ax.set_xlabel(x_dim)
        ax.set_ylabel(y_dim)
        units = section.attrs.get("units")
        fig.colorbar(im, ax=ax, label=units or "")

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "variable": self.variable,
            "time_dim": self.time_dim,
            "keep_dim": self.keep_dim,
            "time_axis": self.time_axis,
            "cmap": self.cmap,
            "norm": self.norm,
            "vmin": self.vmin,
            "vmax": self.vmax,
        }


__all__ = ["HovmollerPanel"]
