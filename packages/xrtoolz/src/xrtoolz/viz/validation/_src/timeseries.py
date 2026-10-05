"""Scalar-over-time panels.

:class:`EnergyTimeSeriesPanel` plots domain-integrated model invariants
(energy, enstrophy, mass, …) against time — the conservation /
drift check every model tutorial otherwise hand-rolls.

:class:`TimeSeriesErrorPanel` plots per-timestep skill scores (RMSE,
nRMSE, correlation, …) for several methods — the "skill curves over
time" figure of every intercomparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel
from xrtoolz.viz.validation._src.palette import method_palette


class EnergyTimeSeriesPanel(_ValidationPanel):
    """One line per scalar invariant over time.

    Args:
        variables: Names of 1-D (``time``) data variables to plot, e.g.
            ``["kinetic_energy", "enstrophy"]``.
        relative: Plot ``X(t) / X(t₀)`` so invariants with different units
            share one axis and drift reads directly off the y-axis.
            Default ``False``.
        log: Log-scale the y-axis. Default ``False``.
        time: Name of the time dimension. Default ``"time"``.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line per variable.

    Raises:
        ValueError: A variable is missing, is not 1-D along ``time``, or
            (with ``relative=True``) starts at zero.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import EnergyTimeSeriesPanel
        >>> t = np.arange(10.0)
        >>> ds = xr.Dataset(
        ...     {"ke": ("time", 2.0 - 0.01 * t), "ens": ("time", 5.0 + 0.1 * t)},
        ...     coords={"time": t},
        ... )
        >>> fig = EnergyTimeSeriesPanel(["ke", "ens"], relative=True)(ds)
        >>> fig.axes[0].lines[0].get_ydata()[0]
        np.float64(1.0)

    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        variables: Sequence[str],
        *,
        relative: bool = False,
        log: bool = False,
        time: str = "time",
        **kw: Any,
    ) -> None:
        if isinstance(variables, str):
            variables = [variables]
        if not variables:
            raise ValueError("EnergyTimeSeriesPanel needs at least one variable.")
        super().__init__(**kw)
        self.variables = list(variables)
        self.relative = bool(relative)
        self.log = bool(log)
        self.time = time

    def _default_title(self) -> str:
        return "Invariants relative to t₀" if self.relative else "Invariants"

    def _series(self, ds: xr.Dataset, name: str) -> xr.DataArray:
        if name not in ds.data_vars:
            raise ValueError(
                f"variable {name!r} not in the input; got {sorted(ds.data_vars)}."
            )
        da = ds[name]
        if da.dims != (self.time,):
            raise ValueError(
                f"{name!r} must be 1-D along {self.time!r}; got dims {da.dims}."
            )
        if not self.relative:
            return da
        first = float(da.isel({self.time: 0}))
        if first == 0.0 or np.isnan(first):
            raise ValueError(
                f"relative=True divides by the first sample, but {name!r} "
                f"starts at {first}."
            )
        return da / first

    def _build(self, fig: mpl_figure.Figure, axes: Any, ds: xr.Dataset) -> None:
        ax = axes
        for name in self.variables:
            series = self._series(ds, name)
            label = ds[name].attrs.get("long_name", name)
            ax.plot(np.asarray(ds[self.time].values), series.values, label=label)
        if self.relative:
            ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
            ax.set_ylabel("X(t) / X(t₀)")
        elif len(self.variables) == 1 and "units" in ds[self.variables[0]].attrs:
            ax.set_ylabel(ds[self.variables[0]].attrs["units"])
        if self.log:
            ax.set_yscale("log")
        ax.set_xlabel(self.time)
        ax.legend()

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "variables": list(self.variables),
            "relative": self.relative,
            "log": self.log,
            "time": self.time,
        }


class TimeSeriesErrorPanel(_ValidationPanel):
    """Scalar skill metric(s) over time, one line per method.

    Two input layouts are accepted:

    - ``method_dim`` set (default ``"method"``): a Dataset whose data
      variables are metrics on ``(method_dim, time_dim)``. Each metric
      gets its own subplot (stacked, shared x) with one line per method.
    - ``method_dim=None``: a Dataset whose data variables *are* the
      methods, each 1-D on ``time_dim``. One axes, one line per variable.

    A DataArray is treated as a single-metric Dataset. Colours come from
    :func:`method_palette`, so a method keeps its colour across figures.

    Args:
        time_dim: Time / step dimension. Default ``"time"``.
        method_dim: Method dimension, or ``None`` for the
            variables-are-methods layout. Default ``"method"``.
        metrics: Subset of data variables to plot (metrics, or methods
            when ``method_dim=None``). ``None`` plots all of them.
        ylabel: Y-axis label for a single-metric figure. Multi-metric
            figures label each subplot with its metric name instead.
            Default ``"Score"``.
        legend_loc: Matplotlib legend location. Default ``"best"``.
        palette: ``None`` (default ``tab:`` cycle), a matplotlib colormap
            name to draw the cycle from, or an explicit
            ``{method: colour}`` mapping.

    Returns:
        A Figure with one axes per metric.

    Raises:
        ValueError: A requested metric is missing, or a series is not
            1-D along ``time_dim`` once ``method_dim`` is accounted for.

    Example:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.viz.validation import TimeSeriesErrorPanel
        >>> ds = xr.Dataset(
        ...     {"rmse": (("method", "time"), np.random.rand(3, 20))},
        ...     coords={
        ...         "method": ["duacs", "miost", "4dvarnet"],
        ...         "time": np.arange(20),
        ...     },
        ... )
        >>> fig = TimeSeriesErrorPanel(ylabel="RMSE [m]")(ds)
        >>> sorted(line.get_label() for line in fig.axes[0].get_lines())
        ['4dvarnet', 'duacs', 'miost']

    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        time_dim: str = "time",
        method_dim: str | None = "method",
        metrics: Sequence[str] | None = None,
        ylabel: str = "Score",
        legend_loc: str = "best",
        palette: str | dict[str, str] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.time_dim = time_dim
        self.method_dim = method_dim
        self.metrics = None if metrics is None else list(metrics)
        self.ylabel = ylabel
        self.legend_loc = legend_loc
        self.palette = dict(palette) if isinstance(palette, dict) else palette

    def _default_title(self) -> str:
        return ""

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        # The subplot count depends on the data, so _build lays them out.
        import matplotlib.pyplot as plt

        return plt.figure(figsize=self.figsize), None

    def _colours(self, names: list[str]) -> dict[str, str]:
        if isinstance(self.palette, dict):
            missing = [n for n in names if n not in self.palette]
            return {**self.palette, **method_palette(missing)}
        if isinstance(self.palette, str):
            import matplotlib as mpl

            cmap = mpl.colormaps[self.palette]
            count = getattr(cmap, "N", len(names))
            cycle = [
                mpl.colors.to_hex(cmap(i / max(count - 1, 1))) for i in range(count)
            ]
            return method_palette(names, cycle)
        return method_palette(names)

    def _selected(self, ds: xr.Dataset) -> list[str]:
        names = list(ds.data_vars) if self.metrics is None else self.metrics
        missing = [n for n in names if n not in ds.data_vars]
        if missing:
            raise ValueError(
                f"metrics {missing} not in the input; got {list(ds.data_vars)}."
            )
        return [str(n) for n in names]

    def _check_1d(self, da: xr.DataArray, label: str) -> None:
        if da.dims != (self.time_dim,):
            raise ValueError(
                f"{label} must be 1-D along {self.time_dim!r}"
                + (f" per {self.method_dim!r}" if self.method_dim else "")
                + f"; got dims {da.dims}."
            )

    def _panels(
        self, ds: xr.Dataset
    ) -> list[tuple[str, list[tuple[str, xr.DataArray]]]]:
        """``[(axes label, [(method, series), ...]), ...]`` — one entry per axes."""
        names = self._selected(ds)
        if self.method_dim is None:
            lines = [(name, ds[name]) for name in names]
            for name, da in lines:
                self._check_1d(da, repr(name))
            return [(self.ylabel, lines)]
        out = []
        for name in names:
            da = ds[name]
            if self.method_dim not in da.dims:
                raise ValueError(
                    f"metric {name!r} has no {self.method_dim!r} dim; got {da.dims}. "
                    "Pass method_dim=None if the variables are the methods."
                )
            if self.method_dim in da.coords:
                methods = [str(m) for m in da[self.method_dim].values]
            else:
                methods = [str(i) for i in range(da.sizes[self.method_dim])]
            lines = []
            for i, method in enumerate(methods):
                series = da.isel({self.method_dim: i}, drop=True)
                self._check_1d(series, f"metric {name!r}")
                lines.append((method, series))
            out.append((name if len(names) > 1 else self.ylabel, lines))
        return out

    def _build(
        self, fig: mpl_figure.Figure, axes: Any, scores: xr.Dataset | xr.DataArray
    ) -> None:
        if isinstance(scores, xr.DataArray):
            scores = scores.to_dataset(name=scores.name or "score")
        panels = self._panels(scores)
        if axes is None:
            axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False).ravel()
        elif len(panels) == 1:
            axes = [axes]
        else:
            raise ValueError(
                f"{len(panels)} metrics need {len(panels)} axes but one was given; "
                "select a single metric with metrics=[...]."
            )
        colours = self._colours(sorted({m for _, lines in panels for m, _ in lines}))
        for ax, (label, lines) in zip(axes, panels, strict=True):
            for method, series in lines:
                ax.plot(
                    np.asarray(series[self.time_dim].values),
                    series.values,
                    label=method,
                    color=colours[method],
                )
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc=self.legend_loc)
        axes[-1].set_xlabel(self.time_dim)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "time_dim": self.time_dim,
            "method_dim": self.method_dim,
            "metrics": self.metrics,
            "ylabel": self.ylabel,
            "legend_loc": self.legend_loc,
            "palette": self.palette,
        }


__all__ = ["EnergyTimeSeriesPanel", "TimeSeriesErrorPanel"]
