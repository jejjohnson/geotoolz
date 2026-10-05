"""Scalar-over-time panels.

:class:`EnergyTimeSeriesPanel` plots domain-integrated model invariants
(energy, enstrophy, mass, …) against time — the conservation /
drift check every model tutorial otherwise hand-rolls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.base import _ValidationPanel


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


__all__ = ["EnergyTimeSeriesPanel"]
