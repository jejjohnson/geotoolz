"""V6.3 panel — process-budget term breakdown.

Stacked-area chart of budget terms (tendency, advection, sources,
sinks) vs the residual. Per V6.3 issue notes, V4.3 currently returns
only the residual field; this panel therefore also accepts a separate
``components`` mapping ``{"tendency": ..., "advection": ..., ...}``
when the user has computed the breakdown manually. If only the
residual is supplied, the panel plots the residual time series and
notes that components were not provided.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xr_toolz.viz.validation._src.base import _ValidationPanel


class ProcessBudgetPanel(_ValidationPanel):
    """Stacked-area chart of budget components + residual line.

    Args:
        time_dim: Time coordinate name to plot against. Default
            ``"time"``.
        reduce_dims: Spatial dims to mean-reduce before plotting.
            Default ``("lat", "lon")``.
    """

    def __init__(
        self,
        *,
        time_dim: str = "time",
        reduce_dims: tuple[str, ...] = ("lat", "lon"),
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.time_dim = time_dim
        self.reduce_dims = tuple(reduce_dims)

    def _default_title(self) -> str:
        return "Process Budget"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        residual: xr.DataArray,
        components: Mapping[str, xr.DataArray] | None = None,
    ) -> None:
        ax = axes
        active = [d for d in self.reduce_dims if d in residual.dims]
        residual_ts = residual.mean(dim=active) if active else residual
        time = residual_ts[self.time_dim].values

        if components:
            stack: list[np.ndarray] = []
            labels: list[str] = []
            for name, da in components.items():
                active_d = [d for d in self.reduce_dims if d in da.dims]
                ts = da.mean(dim=active_d) if active_d else da
                stack.append(np.asarray(ts.values))
                labels.append(name)
            ax.stackplot(time, np.vstack(stack), labels=labels, alpha=0.75)
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(
                0.02,
                0.95,
                "components not supplied — residual only",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                color="grey",
            )

        ax.plot(
            time,
            np.asarray(residual_ts.values),
            color="black",
            lw=1.5,
            label="residual",
        )
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.set_xlabel(self.time_dim)
        ax.set_ylabel("budget terms")
        ax.grid(True, alpha=0.3)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "time_dim": self.time_dim,
            "reduce_dims": list(self.reduce_dims),
        }


__all__ = ["ProcessBudgetPanel"]
