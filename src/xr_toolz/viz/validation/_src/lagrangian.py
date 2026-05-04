"""V6.3 panel — side-by-side Eulerian field + Lagrangian trajectories."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import xarray as xr

from xr_toolz.viz.validation._src.base import _ValidationPanel


class EulerianLagrangianPanel(_ValidationPanel):
    """Side-by-side Eulerian field + Lagrangian trajectory bundle.

    Consumes ``(eulerian_ds, trajectory_ds)``. The trajectory Dataset
    follows the V3.1 schema: dims ``(traj_dim, time_dim)`` with at
    least ``lon``, ``lat`` variables sampled along each track.

    Args:
        eulerian_var: Variable to plot from the Eulerian Dataset.
        traj_dim, time_dim: Dimension names in the trajectory Dataset.
        lon, lat: Coordinate names in *both* Datasets.
    """

    _default_axes_layout = (1, 2)

    def __init__(
        self,
        *,
        eulerian_var: str = "ssh",
        traj_dim: str = "trajectory",
        time_dim: str = "time",
        lon: str = "lon",
        lat: str = "lat",
        **kw: Any,
    ) -> None:
        kw.setdefault("figsize", (11, 4.5))
        super().__init__(**kw)
        self.eulerian_var = eulerian_var
        self.traj_dim = traj_dim
        self.time_dim = time_dim
        self.lon = lon
        self.lat = lat

    def _default_title(self) -> str:
        return "Eulerian field + Lagrangian trajectories"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        eulerian: xr.Dataset,
        trajectories: xr.Dataset,
    ) -> None:
        ax_e, ax_l = axes
        field = eulerian[self.eulerian_var]
        if self.time_dim in field.dims:
            field = field.isel({self.time_dim: 0})
        field_t = field.transpose(self.lat, self.lon)
        mesh = ax_e.pcolormesh(
            field_t[self.lon].values,
            field_t[self.lat].values,
            field_t.values,
            shading="auto",
        )
        fig.colorbar(mesh, ax=ax_e, label=self.eulerian_var)
        ax_e.set_xlabel(self.lon)
        ax_e.set_ylabel(self.lat)
        ax_e.set_title(f"Eulerian: {self.eulerian_var}")

        lons = trajectories[self.lon].transpose(self.traj_dim, self.time_dim).values
        lats = trajectories[self.lat].transpose(self.traj_dim, self.time_dim).values
        for i in range(lons.shape[0]):
            ax_l.plot(lons[i], lats[i], lw=0.8, alpha=0.7)
            ax_l.scatter(lons[i, 0], lats[i, 0], s=12, marker="o", color="green")
            ax_l.scatter(lons[i, -1], lats[i, -1], s=12, marker="x", color="red")
        ax_l.set_xlabel(self.lon)
        ax_l.set_ylabel(self.lat)
        ax_l.set_title("Lagrangian trajectories")
        ax_l.grid(True, alpha=0.3)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "eulerian_var": self.eulerian_var,
            "traj_dim": self.traj_dim,
            "time_dim": self.time_dim,
            "lon": self.lon,
            "lat": self.lat,
        }


__all__ = ["EulerianLagrangianPanel"]
