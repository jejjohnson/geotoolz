"""Layer-1 operator wrappers for :mod:`xr_toolz.budgets`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from xr_toolz.budgets._src.flux import boundary_flux
from xr_toolz.budgets._src.heat import heat_budget_residual
from xr_toolz.budgets._src.ke import kinetic_energy_budget_residual
from xr_toolz.budgets._src.residual import budget_residual
from xr_toolz.budgets._src.salt import salt_budget_residual
from xr_toolz.budgets._src.volume import control_volume_integral
from xr_toolz.budgets._src.volume_budget import volume_budget_residual
from xr_toolz.core import Operator


class ControlVolumeIntegral(Operator):
    """Operator wrapper for :func:`control_volume_integral`."""

    def __init__(
        self,
        variable: str,
        *,
        volume_metrics: xr.Dataset,
        region: xr.DataArray | None = None,
        dims: Sequence[str] = ("z", "lat", "lon"),
        cell_volume_var: str = "cell_volume",
    ) -> None:
        self.variable = variable
        self.volume_metrics = volume_metrics
        self.region = region
        self.dims = tuple(dims)
        self.cell_volume_var = cell_volume_var

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return control_volume_integral(
            ds,
            variable=self.variable,
            volume_metrics=self.volume_metrics,
            region=self.region,
            dims=self.dims,
            cell_volume_var=self.cell_volume_var,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "volume_metrics": "<Dataset>",
            "region": "<DataArray>" if self.region is not None else None,
            "dims": list(self.dims),
            "cell_volume_var": self.cell_volume_var,
        }


class BoundaryFlux(Operator):
    """Operator wrapper for :func:`boundary_flux`."""

    def __init__(
        self,
        *,
        variable: str | None,
        velocity_vars: dict[str, str],
        face_metrics: xr.Dataset,
        region: xr.DataArray | None = None,
    ) -> None:
        self.variable = variable
        self.velocity_vars = dict(velocity_vars)
        self.face_metrics = face_metrics
        self.region = region

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return boundary_flux(
            ds,
            variable=self.variable,
            velocity_vars=self.velocity_vars,
            face_metrics=self.face_metrics,
            region=self.region,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "velocity_vars": dict(self.velocity_vars),
            "face_metrics": "<Dataset>",
            "region": "<DataArray>" if self.region is not None else None,
        }


class BudgetResidual(Operator):
    """Operator wrapper for :func:`budget_residual`."""

    def _apply(
        self,
        tendency: xr.DataArray,
        flux_divergence: xr.DataArray,
        *,
        source: xr.DataArray | None = None,
        sink: xr.DataArray | None = None,
    ) -> xr.DataArray:
        return budget_residual(tendency, flux_divergence, source=source, sink=sink)

    def get_config(self) -> dict[str, Any]:
        return {}


class _TracerBudgetOp(Operator):
    """Shared init for the per-tracer budget operators."""

    def __init__(
        self,
        *,
        tracer_var: str,
        u_var: str = "u",
        v_var: str = "v",
        w_var: str | None = None,
        surface_flux_var: str | None = None,
        time_dim: str = "time",
        lat: str = "lat",
        lon: str = "lon",
        depth: str | None = "depth",
    ) -> None:
        self.tracer_var = tracer_var
        self.u_var = u_var
        self.v_var = v_var
        self.w_var = w_var
        self.surface_flux_var = surface_flux_var
        self.time_dim = time_dim
        self.lat = lat
        self.lon = lon
        self.depth = depth

    def get_config(self) -> dict[str, Any]:
        return {
            "tracer_var": self.tracer_var,
            "u_var": self.u_var,
            "v_var": self.v_var,
            "w_var": self.w_var,
            "surface_flux_var": self.surface_flux_var,
            "time_dim": self.time_dim,
            "lat": self.lat,
            "lon": self.lon,
            "depth": self.depth,
        }


class HeatBudgetResidual(_TracerBudgetOp):
    """Operator wrapper for :func:`heat_budget_residual`."""

    def __init__(self, *, temp_var: str = "theta", **kw: Any) -> None:
        super().__init__(tracer_var=temp_var, **kw)

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return heat_budget_residual(
            ds,
            temp_var=self.tracer_var,
            u_var=self.u_var,
            v_var=self.v_var,
            w_var=self.w_var,
            surface_flux_var=self.surface_flux_var,
            time_dim=self.time_dim,
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
        )


class SaltBudgetResidual(_TracerBudgetOp):
    """Operator wrapper for :func:`salt_budget_residual`."""

    def __init__(self, *, salt_var: str = "so", **kw: Any) -> None:
        super().__init__(tracer_var=salt_var, **kw)

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return salt_budget_residual(
            ds,
            salt_var=self.tracer_var,
            u_var=self.u_var,
            v_var=self.v_var,
            w_var=self.w_var,
            surface_flux_var=self.surface_flux_var,
            time_dim=self.time_dim,
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
        )


class VolumeBudgetResidual(Operator):
    """Operator wrapper for :func:`volume_budget_residual`."""

    def __init__(
        self,
        *,
        u_var: str = "u",
        v_var: str = "v",
        w_var: str | None = None,
        lat: str = "lat",
        lon: str = "lon",
        depth: str | None = "depth",
    ) -> None:
        self.u_var = u_var
        self.v_var = v_var
        self.w_var = w_var
        self.lat = lat
        self.lon = lon
        self.depth = depth

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return volume_budget_residual(
            ds,
            u_var=self.u_var,
            v_var=self.v_var,
            w_var=self.w_var,
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "u_var": self.u_var,
            "v_var": self.v_var,
            "w_var": self.w_var,
            "lat": self.lat,
            "lon": self.lon,
            "depth": self.depth,
        }


class KineticEnergyBudgetResidual(Operator):
    """Operator wrapper for :func:`kinetic_energy_budget_residual`."""

    def __init__(
        self,
        *,
        u_var: str = "u",
        v_var: str = "v",
        forcing_vars: Sequence[str] | None = None,
        time_dim: str = "time",
        lat: str = "lat",
        lon: str = "lon",
        depth: str | None = "depth",
        w_var: str | None = None,
    ) -> None:
        self.u_var = u_var
        self.v_var = v_var
        self.forcing_vars = None if forcing_vars is None else tuple(forcing_vars)
        self.time_dim = time_dim
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.w_var = w_var

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return kinetic_energy_budget_residual(
            ds,
            u_var=self.u_var,
            v_var=self.v_var,
            forcing_vars=self.forcing_vars,
            time_dim=self.time_dim,
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
            w_var=self.w_var,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "u_var": self.u_var,
            "v_var": self.v_var,
            "forcing_vars": (
                None if self.forcing_vars is None else list(self.forcing_vars)
            ),
            "time_dim": self.time_dim,
            "lat": self.lat,
            "lon": self.lon,
            "depth": self.depth,
            "w_var": self.w_var,
        }


__all__ = [
    "BoundaryFlux",
    "BudgetResidual",
    "ControlVolumeIntegral",
    "HeatBudgetResidual",
    "KineticEnergyBudgetResidual",
    "SaltBudgetResidual",
    "VolumeBudgetResidual",
]
