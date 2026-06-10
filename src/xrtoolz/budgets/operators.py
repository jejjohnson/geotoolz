"""Layer-1 operator wrappers for :mod:`xrtoolz.budgets`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.budgets._src.flux import boundary_flux
from xrtoolz.budgets._src.heat import heat_budget_residual
from xrtoolz.budgets._src.ke import kinetic_energy_budget_residual
from xrtoolz.budgets._src.residual import budget_residual
from xrtoolz.budgets._src.salt import salt_budget_residual
from xrtoolz.budgets._src.volume import control_volume_integral
from xrtoolz.budgets._src.volume_budget import volume_budget_residual


class ControlVolumeIntegral(Operator):
    """Volume-weighted integral of a field over a control volume.

    Integrates ``variable`` over ``dims`` weighted by the cell volumes in
    ``volume_metrics`` (optionally masked to ``region``). The metrics are
    explicit, never auto-derived — build them with
    :func:`xrtoolz.calc.grid_metrics_from_coords` if the data does not ship
    them.

    Args:
        variable: Name of the field to integrate.
        volume_metrics: Dataset carrying the ``cell_volume_var`` weights.
        region: Optional boolean mask selecting the control volume.
        dims: Dimensions integrated over.
        cell_volume_var: Name of the cell-volume variable in
            ``volume_metrics``.

    Returns:
        The volume integral as a DataArray (reduced over ``dims``).
    """

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
    """Advective flux through the faces of a control volume.

    Integrates the ``velocity_vars`` (optionally carrying ``variable``) over
    the face areas in ``face_metrics``, optionally restricted to ``region``.
    The metrics are explicit, never auto-derived.

    Args:
        variable: Tracer carried by the flow, or ``None`` for a pure volume
            (velocity) flux.
        velocity_vars: Mapping of face direction → velocity variable name.
        face_metrics: Dataset carrying the face-area weights.
        region: Optional boolean mask selecting the control volume.

    Returns:
        Dataset of per-boundary fluxes.
    """

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
    """Generic conservation residual ``∂φ/∂t + ∇·F − source + sink``.

    Combines a precomputed tendency and flux divergence (with optional
    source / sink terms) into the budget-closure residual; a perfectly
    closed budget is zero. Inputs are passed to ``__call__``, not the
    constructor.

    Args:
        tendency: Local time tendency ``∂φ/∂t``.
        flux_divergence: Divergence of the flux ``∇·F``.
        source: Optional source term.
        sink: Optional sink term.

    Returns:
        The budget-closure residual DataArray.
    """

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
    """Shared ``__init__`` / ``get_config`` for per-tracer budget operators.

    Holds the tracer, velocity, surface-flux, and coordinate-name
    configuration common to :class:`HeatBudgetResidual` and
    :class:`SaltBudgetResidual`.
    """

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
    """Per-cell heat-budget residual ``∂θ/∂t + ∇·(u θ) − Q``.

    Uses the spherical-metric divergence from :mod:`xrtoolz.calc`; returns
    the per-cell residual (zero for a closed budget). It does **not** consume
    volume/face metrics — multiply by ``cell_volume`` and integrate with
    :class:`ControlVolumeIntegral` for a control-volume closure.

    Args:
        temp_var: Temperature (θ) variable name.
        u_var: Eastward velocity variable name.
        v_var: Northward velocity variable name.
        w_var: Vertical velocity variable name, or ``None``.
        surface_flux_var: Optional surface heat-flux variable name.
        time_dim: Time dimension name.
        lat: Latitude coordinate name.
        lon: Longitude coordinate name.
        depth: Vertical coordinate name, or ``None`` for a 2-D budget.

    Returns:
        The per-cell heat-budget residual DataArray.
    """

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
    """Per-cell salt-budget residual ``∂S/∂t + ∇·(u S) − F``.

    Like :class:`HeatBudgetResidual` but for salinity; returns the per-cell
    residual (zero for a closed budget).

    Args:
        salt_var: Salinity (S) variable name.
        u_var: Eastward velocity variable name.
        v_var: Northward velocity variable name.
        w_var: Vertical velocity variable name, or ``None``.
        surface_flux_var: Optional surface salt-flux variable name.
        time_dim: Time dimension name.
        lat: Latitude coordinate name.
        lon: Longitude coordinate name.
        depth: Vertical coordinate name, or ``None`` for a 2-D budget.

    Returns:
        The per-cell salt-budget residual DataArray.
    """

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
    """Per-cell volume (continuity) residual ``∇·u``.

    The horizontal + vertical velocity divergence; zero for a non-divergent
    (volume-conserving) flow.

    Args:
        u_var: Eastward velocity variable name.
        v_var: Northward velocity variable name.
        w_var: Vertical velocity variable name, or ``None``.
        lat: Latitude coordinate name.
        lon: Longitude coordinate name.
        depth: Vertical coordinate name, or ``None`` for a 2-D budget.

    Returns:
        The per-cell continuity residual DataArray.
    """

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
    """Per-cell kinetic-energy budget residual.

    Residual of ``∂KE/∂t + ∇·(u KE) − forcing`` with ``KE = ½(u² + v²)``;
    zero for a closed KE budget.

    Args:
        u_var: Eastward velocity variable name.
        v_var: Northward velocity variable name.
        forcing_vars: Optional forcing / dissipation variable names summed as
            the source term.
        time_dim: Time dimension name.
        lat: Latitude coordinate name.
        lon: Longitude coordinate name.
        depth: Vertical coordinate name, or ``None`` for a 2-D budget.
        w_var: Vertical velocity variable name, or ``None``.

    Returns:
        The per-cell KE-budget residual DataArray.
    """

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
