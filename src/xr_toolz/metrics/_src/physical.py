"""Physical-consistency metrics — V4.1.

Score how well a predicted field obeys known physical balances. A model
can match RMSE on velocity while violating geostrophic balance,
divergence-free flow, density stratification, or potential-vorticity
conservation; this module makes those failure modes first-class.

Layer-0 free functions:

- :func:`geostrophic_balance_error` — residual of
  ``f u + g ∂η/∂y`` and ``f v - g ∂η/∂x``.
- :func:`divergence_error` — magnitude of ``∂u/∂x + ∂v/∂y`` (≈ 0 in the
  geostrophic limit).
- :func:`density_inversion_fraction` — fraction of cells where
  ``∂ρ/∂z < 0``.
- :func:`pv_conservation_error` — relative drift of potential vorticity
  along V3-style trajectories.

Layer-1 wrappers: :class:`GeostrophicBalanceError`, :class:`DivergenceError`,
:class:`DensityInversionFraction`, :class:`PVConservationError`.

Notes:
    Derivative-based metrics inherit the bias / noise of the underlying
    finite-difference stencil. Apply :func:`xr_toolz.metrics.PSDScore` or
    a coarse-grain step before the metric if the grid has resolved-scale
    noise that would dominate the residual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from xr_toolz import calc
from xr_toolz.core import Operator
from xr_toolz.ocn._src.kinematics import coriolis_parameter


# ---------- Layer-0: geostrophic balance ---------------------------------


def geostrophic_balance_error(
    ds: xr.Dataset,
    *,
    ssh_var: str = "ssh",
    u_var: str = "u",
    v_var: str = "v",
    lat: str = "lat",
    lon: str = "lon",
    g: float = calc.GRAVITY,
) -> xr.Dataset:
    """Residual of geostrophic balance for ``(η, u, v)``.

    Returns the two scalar residuals::

        r_u = f * u + g * ∂η/∂y
        r_v = f * v - g * ∂η/∂x

    Both should be ≈ 0 for a geostrophic flow. Differencing uses the
    spherical-metric :func:`xr_toolz.calc.gradient`.

    Args:
        ds: Dataset containing ``ssh_var``, ``u_var``, ``v_var`` on
            ``(lat, lon)``.
        ssh_var, u_var, v_var: Variable names.
        lat, lon: Names of latitude / longitude coordinates (degrees).
        g: Gravitational acceleration (m/s²). Defaults to
            :data:`xr_toolz.calc.GRAVITY`.

    Returns:
        Dataset with two variables, ``"r_u"`` and ``"r_v"``, on the
        prediction grid.
    """
    eta = ds[ssh_var]
    u = ds[u_var]
    v = ds[v_var]
    f = coriolis_parameter(ds[lat])

    grads = calc.gradient(eta, dims=(lon, lat), geometry="spherical", lon=lon, lat=lat)
    deta_dx = grads[f"d{eta.name}_dx"]
    deta_dy = grads[f"d{eta.name}_dy"]

    r_u = (f * u + g * deta_dy).rename("r_u")
    r_v = (f * v - g * deta_dx).rename("r_v")
    return xr.Dataset({"r_u": r_u, "r_v": r_v})


# ---------- Layer-0: divergence ------------------------------------------


def divergence_error(
    ds: xr.Dataset,
    *,
    u_var: str = "u",
    v_var: str = "v",
    lat: str = "lat",
    lon: str = "lon",
) -> xr.DataArray:
    """Surface horizontal divergence ``∇·u`` with spherical curvature.

    For a purely geostrophic flow this is ≈ 0; values away from zero
    indicate either ageostrophic flow or numerical noise. Uses
    :func:`xr_toolz.calc.divergence` so the curvature term is included.
    """
    flow = ds[[u_var, v_var]]
    div = calc.divergence(
        flow,
        (u_var, v_var),
        dims=(lon, lat),
        geometry="spherical",
        lon=lon,
        lat=lat,
    )
    return div.rename("divergence")


# ---------- Layer-0: density inversion -----------------------------------


def density_inversion_fraction(
    ds: xr.Dataset,
    *,
    density_var: str = "rho",
    depth_dim: str = "depth",
) -> xr.DataArray:
    """Fraction of inversion cells, averaged over all input dims.

    Counts ``∂ρ/∂z < 0`` along ``depth_dim`` then averages the
    Boolean mask over **every** dim of the difference field
    (including ``depth_dim``), returning a scalar.

    Args:
        ds: Dataset with ``density_var`` on a vertical axis.
        density_var: Density variable name.
        depth_dim: Vertical dimension name. Convention: increasing
            ``depth`` points downward (deeper). Inversions are pairs
            with ``∂ρ/∂z < 0``.

    Returns:
        Scalar :class:`xr.DataArray` in ``[0, 1]``.
    """
    rho = ds[density_var]
    if depth_dim not in rho.dims:
        raise ValueError(
            f"Density variable {density_var!r} is missing depth dim "
            f"{depth_dim!r}; got dims={tuple(rho.dims)}."
        )
    drho_dz = rho.diff(depth_dim)
    inversions = (drho_dz < 0).astype(float)
    frac = inversions.mean()
    return frac.rename("density_inversion_fraction")


# ---------- Layer-0: PV conservation -------------------------------------


def pv_conservation_error(
    trajectories: xr.Dataset,
    *,
    pv_var: str = "pv",
    traj_dim: str = "trajectory",
    time_dim: str = "time",
) -> xr.DataArray:
    """Relative drift of potential vorticity along Lagrangian trajectories.

    For each trajectory, computes ``std(PV) / mean(|PV|)`` along the
    time axis, then averages across trajectories. PV is materially
    conserved in the absence of friction / diabatic forcing, so a
    well-resolved Lagrangian model should return ≈ 0.

    Args:
        trajectories: V3.1-conformant trajectory Dataset with dims
            ``(traj_dim, time_dim)`` and a ``pv_var`` variable.
        pv_var, traj_dim, time_dim: See V3.1 schema.
    """
    pv = trajectories[pv_var]
    if traj_dim not in pv.dims or time_dim not in pv.dims:
        raise ValueError(
            f"Trajectories must have dims ({traj_dim!r}, {time_dim!r}); "
            f"got {tuple(pv.dims)}."
        )
    std_per_traj = pv.std(dim=time_dim)
    mean_abs_per_traj = np.abs(pv).mean(dim=time_dim)
    rel = std_per_traj / mean_abs_per_traj.where(mean_abs_per_traj != 0)
    return rel.mean(dim=traj_dim).rename("pv_conservation_error")


# ---------- Layer-1: Operators -------------------------------------------


class GeostrophicBalanceError(Operator):
    """Operator wrapper for :func:`geostrophic_balance_error`."""

    def __init__(
        self,
        *,
        ssh_var: str = "ssh",
        u_var: str = "u",
        v_var: str = "v",
        lat: str = "lat",
        lon: str = "lon",
        g: float = calc.GRAVITY,
    ) -> None:
        self.ssh_var = ssh_var
        self.u_var = u_var
        self.v_var = v_var
        self.lat = lat
        self.lon = lon
        self.g = g

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return geostrophic_balance_error(
            ds,
            ssh_var=self.ssh_var,
            u_var=self.u_var,
            v_var=self.v_var,
            lat=self.lat,
            lon=self.lon,
            g=self.g,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "ssh_var": self.ssh_var,
            "u_var": self.u_var,
            "v_var": self.v_var,
            "lat": self.lat,
            "lon": self.lon,
            "g": self.g,
        }


class DivergenceError(Operator):
    """Operator wrapper for :func:`divergence_error`."""

    def __init__(
        self,
        *,
        u_var: str = "u",
        v_var: str = "v",
        lat: str = "lat",
        lon: str = "lon",
    ) -> None:
        self.u_var = u_var
        self.v_var = v_var
        self.lat = lat
        self.lon = lon

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return divergence_error(
            ds, u_var=self.u_var, v_var=self.v_var, lat=self.lat, lon=self.lon
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "u_var": self.u_var,
            "v_var": self.v_var,
            "lat": self.lat,
            "lon": self.lon,
        }


class DensityInversionFraction(Operator):
    """Operator wrapper for :func:`density_inversion_fraction`."""

    def __init__(self, *, density_var: str = "rho", depth_dim: str = "depth") -> None:
        self.density_var = density_var
        self.depth_dim = depth_dim

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return density_inversion_fraction(
            ds, density_var=self.density_var, depth_dim=self.depth_dim
        )

    def get_config(self) -> dict[str, Any]:
        return {"density_var": self.density_var, "depth_dim": self.depth_dim}


class PVConservationError(Operator):
    """Operator wrapper for :func:`pv_conservation_error`."""

    def __init__(
        self,
        *,
        pv_var: str = "pv",
        traj_dim: str = "trajectory",
        time_dim: str = "time",
    ) -> None:
        self.pv_var = pv_var
        self.traj_dim = traj_dim
        self.time_dim = time_dim

    def _apply(self, trajectories: xr.Dataset) -> xr.DataArray:
        return pv_conservation_error(
            trajectories,
            pv_var=self.pv_var,
            traj_dim=self.traj_dim,
            time_dim=self.time_dim,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "pv_var": self.pv_var,
            "traj_dim": self.traj_dim,
            "time_dim": self.time_dim,
        }


__all__ = [
    "DensityInversionFraction",
    "DivergenceError",
    "GeostrophicBalanceError",
    "PVConservationError",
    "density_inversion_fraction",
    "divergence_error",
    "geostrophic_balance_error",
    "pv_conservation_error",
]
