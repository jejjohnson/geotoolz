"""Physical-consistency metrics — V4.1.

Score how well a predicted field obeys known physical balances. A model
can match RMSE on velocity while violating geostrophic balance,
divergence-free flow, density stratification, or potential-vorticity
conservation; this module makes those failure modes first-class.

Layer-0 free functions:

- :func:`geostrophic_balance_error` — residual of
  ``f u + g ∂η/∂y`` and ``f v - g ∂η/∂x``.
- :func:`geostrophic_imbalance` — scale-invariant scalar
  ``rms(residual) / rms(f k×u)``.
- :func:`divergence_error` — magnitude of ``∂u/∂x + ∂v/∂y`` (≈ 0 in the
  geostrophic limit), as a field or its RMS.
- :func:`density_inversion_fraction` — fraction of cells where
  ``∂ρ/∂z < 0``.
- :func:`pv_conservation_error` — relative drift of potential vorticity
  along V3-style trajectories.

Layer-1 wrappers: :class:`GeostrophicBalanceError`,
:class:`GeostrophicImbalance`, :class:`DivergenceError`,
:class:`DensityInversionFraction`, :class:`PVConservationError`.

The derivative-based metrics default to lon/lat in degrees with
latitude-derived Coriolis. Idealised f/β-plane model output passes
``dims=("x", "y")``, ``geometry="cartesian"`` and an explicit ``f=``
(see :mod:`xrtoolz.ocn` for the accepted ``f`` forms).

Notes:
    Derivative-based metrics inherit the bias / noise of the underlying
    finite-difference stencil. Apply :func:`xrtoolz.metrics.PSDScore` or
    a coarse-grain step before the metric if the grid has resolved-scale
    noise that would dominate the residual.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr

import xrgrad
from xrcore import Operator
from xrtoolz.ocn._src.kinematics import (
    Coriolis,
    Geometry,
    _coriolis_field,
    _geom_kw,
    _partial,
)


# ---------- Layer-0: geostrophic balance ---------------------------------


def geostrophic_balance_error(
    ds: xr.Dataset,
    *,
    ssh_var: str = "ssh",
    u_var: str = "u",
    v_var: str = "v",
    lat: str = "lat",
    lon: str = "lon",
    g: float = xrgrad.GRAVITY,
    dims: tuple[str, str] | None = None,
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Residual of geostrophic balance for ``(η, u, v)``.

    Returns the two residual fields::

        r_u = f * u + g * ∂η/∂y
        r_v = f * v - g * ∂η/∂x

    Both should be ≈ 0 for a geostrophic flow. Differencing uses
    :func:`xrgrad.partial` under ``geometry`` (spherical metric by
    default).

    Args:
        ds: Dataset containing ``ssh_var``, ``u_var``, ``v_var``.
        ssh_var, u_var, v_var: Variable names.
        lat, lon: Names of latitude / longitude coordinates (degrees).
            Used when ``dims`` is ``None``.
        g: Gravitational acceleration (m/s²). Defaults to
            :data:`xrgrad.GRAVITY`.
        dims: ``(x_dim, y_dim)`` horizontal dims. ``None`` (default)
            means ``(lon, lat)``.
        geometry: ``"spherical"`` (default), ``"cartesian"`` or
            ``"rectilinear"``.
        f: Coriolis parameter. ``None`` derives it from latitude
            (spherical only); otherwise a scalar ``f₀``, an ``(f₀, β)`` /
            ``(f₀, β, y₀)`` β-plane tuple, or a DataArray.

    Returns:
        Dataset with two variables, ``"r_u"`` and ``"r_v"``, on the
        prediction grid.
    """
    xy = (lon, lat) if dims is None else dims
    eta = ds[ssh_var]
    u = ds[u_var]
    v = ds[v_var]
    f_field = _coriolis_field(eta, xy, geometry, f)

    deta_dx = _partial(eta, xy[0], xy, geometry)
    deta_dy = _partial(eta, xy[1], xy, geometry)

    r_u = (f_field * u + g * deta_dy).rename("r_u")
    r_v = (f_field * v - g * deta_dx).rename("r_v")
    return xr.Dataset({"r_u": r_u, "r_v": r_v})


def geostrophic_imbalance(
    ds: xr.Dataset,
    *,
    ssh_var: str = "ssh",
    u_var: str = "u",
    v_var: str = "v",
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
    g: float = xrgrad.GRAVITY,
    eps: float = 1e-30,
) -> xr.DataArray:
    """Dimensionless ageostrophic fraction ``rms(r) / rms(f k×u)``.

    Reduces the :func:`geostrophic_balance_error` residual ``r`` over
    every dim and normalises by the Coriolis acceleration, so the score
    is scale-invariant: ``0`` for an exactly geostrophic flow, ``O(1)``
    when ageostrophic accelerations rival Coriolis. NaN cells (land,
    stencil halo) are skipped.

    Args:
        ds: Dataset containing ``ssh_var``, ``u_var``, ``v_var``.
        ssh_var, u_var, v_var: Variable names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default), ``"cartesian"`` or
            ``"rectilinear"``.
        f: Coriolis parameter; see :func:`geostrophic_balance_error`.
        g: Gravitational acceleration (m/s²).
        eps: Floor on the denominator so a motionless state stays finite.

    Returns:
        Scalar :class:`xr.DataArray` named ``"geostrophic_imbalance"``.

    Example:
        Score a β-plane model run in metres::

            geostrophic_imbalance(
                run, ssh_var="eta", dims=("x", "y"),
                geometry="cartesian", f=(1e-4, 1.6e-11),
            )
    """
    res = geostrophic_balance_error(
        ds,
        ssh_var=ssh_var,
        u_var=u_var,
        v_var=v_var,
        g=g,
        dims=dims,
        geometry=geometry,
        f=f,
    )
    f_field = _coriolis_field(ds[ssh_var], dims, geometry, f)
    cor_u = f_field * ds[v_var]
    cor_v = f_field * ds[u_var]
    residual = np.sqrt((res["r_u"] ** 2).mean() + (res["r_v"] ** 2).mean())
    scale = np.sqrt((cor_u**2).mean() + (cor_v**2).mean())
    return (residual / (scale + eps)).rename("geostrophic_imbalance")


# ---------- Layer-0: divergence ------------------------------------------


def divergence_error(
    ds: xr.Dataset,
    *,
    u_var: str = "u",
    v_var: str = "v",
    lat: str = "lat",
    lon: str = "lon",
    dims: tuple[str, str] | None = None,
    geometry: Geometry = "spherical",
    reduce: Literal["rms"] | None = None,
) -> xr.DataArray:
    """Surface horizontal divergence ``∇·u``.

    For a purely geostrophic flow this is ≈ 0; values away from zero
    indicate either ageostrophic flow or numerical noise. Uses
    :func:`xrgrad.divergence`, so the spherical curvature term is
    included on the default geometry.

    Args:
        ds: Dataset containing ``u_var`` and ``v_var``.
        u_var, v_var: Velocity variable names.
        lat, lon: Latitude / longitude coordinate names, used when
            ``dims`` is ``None``.
        dims: ``(x_dim, y_dim)`` horizontal dims. ``None`` (default)
            means ``(lon, lat)``.
        geometry: ``"spherical"`` (default), ``"cartesian"`` or
            ``"rectilinear"``.
        reduce: ``None`` (default) returns the divergence field;
            ``"rms"`` returns the scalar ``sqrt(mean((∇·u)²))`` over every
            dim, skipping NaN.

    Returns:
        :class:`xr.DataArray` named ``"divergence"`` (field, 1/s) or
        ``"rms_divergence"`` (scalar, 1/s).
    """
    xy = (lon, lat) if dims is None else dims
    flow = ds[[u_var, v_var]]
    div = xrgrad.divergence(
        flow,
        (u_var, v_var),
        dims=xy,
        geometry=geometry,
        **_geom_kw(xy, geometry),
    )
    if reduce is None:
        return div.rename("divergence")
    if reduce == "rms":
        return np.sqrt((div**2).mean()).rename("rms_divergence")
    raise ValueError(f"reduce={reduce!r} must be None or 'rms'.")


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
        g: float = xrgrad.GRAVITY,
        dims: tuple[str, str] | None = None,
        geometry: Geometry = "spherical",
        f: Coriolis = None,
    ) -> None:
        self.ssh_var = ssh_var
        self.u_var = u_var
        self.v_var = v_var
        self.lat = lat
        self.lon = lon
        self.g = g
        self.dims = dims
        self.geometry = geometry
        self.f = f

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return geostrophic_balance_error(ds, **self.get_config())

    def get_config(self) -> dict[str, Any]:
        return {
            "ssh_var": self.ssh_var,
            "u_var": self.u_var,
            "v_var": self.v_var,
            "lat": self.lat,
            "lon": self.lon,
            "g": self.g,
            "dims": self.dims,
            "geometry": self.geometry,
            "f": self.f,
        }


class GeostrophicImbalance(Operator):
    """Operator wrapper for :func:`geostrophic_imbalance`."""

    def __init__(
        self,
        *,
        ssh_var: str = "ssh",
        u_var: str = "u",
        v_var: str = "v",
        dims: tuple[str, str] = ("lon", "lat"),
        geometry: Geometry = "spherical",
        f: Coriolis = None,
        g: float = xrgrad.GRAVITY,
        eps: float = 1e-30,
    ) -> None:
        self.ssh_var = ssh_var
        self.u_var = u_var
        self.v_var = v_var
        self.dims = dims
        self.geometry = geometry
        self.f = f
        self.g = g
        self.eps = eps

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return geostrophic_imbalance(ds, **self.get_config())

    def get_config(self) -> dict[str, Any]:
        return {
            "ssh_var": self.ssh_var,
            "u_var": self.u_var,
            "v_var": self.v_var,
            "dims": self.dims,
            "geometry": self.geometry,
            "f": self.f,
            "g": self.g,
            "eps": self.eps,
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
        dims: tuple[str, str] | None = None,
        geometry: Geometry = "spherical",
        reduce: Literal["rms"] | None = None,
    ) -> None:
        self.u_var = u_var
        self.v_var = v_var
        self.lat = lat
        self.lon = lon
        self.dims = dims
        self.geometry = geometry
        self.reduce = reduce

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return divergence_error(ds, **self.get_config())

    def get_config(self) -> dict[str, Any]:
        return {
            "u_var": self.u_var,
            "v_var": self.v_var,
            "lat": self.lat,
            "lon": self.lon,
            "dims": self.dims,
            "geometry": self.geometry,
            "reduce": self.reduce,
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
    "GeostrophicImbalance",
    "PVConservationError",
    "density_inversion_fraction",
    "divergence_error",
    "geostrophic_balance_error",
    "geostrophic_imbalance",
    "pv_conservation_error",
]
