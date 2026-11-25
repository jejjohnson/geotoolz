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
(see :mod:`xrops.ocn` for the accepted ``f`` forms).

Notes:
    Derivative-based metrics inherit the bias / noise of the underlying
    finite-difference stencil. Apply :func:`xrops.metrics.PSDScore` or
    a coarse-grain step before the metric if the grid has resolved-scale
    noise that would dominate the residual.
"""

from __future__ import annotations

from typing import Any, Literal

import gridax
import numpy as np
import xarray as xr
from diagnostix import kinematics, skill

from xrops import grad
from xrops._src.collocated import Geometry, on_grid, over_grid
from xrops._src.columns import levels_of, on_columns
from xrops._src.operator import Operator
from xrops.ocn._src.kinematics import Coriolis, _coriolis_on


# ---------- Layer-0: geostrophic balance ---------------------------------


def geostrophic_balance_error(
    ds: xr.Dataset,
    *,
    ssh_var: str = "ssh",
    u_var: str = "u",
    v_var: str = "v",
    lat: str = "lat",
    lon: str = "lon",
    g: float = grad.GRAVITY,
    dims: tuple[str, str] | None = None,
    geometry: Geometry = "spherical",
    f: Coriolis = None,
) -> xr.Dataset:
    """Residual of geostrophic balance for ``(η, u, v)``.

    Returns the two residual fields::

        r_u = f * u + g * ∂η/∂y
        r_v = f * v - g * ∂η/∂x

    Both should be ≈ 0 for a geostrophic flow. Differencing uses
    :func:`gridax.gradient` under ``geometry`` (spherical metric by
    default).

    Args:
        ds: Dataset containing ``ssh_var``, ``u_var``, ``v_var``.
        ssh_var, u_var, v_var: Variable names.
        lat, lon: Names of latitude / longitude coordinates (degrees).
            Used when ``dims`` is ``None``.
        g: Gravitational acceleration (m/s²). Defaults to
            :data:`xrops.grad.GRAVITY`.
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
    f_field = _coriolis_on(eta, xy, geometry, f)

    def component(index: int) -> xr.DataArray:
        return on_grid(
            lambda u_, v_, height, coriolis, grid: kinematics.geostrophic_residual(
                u_, v_, height, grid, f=coriolis, gravity=g
            )[index],
            u,
            v,
            eta,
            f_field,
            dims=xy,
            geometry=geometry,
        )

    return xr.Dataset({"r_u": component(0), "r_v": component(1)})


def geostrophic_imbalance(
    ds: xr.Dataset,
    *,
    ssh_var: str = "ssh",
    u_var: str = "u",
    v_var: str = "v",
    dims: tuple[str, str] = ("lon", "lat"),
    geometry: Geometry = "spherical",
    f: Coriolis = None,
    g: float = grad.GRAVITY,
) -> xr.DataArray:
    """Dimensionless ageostrophic fraction ``rms(r) / rms(f k×u)``.

    The :func:`geostrophic_balance_error` residual ``r`` against the Coriolis
    acceleration, both as means over the area of each plane
    (``diagnostix.skill.geostrophic_imbalance``): ``0`` for an exactly
    geostrophic flow, ``O(1)`` when ageostrophic accelerations rival Coriolis.
    A cell that is missing in any plane of any field is land, and is not read.

    Args:
        ds: Dataset containing ``ssh_var``, ``u_var``, ``v_var``.
        ssh_var, u_var, v_var: Variable names.
        dims: ``(x_dim, y_dim)`` horizontal dims. Default ``("lon", "lat")``.
        geometry: ``"spherical"`` (default), ``"cartesian"`` or
            ``"rectilinear"``.
        f: Coriolis parameter; see :func:`geostrophic_balance_error`.
        g: Gravitational acceleration (m/s²).

    Returns:
        :class:`xr.DataArray` named ``"geostrophic_imbalance"``, on the dims
        that are not horizontal. NaN for a state of rest.

    Example:
        Score a β-plane model run in metres::

            geostrophic_imbalance(
                run, ssh_var="eta", dims=("x", "y"),
                geometry="cartesian", f=(1e-4, 1.6e-11),
            )
    """
    eta, u, v = ds[ssh_var], ds[u_var], ds[v_var]
    mask = _land_mask([u, v, eta], dims, geometry)

    def kernel(u_, v_, height, coriolis, grid):
        return skill.geostrophic_imbalance(
            u_, v_, height, grid, f=coriolis, gravity=g, mask=mask
        )

    out = over_grid(
        kernel,
        u,
        v,
        eta,
        _coriolis_on(eta, dims, geometry, f),
        dims=dims,
        geometry=geometry,
    )
    return out.rename("geostrophic_imbalance")


def _land_mask(
    fields: list[xr.DataArray], dims: tuple[str, str], geometry: Geometry
) -> gridax.Mask2D | None:
    """The cells that every plane of every field has; none if that is all.

    A row at a pole has no eastward derivative, and is land too.
    """
    wet = fields[0].notnull()
    for field in fields[1:]:
        wet = wet & field.notnull()
    wet = wet.all([d for d in wet.dims if d not in dims])
    inside = np.array(wet.transpose(dims[1], dims[0]).values)
    if geometry == "spherical":
        inside[np.isclose(np.abs(fields[0][dims[1]].values), 90.0)] = False
    return None if inside.all() else gridax.Mask2D.from_mask(inside)


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
    :func:`gridax.divergence`, so the spherical curvature term is
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
            ``"rms"`` returns ``sqrt(<(∇·u)²>)``, with the mean over the area
            of each plane (``diagnostix.skill.rms_divergence``). A cell that is
            missing in any plane is land, and is not read.

    Returns:
        :class:`xr.DataArray` named ``"divergence"`` (field, 1/s) or
        ``"rms_divergence"`` (1/s), on the dims that are not horizontal.
    """
    xy = (lon, lat) if dims is None else dims
    u, v = ds[u_var], ds[v_var]
    if reduce is None:
        div = on_grid(gridax.divergence, u, v, dims=xy, geometry=geometry)
        return div.rename("divergence")
    if reduce != "rms":
        raise ValueError(f"reduce={reduce!r} must be None or 'rms'.")
    mask = _land_mask([u, v], xy, geometry)
    out = over_grid(
        lambda u_, v_, grid: skill.rms_divergence(u_, v_, grid, mask=mask),
        u,
        v,
        dims=xy,
        geometry=geometry,
    )
    return out.rename("rms_divergence")


# ---------- Layer-0: density inversion -----------------------------------


def density_inversion_fraction(
    ds: xr.Dataset,
    *,
    density_var: str = "rho",
    depth_dim: str = "depth",
) -> xr.DataArray:
    """Share of neighbouring levels of each column with the denser one on top.

    Over the pairs of levels that both have data
    (``diagnostix.skill.density_inversion_fraction``).

    Args:
        ds: Dataset with ``density_var`` on a vertical axis.
        density_var: Density variable name.
        depth_dim: Vertical dimension name, with a coordinate that is positive
            downward.

    Returns:
        :class:`xr.DataArray` in ``[0, 1]``, on the dims that are not vertical.
        NaN in a column without such a pair, such as land.
    """
    rho = ds[density_var]
    if depth_dim not in rho.dims:
        raise ValueError(
            f"Density variable {density_var!r} is missing depth dim "
            f"{depth_dim!r}; got dims={tuple(rho.dims)}."
        )
    out = on_columns(
        lambda rho_, z: skill.density_inversion_fraction(
            rho_, z, valid=~np.isnan(rho_)
        ),
        rho,
        -levels_of(rho, depth_dim),
        dim=depth_dim,
        reduces=True,
    )
    return out.rename("density_inversion_fraction")


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
        g: float = grad.GRAVITY,
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
        g: float = grad.GRAVITY,
    ) -> None:
        self.ssh_var = ssh_var
        self.u_var = u_var
        self.v_var = v_var
        self.dims = dims
        self.geometry = geometry
        self.f = f
        self.g = g

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
