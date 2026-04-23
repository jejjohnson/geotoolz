"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.ocn._src`."""

from __future__ import annotations

from typing import Any

from xr_toolz.core import Operator
from xr_toolz.ocn._src import (
    geostrophic as _geostrophic,
    ssh as _ssh,
    validation as _validation,
)


# ---------- validation -----------------------------------------------------


class ValidateSSH(Operator):
    def __init__(self, variable: str = "ssh"):
        self.variable = variable

    def _apply(self, ds):
        return _validation.validate_ssh(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class ValidateVelocity(Operator):
    def __init__(self, u: str = "u", v: str = "v"):
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _validation.validate_velocity(ds, u=self.u, v=self.v)

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v}


# ---------- SSH composition ------------------------------------------------


class CalculateSSHAlongtrack(Operator):
    def __init__(
        self,
        variable: str = "ssh",
        sla: str = "sla_filtered",
        mdt: str = "mdt",
        lwe: str = "lwe",
    ):
        self.variable = variable
        self.sla = sla
        self.mdt = mdt
        self.lwe = lwe

    def _apply(self, ds):
        return _ssh.calculate_ssh_alongtrack(
            ds,
            variable=self.variable,
            sla=self.sla,
            mdt=self.mdt,
            lwe=self.lwe,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "sla": self.sla,
            "mdt": self.mdt,
            "lwe": self.lwe,
        }


# ---------- geostrophic physics -------------------------------------------


class Streamfunction(Operator):
    def __init__(
        self,
        variable: str = "ssh",
        g: float | None = None,
        f0: float | None = None,
    ):
        self.variable = variable
        self.g = g
        self.f0 = f0

    def _apply(self, ds):
        return _geostrophic.streamfunction(
            ds, variable=self.variable, g=self.g, f0=self.f0
        )

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "g": self.g, "f0": self.f0}


class GeostrophicVelocities(Operator):
    def __init__(self, variable: str = "ssh"):
        self.variable = variable

    def _apply(self, ds):
        return _geostrophic.geostrophic_velocities(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class _UVOperator(Operator):
    """Shared init/config for operators that take ``u`` and ``v``."""

    def __init__(self, u: str = "u", v: str = "v"):
        self.u = u
        self.v = v

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v}


class KineticEnergy(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.kinetic_energy(ds, u=self.u, v=self.v)


class RelativeVorticity(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.relative_vorticity(ds, u=self.u, v=self.v)


class AbsoluteVorticity(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.absolute_vorticity(ds, u=self.u, v=self.v)


class Divergence(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.divergence(ds, u=self.u, v=self.v)


class ShearStrain(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.shear_strain(ds, u=self.u, v=self.v)


class TensorStrain(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.tensor_strain(ds, u=self.u, v=self.v)


class StrainMagnitude(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.strain_magnitude(ds, u=self.u, v=self.v)


class OkuboWeiss(_UVOperator):
    def _apply(self, ds):
        return _geostrophic.okubo_weiss(ds, u=self.u, v=self.v)


class Enstrophy(Operator):
    def __init__(self, variable: str = "vort_r"):
        self.variable = variable

    def _apply(self, ds):
        return _geostrophic.enstrophy(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class CoriolisNormalized(Operator):
    def __init__(self, variable: str, f0: float | None = None):
        self.variable = variable
        self.f0 = f0

    def _apply(self, ds):
        return _geostrophic.coriolis_normalized(ds, variable=self.variable, f0=self.f0)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "f0": self.f0}


__all__ = [
    "AbsoluteVorticity",
    "CalculateSSHAlongtrack",
    "CoriolisNormalized",
    "Divergence",
    "Enstrophy",
    "GeostrophicVelocities",
    "KineticEnergy",
    "OkuboWeiss",
    "RelativeVorticity",
    "ShearStrain",
    "StrainMagnitude",
    "Streamfunction",
    "TensorStrain",
    "ValidateSSH",
    "ValidateVelocity",
]
