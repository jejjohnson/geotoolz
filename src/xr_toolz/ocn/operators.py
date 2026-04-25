"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.ocn._src`."""

from __future__ import annotations

from typing import Any

from xr_toolz.core import Operator
from xr_toolz.ocn._src import (
    kinematics as _kinematics,
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


# ---------- kinematic physics ---------------------------------------------


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
        return _kinematics.streamfunction(
            ds, variable=self.variable, g=self.g, f0=self.f0
        )

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "g": self.g, "f0": self.f0}


class GeostrophicVelocities(Operator):
    def __init__(self, variable: str = "ssh"):
        self.variable = variable

    def _apply(self, ds):
        return _kinematics.geostrophic_velocities(ds, variable=self.variable)

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
        return _kinematics.kinetic_energy(ds, u=self.u, v=self.v)


class RelativeVorticity(_UVOperator):
    def _apply(self, ds):
        return _kinematics.relative_vorticity(ds, u=self.u, v=self.v)


class AbsoluteVorticity(_UVOperator):
    def _apply(self, ds):
        return _kinematics.absolute_vorticity(ds, u=self.u, v=self.v)


class Divergence(_UVOperator):
    def _apply(self, ds):
        return _kinematics.divergence(ds, u=self.u, v=self.v)


class ShearStrain(_UVOperator):
    def _apply(self, ds):
        return _kinematics.shear_strain(ds, u=self.u, v=self.v)


class TensorStrain(_UVOperator):
    def _apply(self, ds):
        return _kinematics.tensor_strain(ds, u=self.u, v=self.v)


class StrainMagnitude(_UVOperator):
    def _apply(self, ds):
        return _kinematics.strain_magnitude(ds, u=self.u, v=self.v)


class OkuboWeiss(_UVOperator):
    def _apply(self, ds):
        return _kinematics.okubo_weiss(ds, u=self.u, v=self.v)


class Enstrophy(Operator):
    def __init__(self, variable: str = "vort_r"):
        self.variable = variable

    def _apply(self, ds):
        return _kinematics.enstrophy(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class CoriolisNormalized(Operator):
    def __init__(self, variable: str, f0: float | None = None):
        self.variable = variable
        self.f0 = f0

    def _apply(self, ds):
        return _kinematics.coriolis_normalized(ds, variable=self.variable, f0=self.f0)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "f0": self.f0}


class AgeostrophicVelocities(Operator):
    def __init__(self, variable: str = "ssh", u: str = "u", v: str = "v"):
        self.variable = variable
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _kinematics.ageostrophic_velocities(
            ds, variable=self.variable, u=self.u, v=self.v
        )

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "u": self.u, "v": self.v}


class Advection(Operator):
    def __init__(
        self,
        scalar: str,
        components: tuple[str, ...] = ("u", "v"),
        dims: tuple[str, ...] = ("lon", "lat"),
    ):
        self.scalar = scalar
        self.components = components
        self.dims = dims

    def _apply(self, ds):
        return _kinematics.advection(
            ds, scalar=self.scalar, components=self.components, dims=self.dims
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "scalar": self.scalar,
            "components": self.components,
            "dims": self.dims,
        }


class ShearVorticity(_UVOperator):
    def _apply(self, ds):
        return _kinematics.shear_vorticity(ds, u=self.u, v=self.v)


class CurvatureVorticity(_UVOperator):
    def _apply(self, ds):
        return _kinematics.curvature_vorticity(ds, u=self.u, v=self.v)


class Frontogenesis(Operator):
    def __init__(self, scalar: str, u: str = "u", v: str = "v"):
        self.scalar = scalar
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _kinematics.frontogenesis(ds, scalar=self.scalar, u=self.u, v=self.v)

    def get_config(self) -> dict[str, Any]:
        return {"scalar": self.scalar, "u": self.u, "v": self.v}


class PotentialVorticityBarotropic(Operator):
    def __init__(self, height: str = "h", u: str = "u", v: str = "v"):
        self.height = height
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _kinematics.potential_vorticity_barotropic(
            ds, height=self.height, u=self.u, v=self.v
        )

    def get_config(self) -> dict[str, Any]:
        return {"height": self.height, "u": self.u, "v": self.v}


class VelocityMagnitude(Operator):
    def __init__(self, u: str = "u", v: str = "v", w: str | None = None):
        self.u = u
        self.v = v
        self.w = w

    def _apply(self, ds):
        return _kinematics.velocity_magnitude(ds, u=self.u, v=self.v, w=self.w)

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v, "w": self.w}


class HorizontalVelocityMagnitude(_UVOperator):
    def _apply(self, ds):
        return _kinematics.horizontal_velocity_magnitude(ds, u=self.u, v=self.v)


class EddyKineticEnergy(Operator):
    def __init__(self, u_anom: str = "u_anom", v_anom: str = "v_anom"):
        self.u_anom = u_anom
        self.v_anom = v_anom

    def _apply(self, ds):
        return _kinematics.eddy_kinetic_energy(
            ds, u_anom=self.u_anom, v_anom=self.v_anom
        )

    def get_config(self) -> dict[str, Any]:
        return {"u_anom": self.u_anom, "v_anom": self.v_anom}


class BruntVaisalaFrequency(Operator):
    def __init__(
        self,
        density: str = "rho",
        depth: str = "depth",
        rho0: float = 1025.0,
        g: float | None = None,
        positive: str = "down",
    ):
        self.density = density
        self.depth = depth
        self.rho0 = rho0
        self.g = g
        self.positive = positive

    def _apply(self, ds):
        return _kinematics.brunt_vaisala_frequency(
            ds,
            density=self.density,
            depth=self.depth,
            rho0=self.rho0,
            g=self.g,
            positive=self.positive,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "density": self.density,
            "depth": self.depth,
            "rho0": self.rho0,
            "g": self.g,
            "positive": self.positive,
        }


class LapseRate(Operator):
    def __init__(
        self,
        temperature: str = "T",
        depth: str = "depth",
        positive: str = "down",
    ):
        self.temperature = temperature
        self.depth = depth
        self.positive = positive

    def _apply(self, ds):
        return _kinematics.lapse_rate(
            ds,
            temperature=self.temperature,
            depth=self.depth,
            positive=self.positive,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "depth": self.depth,
            "positive": self.positive,
        }


class MixedLayerDepth(Operator):
    def __init__(
        self,
        density: str = "rho",
        depth: str = "depth",
        reference_depth: float = 10.0,
        threshold: float = 0.03,
    ):
        self.density = density
        self.depth = depth
        self.reference_depth = reference_depth
        self.threshold = threshold

    def _apply(self, ds):
        return _kinematics.mixed_layer_depth(
            ds,
            density=self.density,
            depth=self.depth,
            reference_depth=self.reference_depth,
            threshold=self.threshold,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "density": self.density,
            "depth": self.depth,
            "reference_depth": self.reference_depth,
            "threshold": self.threshold,
        }


__all__ = [
    "AbsoluteVorticity",
    "Advection",
    "AgeostrophicVelocities",
    "BruntVaisalaFrequency",
    "CalculateSSHAlongtrack",
    "CoriolisNormalized",
    "CurvatureVorticity",
    "Divergence",
    "EddyKineticEnergy",
    "Enstrophy",
    "Frontogenesis",
    "GeostrophicVelocities",
    "HorizontalVelocityMagnitude",
    "KineticEnergy",
    "LapseRate",
    "MixedLayerDepth",
    "OkuboWeiss",
    "PotentialVorticityBarotropic",
    "RelativeVorticity",
    "ShearStrain",
    "ShearVorticity",
    "StrainMagnitude",
    "Streamfunction",
    "TensorStrain",
    "ValidateSSH",
    "ValidateVelocity",
    "VelocityMagnitude",
]
