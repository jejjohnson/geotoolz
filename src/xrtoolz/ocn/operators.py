"""Layer-1 ``Operator`` wrappers around :mod:`xrtoolz.ocn._src`.

Each operator is a thin, configurable adapter over a pure-function physics
primitive: ``__init__`` captures the variable / coordinate names, and
``_apply`` forwards the dataset to the primitive. Operators take an
:class:`xarray.Dataset` and return it augmented with the computed
diagnostic. See :mod:`xrtoolz.ocn._src.kinematics`,
:mod:`xrtoolz.ocn._src.ssh`, and :mod:`xrtoolz.ocn._src.validation` for the
underlying implementations.
"""

from __future__ import annotations

from typing import Any

from xrtoolz._operator import Operator
from xrtoolz.ocn._src import (
    kinematics as _kinematics,
    ssh as _ssh,
    validation as _validation,
)


# ---------- validation -----------------------------------------------------


class ValidateSSH(Operator):
    """Validate and CF-normalise a sea-surface-height field.

    Checks that ``variable`` is present and finite and applies the canonical
    SSH metadata (standard name, units), raising on malformed input.

    Args:
        variable: Name of the sea-surface-height variable to validate.

    Returns:
        The input dataset with ``variable`` validated and CF-tagged.
    """

    def __init__(self, variable: str = "ssh"):
        self.variable = variable

    def _apply(self, ds):
        return _validation.validate_ssh(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class ValidateVelocity(Operator):
    """Validate a horizontal velocity pair ``(u, v)``.

    Ensures both components are present, finite, and consistently shaped, and
    applies the canonical CF velocity metadata.

    Args:
        u: Name of the eastward (zonal) velocity variable.
        v: Name of the northward (meridional) velocity variable.

    Returns:
        The input dataset with ``u`` and ``v`` validated and CF-tagged.
    """

    def __init__(self, u: str = "u", v: str = "v"):
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _validation.validate_velocity(ds, u=self.u, v=self.v)

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v}


# ---------- SSH composition ------------------------------------------------


class CalculateSSHAlongtrack(Operator):
    """Compose along-track SSH from its altimetry components.

    Reconstructs sea-surface height as ``ssh = sla + mdt − lwe``: filtered
    sea-level anomaly plus the mean dynamic topography, minus an optional
    long-wavelength-error correction.

    Args:
        variable: Name of the output SSH variable to create.
        sla: Name of the (filtered) sea-level-anomaly variable.
        mdt: Name of the mean-dynamic-topography variable.
        lwe: Name of the long-wavelength-error variable to subtract, or
            ``None`` to skip the correction.

    Returns:
        The input dataset with the composed ``variable`` (SSH) added.
    """

    def __init__(
        self,
        variable: str = "ssh",
        sla: str = "sla_filtered",
        mdt: str = "mdt",
        lwe: str | None = "lwe",
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
    """Geostrophic stream function ``ψ = (g / f₀) · η`` from SSH.

    Args:
        variable: Name of the sea-surface-height (η) variable.
        g: Gravitational acceleration (m s⁻²). ``None`` uses the package
            default.
        f0: Reference Coriolis parameter (s⁻¹). ``None`` derives ``f`` from
            latitude on the f-plane reference.

    Returns:
        The input dataset with a ``psi`` stream-function variable (m² s⁻¹).
    """

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
    """Geostrophic velocities ``(u_g, v_g)`` from a height field.

    Computes ``u_g = −(g/f) ∂η/∂y`` and ``v_g = (g/f) ∂η/∂x`` on the
    lon/lat sphere.

    Args:
        variable: Name of the sea-surface-height (η) variable.

    Returns:
        The input dataset with the geostrophic ``u`` and ``v`` velocity
        variables (m s⁻¹).
    """

    def __init__(self, variable: str = "ssh"):
        self.variable = variable

    def _apply(self, ds):
        return _kinematics.geostrophic_velocities(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class _UVOperator(Operator):
    """Shared ``__init__`` / ``get_config`` for ``(u, v)`` velocity operators.

    Args:
        u: Name of the eastward (zonal) velocity variable.
        v: Name of the northward (meridional) velocity variable.
    """

    def __init__(self, u: str = "u", v: str = "v"):
        self.u = u
        self.v = v

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v}


class KineticEnergy(_UVOperator):
    """Kinetic energy ``KE = ½ (u² + v²)``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``ke`` variable (m² s⁻²).
    """

    def _apply(self, ds):
        return _kinematics.kinetic_energy(ds, u=self.u, v=self.v)


class RelativeVorticity(_UVOperator):
    """Relative (vertical) vorticity ``ζ = ∂v/∂x − ∂u/∂y``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``vort_r`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.relative_vorticity(ds, u=self.u, v=self.v)


class AbsoluteVorticity(_UVOperator):
    """Absolute vorticity ``ζ_a = ζ + f`` (relative plus planetary).

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``vort_a`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.absolute_vorticity(ds, u=self.u, v=self.v)


class Divergence(_UVOperator):
    """Horizontal divergence ``∇·u = ∂u/∂x + ∂v/∂y`` on the lon/lat sphere.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``div`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.divergence(ds, u=self.u, v=self.v)


class ShearStrain(_UVOperator):
    """Shear strain rate ``S_s = ∂v/∂x + ∂u/∂y``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``shear_strain`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.shear_strain(ds, u=self.u, v=self.v)


class TensorStrain(_UVOperator):
    """Normal (tensor) strain rate ``S_n = ∂u/∂x − ∂v/∂y``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``tensor_strain`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.tensor_strain(ds, u=self.u, v=self.v)


class StrainMagnitude(_UVOperator):
    """Total strain-rate magnitude ``√(S_n² + S_s²)``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``strain`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.strain_magnitude(ds, u=self.u, v=self.v)


class OkuboWeiss(_UVOperator):
    """Okubo–Weiss parameter ``W = S_n² + S_s² − ζ²``.

    Negative ``W`` marks vorticity-dominated (eddy-core) regions; positive
    ``W`` marks strain-dominated regions.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with an ``ow`` variable (s⁻²).
    """

    def _apply(self, ds):
        return _kinematics.okubo_weiss(ds, u=self.u, v=self.v)


class Enstrophy(Operator):
    """Enstrophy ``½ ζ²`` from a relative-vorticity field.

    Args:
        variable: Name of the (relative) vorticity variable to square.

    Returns:
        The input dataset with an ``ens`` variable (s⁻²).
    """

    def __init__(self, variable: str = "vort_r"):
        self.variable = variable

    def _apply(self, ds):
        return _kinematics.enstrophy(ds, variable=self.variable)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable}


class CoriolisNormalized(Operator):
    """Normalise a field by the Coriolis parameter ``f₀``.

    Divides ``variable`` by ``f`` (e.g. to form the Rossby number ``ζ/f``).

    Args:
        variable: Name of the variable to normalise.
        f0: Reference Coriolis parameter (s⁻¹). ``None`` derives ``f`` from
            latitude.

    Returns:
        The input dataset with ``variable`` replaced by ``variable / f``.
    """

    def __init__(self, variable: str, f0: float | None = None):
        self.variable = variable
        self.f0 = f0

    def _apply(self, ds):
        return _kinematics.coriolis_normalized(ds, variable=self.variable, f0=self.f0)

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "f0": self.f0}


class AgeostrophicVelocities(Operator):
    """Ageostrophic velocities ``(u_a, v_a) = (u, v) − u_g(η)``.

    Subtracts the geostrophic velocities derived from ``variable`` (SSH)
    from the total velocities.

    Args:
        variable: Name of the sea-surface-height (η) variable.
        u: Name of the total eastward velocity variable.
        v: Name of the total northward velocity variable.

    Returns:
        The input dataset with ``u_a`` and ``v_a`` ageostrophic-velocity
        variables (m s⁻¹).
    """

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
    """Horizontal tracer advection ``−u·∇c`` on the lon/lat sphere.

    Args:
        scalar: Name of the advected tracer variable ``c``.
        components: Names of the velocity components ``(u, v)``.
        dims: Spatial dimension names ``(lon, lat)`` the gradient is taken
            over.

    Returns:
        The input dataset with a ``<scalar>_advection`` tendency variable
        added.
    """

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
    """Along-flow shear component of the relative vorticity.

    The part of ``ζ`` due to cross-stream changes in flow speed (as opposed
    to streamline curvature).

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``vort_shear`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.shear_vorticity(ds, u=self.u, v=self.v)


class CurvatureVorticity(_UVOperator):
    """Cross-flow curvature component of the relative vorticity.

    The part of ``ζ`` due to streamline curvature; ``vort_shear`` and
    ``vort_curv`` sum to the relative vorticity ``vort_r``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``vort_curv`` variable (s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.curvature_vorticity(ds, u=self.u, v=self.v)


class Frontogenesis(Operator):
    """Petterssen 2-D kinematic frontogenesis of a scalar on the sphere.

    The rate of change of the horizontal gradient magnitude of ``scalar``
    following the flow; positive values indicate front sharpening.

    Args:
        scalar: Name of the tracer variable (e.g. temperature).
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``<scalar>_frontogenesis`` variable.
    """

    def __init__(self, scalar: str, u: str = "u", v: str = "v"):
        self.scalar = scalar
        self.u = u
        self.v = v

    def _apply(self, ds):
        return _kinematics.frontogenesis(ds, scalar=self.scalar, u=self.u, v=self.v)

    def get_config(self) -> dict[str, Any]:
        return {"scalar": self.scalar, "u": self.u, "v": self.v}


class PotentialVorticityBarotropic(Operator):
    """Single-layer barotropic potential vorticity ``(ζ + f) / h``.

    Args:
        height: Name of the layer-thickness (h) variable.
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``pv_barotropic`` variable (m⁻¹ s⁻¹).
    """

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
    """Velocity magnitude ``|U| = √(u² + v² [+ w²])``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.
        w: Name of the vertical velocity variable, or ``None`` for the
            horizontal-only magnitude.

    Returns:
        The input dataset with a ``speed`` variable (m s⁻¹).
    """

    def __init__(self, u: str = "u", v: str = "v", w: str | None = None):
        self.u = u
        self.v = v
        self.w = w

    def _apply(self, ds):
        return _kinematics.velocity_magnitude(ds, u=self.u, v=self.v, w=self.w)

    def get_config(self) -> dict[str, Any]:
        return {"u": self.u, "v": self.v, "w": self.w}


class HorizontalVelocityMagnitude(_UVOperator):
    """Horizontal current speed ``√(u² + v²)``.

    Args:
        u: Name of the eastward velocity variable.
        v: Name of the northward velocity variable.

    Returns:
        The input dataset with a ``speed`` variable (m s⁻¹).
    """

    def _apply(self, ds):
        return _kinematics.horizontal_velocity_magnitude(ds, u=self.u, v=self.v)


class EddyKineticEnergy(Operator):
    """Eddy kinetic energy ``EKE = ½ (u'² + v'²)`` from velocity anomalies.

    Args:
        u_anom: Name of the eastward velocity-anomaly variable ``u'``.
        v_anom: Name of the northward velocity-anomaly variable ``v'``.

    Returns:
        The input dataset with an ``eke`` variable (m² s⁻²).
    """

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
    """Squared Brunt–Väisälä (buoyancy) frequency ``N² = −(g/ρ₀) ∂ρ/∂z``.

    Args:
        density: Name of the (potential) density variable ``ρ``.
        depth: Name of the vertical coordinate variable.
        rho0: Reference density ``ρ₀`` (kg m⁻³).
        g: Gravitational acceleration (m s⁻²). ``None`` uses the package
            default.
        positive: Orientation of ``depth`` — ``"down"`` (depth increases
            downward) or ``"up"``.

    Returns:
        The input dataset with an ``n_squared`` variable (s⁻²).
    """

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
    """Vertical lapse rate ``Γ = −∂T/∂z`` (z positive upward).

    Args:
        temperature: Name of the temperature variable.
        depth: Name of the vertical coordinate variable.
        positive: Orientation of ``depth`` — ``"down"`` (depth increases
            downward) or ``"up"``.

    Returns:
        The input dataset with a ``lapse_rate`` variable (K m⁻¹).
    """

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
    """Mixed-layer depth via the de Boyer Montégut density threshold.

    Finds the depth at which density first exceeds the ``reference_depth``
    value by ``threshold`` (kg m⁻³).

    Args:
        density: Name of the (potential) density variable.
        depth: Name of the vertical coordinate variable.
        reference_depth: Near-surface reference depth (m) the threshold is
            measured from.
        threshold: Density increase defining the mixed-layer base
            (kg m⁻³).

    Returns:
        The input dataset with an ``mld`` variable (m).
    """

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
