# Oceanography

Physical oceanography diagnostics on gridded fields. The Coriolis
parameter $f=2\Omega\sin\phi$ sets the rotation; geostrophic balance gives
the surface velocity from sea-surface height $\eta$:

$$
u_g=-\frac{g}{f}\frac{\partial\eta}{\partial y},\qquad
v_g=\;\frac{g}{f}\frac{\partial\eta}{\partial x}.
$$

Vorticity, strain, kinetic energy, the Okubo–Weiss parameter, and
stratification (Brunt–Väisälä frequency, mixed-layer depth) follow from the
velocity and density fields.

## Cartesian / β-plane grids

By default every horizontal diagnostic works on a lon/lat grid in degrees
(`dims=("lon", "lat")`, `geometry="spherical"`) and derives $f$ from
latitude. Idealised model output on an $f$- or β-plane — `x`/`y` in metres,
no lon/lat — passes the dims, the geometry and an explicit Coriolis
parameter instead:

```python
from xrops.ocn import GeostrophicVelocities, OkuboWeiss, relative_vorticity

cart = {"dims": ("x", "y"), "geometry": "cartesian"}  # or "rectilinear"

zeta = relative_vorticity(ds, "u", "v", **cart)
ug = GeostrophicVelocities("eta", **cart, f=(1e-4, 1.6e-11))(ds)
ow = OkuboWeiss("u", "v", **cart)(ds)
```

`f=` accepts a scalar $f_0$ (f-plane), an `(f0, beta)` tuple for the
β-plane $f = f_0 + \beta\,(y - y_0)$ with $y_0$ the centre of the `y`
coordinate, `(f0, beta, y0)` to pin $y_0$, or a DataArray used as-is. It is
required on non-spherical grids, where there is no latitude to derive it
from. Fields must be collocated on one grid; destagger C-grid output before
calling these diagnostics.

## Rotation & geostrophy

::: xrops.ocn.operators.CoriolisNormalized

::: xrops.ocn.operators.Streamfunction

::: xrops.ocn.operators.GeostrophicVelocities

::: xrops.ocn.operators.AgeostrophicVelocities

## Vorticity & strain

::: xrops.ocn.operators.RelativeVorticity

::: xrops.ocn.operators.AbsoluteVorticity

::: xrops.ocn.operators.ShearVorticity

::: xrops.ocn.operators.CurvatureVorticity

::: xrops.ocn.operators.Divergence

::: xrops.ocn.operators.ShearStrain

::: xrops.ocn.operators.TensorStrain

::: xrops.ocn.operators.StrainMagnitude

::: xrops.ocn.operators.OkuboWeiss

::: xrops.ocn.operators.Enstrophy

::: xrops.ocn.operators.PotentialVorticityBarotropic

## Energy & transport

::: xrops.ocn.operators.KineticEnergy

::: xrops.ocn.operators.EddyKineticEnergy

::: xrops.ocn.operators.VelocityMagnitude

::: xrops.ocn.operators.HorizontalVelocityMagnitude

::: xrops.ocn.operators.Advection

::: xrops.ocn.operators.Frontogenesis

## Stratification

::: xrops.ocn.operators.BruntVaisalaFrequency

::: xrops.ocn.operators.MixedLayerDepth

::: xrops.ocn.operators.LapseRate

## Sea-surface height

::: xrops.ocn.operators.CalculateSSHAlongtrack

## Validation

::: xrops.ocn.operators.ValidateSSH

::: xrops.ocn.operators.ValidateVelocity

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.ocn.coriolis_parameter

::: xrops.ocn.coriolis_normalized

::: xrops.ocn.streamfunction

::: xrops.ocn.geostrophic_velocities

::: xrops.ocn.ageostrophic_velocities

::: xrops.ocn.relative_vorticity

::: xrops.ocn.absolute_vorticity

::: xrops.ocn.shear_vorticity

::: xrops.ocn.curvature_vorticity

::: xrops.ocn.divergence

::: xrops.ocn.shear_strain

::: xrops.ocn.tensor_strain

::: xrops.ocn.strain_magnitude

::: xrops.ocn.okubo_weiss

::: xrops.ocn.enstrophy

::: xrops.ocn.potential_vorticity_barotropic

::: xrops.ocn.kinetic_energy

::: xrops.ocn.eddy_kinetic_energy

::: xrops.ocn.velocity_magnitude

::: xrops.ocn.horizontal_velocity_magnitude

::: xrops.ocn.advection

::: xrops.ocn.frontogenesis

::: xrops.ocn.brunt_vaisala_frequency

::: xrops.ocn.mixed_layer_depth

::: xrops.ocn.lapse_rate

::: xrops.ocn.density_from_ts

::: xrops.ocn.calculate_ssh_alongtrack

::: xrops.ocn.calculate_ssh_unfiltered

::: xrops.ocn.validate_ssh

::: xrops.ocn.validate_velocity
