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

## Rotation & geostrophy

::: xrtoolz.ocn.operators.CoriolisNormalized

::: xrtoolz.ocn.operators.Streamfunction

::: xrtoolz.ocn.operators.GeostrophicVelocities

::: xrtoolz.ocn.operators.AgeostrophicVelocities

## Vorticity & strain

::: xrtoolz.ocn.operators.RelativeVorticity

::: xrtoolz.ocn.operators.AbsoluteVorticity

::: xrtoolz.ocn.operators.ShearVorticity

::: xrtoolz.ocn.operators.CurvatureVorticity

::: xrtoolz.ocn.operators.Divergence

::: xrtoolz.ocn.operators.ShearStrain

::: xrtoolz.ocn.operators.TensorStrain

::: xrtoolz.ocn.operators.StrainMagnitude

::: xrtoolz.ocn.operators.OkuboWeiss

::: xrtoolz.ocn.operators.Enstrophy

::: xrtoolz.ocn.operators.PotentialVorticityBarotropic

## Energy & transport

::: xrtoolz.ocn.operators.KineticEnergy

::: xrtoolz.ocn.operators.EddyKineticEnergy

::: xrtoolz.ocn.operators.VelocityMagnitude

::: xrtoolz.ocn.operators.HorizontalVelocityMagnitude

::: xrtoolz.ocn.operators.Advection

::: xrtoolz.ocn.operators.Frontogenesis

## Stratification

::: xrtoolz.ocn.operators.BruntVaisalaFrequency

::: xrtoolz.ocn.operators.MixedLayerDepth

::: xrtoolz.ocn.operators.LapseRate

## Sea-surface height

::: xrtoolz.ocn.operators.CalculateSSHAlongtrack

## Validation

::: xrtoolz.ocn.operators.ValidateSSH

::: xrtoolz.ocn.operators.ValidateVelocity

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.ocn.coriolis_parameter

::: xrtoolz.ocn.coriolis_normalized

::: xrtoolz.ocn.streamfunction

::: xrtoolz.ocn.geostrophic_velocities

::: xrtoolz.ocn.ageostrophic_velocities

::: xrtoolz.ocn.relative_vorticity

::: xrtoolz.ocn.absolute_vorticity

::: xrtoolz.ocn.shear_vorticity

::: xrtoolz.ocn.curvature_vorticity

::: xrtoolz.ocn.divergence

::: xrtoolz.ocn.shear_strain

::: xrtoolz.ocn.tensor_strain

::: xrtoolz.ocn.strain_magnitude

::: xrtoolz.ocn.okubo_weiss

::: xrtoolz.ocn.enstrophy

::: xrtoolz.ocn.potential_vorticity_barotropic

::: xrtoolz.ocn.kinetic_energy

::: xrtoolz.ocn.eddy_kinetic_energy

::: xrtoolz.ocn.velocity_magnitude

::: xrtoolz.ocn.horizontal_velocity_magnitude

::: xrtoolz.ocn.advection

::: xrtoolz.ocn.frontogenesis

::: xrtoolz.ocn.brunt_vaisala_frequency

::: xrtoolz.ocn.mixed_layer_depth

::: xrtoolz.ocn.lapse_rate

::: xrtoolz.ocn.density_from_ts

::: xrtoolz.ocn.calculate_ssh_alongtrack

::: xrtoolz.ocn.calculate_ssh_unfiltered

::: xrtoolz.ocn.validate_ssh

::: xrtoolz.ocn.validate_velocity
