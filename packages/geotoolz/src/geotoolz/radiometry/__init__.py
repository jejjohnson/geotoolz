"""Sensor-agnostic radiometric transforms.

A small palette of *generic* radiometric Operators — DN→radiance and
DN→reflectance scalar decodes, dtype casts, and display-prep
(min-max, percentile clip, gamma). They take user-supplied gain /
offset / scale constants rather than reading sensor metadata.

Solar-geometry-aware TOA conversion is covered by `RadianceToReflectance`
/ `ReflectanceToRadiance` (carrier-aware wrappers over
`georeader.reflectance.radiance_to_reflectance`), with `ComputeSZA`,
`EarthSunDistanceCorrection` and `IntegratedIrradiance` for the
individual geometry terms. Sensor presets that read these constants
from product metadata are not part of this module: pass the per-band
irradiance / acquisition date yourself.

Examples:
    Sentinel-2 L1C display pipeline::

        import geotoolz as gz

        s2_display = (
            gz.radiometry.ToFloat32()
            | gz.radiometry.DNToReflectance(scale=1e-4)
            | gz.radiometry.PercentileClip(lower=2, upper=98)
            | gz.radiometry.Gamma(gamma=1.2)
        )

        rgb = s2_display(s2_dn_geotensor)

    Per-band gain and offset (Landsat-style)::

        import numpy as np
        gains   = np.array([0.012, 0.013, 0.011, 0.009])
        offsets = np.array([-60.0, -61.0, -55.0, -45.0])
        op = gz.radiometry.DNToRadiance(gain=gains, offset=offsets)
        radiance = op(dn_geotensor)
"""

from __future__ import annotations

from geotoolz.radiometry._src.array import (
    bt_from_radiance,
    dn_to_radiance,
    dn_to_reflectance,
    dos1,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
    radiance_to_dn,
)
from geotoolz.radiometry._src.operators import (
    DOS1,
    ApplySRF,
    BTFromRadiance,
    ComputeSZA,
    DNToRadiance,
    DNToReflectance,
    EarthSunDistanceCorrection,
    Gamma,
    IntegratedIrradiance,
    MinMax,
    PercentileClip,
    RadianceToDN,
    RadianceToReflectance,
    ReflectanceToRadiance,
    SimpleAtmosphericCorrection,
    ToFloat32,
)


__all__ = [
    "DOS1",
    "ApplySRF",
    "BTFromRadiance",
    "ComputeSZA",
    "DNToRadiance",
    "DNToReflectance",
    "EarthSunDistanceCorrection",
    "Gamma",
    "IntegratedIrradiance",
    "MinMax",
    "PercentileClip",
    "RadianceToDN",
    "RadianceToReflectance",
    "ReflectanceToRadiance",
    "SimpleAtmosphericCorrection",
    "ToFloat32",
    "bt_from_radiance",
    "dn_to_radiance",
    "dn_to_reflectance",
    "dos1",
    "gamma_correct",
    "min_max_normalize",
    "percentile_clip",
    "radiance_to_dn",
]
