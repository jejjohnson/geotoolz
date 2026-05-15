"""Sensor-agnostic radiometric transforms.

A small palette of *generic* radiometric Operators — DN→radiance and
DN→reflectance scalar decodes, dtype casts, and display-prep
(min-max, percentile clip, gamma). They take user-supplied gain /
offset / scale constants rather than reading sensor metadata.

Sensor-specific TOA pipelines that need full solar-geometry awareness
(Sentinel-2 ESA processor, Landsat USGS processor, MODIS, EMIT) belong
in the v0.4 ``geotoolz.presets.*`` modules that will wrap
`georeader.reflectance.radiance_to_reflectance`. For now, users with
those needs call georeader directly and wrap the result in
``gz.Lambda(...)`` if they need it inside a `Sequential`.

Examples:
    Sentinel-2 L1C display pipeline::

        import geotoolz as gz

        s2_display = (
            gz.radiometry.ToFloat32()
            | gz.radiometry.DNToReflectance(scale=1e-4)
            | gz.radiometry.PercentileClip(p_min=2, p_max=98)
            | gz.radiometry.Gamma(g=1.2)
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
    dn_to_radiance,
    dn_to_reflectance,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
)
from geotoolz.radiometry._src.operators import (
    DNToRadiance,
    DNToReflectance,
    Gamma,
    MinMax,
    PercentileClip,
    ToFloat32,
)


__all__ = [
    "DNToRadiance",
    "DNToReflectance",
    "Gamma",
    "MinMax",
    "PercentileClip",
    "ToFloat32",
    "dn_to_radiance",
    "dn_to_reflectance",
    "gamma_correct",
    "min_max_normalize",
    "percentile_clip",
]
