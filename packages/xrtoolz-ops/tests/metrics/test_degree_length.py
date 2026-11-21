"""xrops' km per degree is one degree on geophysix's mean sphere (geotoolz#755)."""

from __future__ import annotations

import math

from geophysix.earth import SPHERE

from xrops.mask._src.regimes import KM_PER_DEGREE_AT_EQUATOR
from xrops.metrics._src.spectral import _KM_PER_DEGREE


def test_km_per_degree_is_the_mean_sphere():
    expected = SPHERE.semi_major * math.pi / 180.0 / 1e3
    assert math.isclose(KM_PER_DEGREE_AT_EQUATOR, expected, rel_tol=1e-15)
    assert math.isclose(_KM_PER_DEGREE, expected, rel_tol=1e-15)
    # 111.195 km, 0.18 % more than the 111.0 km it replaces
    assert math.isclose(expected, 111.195, rel_tol=1e-6)
