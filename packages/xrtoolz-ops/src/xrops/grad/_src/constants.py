"""Physical constants used by xrops.grad operators.

They are geophysix's: the radius of the IUGG mean sphere, WGS84's rotation
rate and standard gravity. The names stay here because ``xrops.grad``
re-exports them.
"""

from __future__ import annotations

from geophysix.constants import STANDARD
from geophysix.earth import SPHERE, WGS84


EARTH_RADIUS: float = SPHERE.semi_major
"""Mean Earth radius in metres (IUGG R₁ = 6,371,008.77 m)."""

OMEGA: float = WGS84.rotation_rate
"""Earth's rotation rate in rad/s (WGS84, 7.292115e-5)."""

GRAVITY: float = STANDARD.standard_gravity
"""Standard gravitational acceleration in m/s² (9.80665)."""
