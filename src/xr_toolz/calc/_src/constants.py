"""Physical constants used by xr_toolz.calc operators.

These replace the equivalent ``metpy.constants`` lookups so the package
can drop the metpy dependency.
"""

from __future__ import annotations


EARTH_RADIUS: float = 6_371_000.0
"""Mean Earth radius in metres."""

OMEGA: float = 7.2921159e-5
"""Earth's rotation rate in rad/s."""

GRAVITY: float = 9.80665
"""Standard gravitational acceleration in m/s²."""
