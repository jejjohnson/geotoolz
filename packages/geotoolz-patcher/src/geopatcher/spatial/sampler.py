"""`geopatcher.spatial.sampler` — Where patch anchors go.

`RegularStride` (a grid), `JitteredStride`, `Random`, `PoissonDisk`,
`Explicit` anchors, `ExplicitCoords` (CRS coordinates), `AlongTrack`
(a swath track); `Sampler` is the base. `IncompleteScanConfiguration` is
raised by ``check_full_scan=True`` when a stride would drop pixels.
"""

from __future__ import annotations

from geopatcher._src.exceptions import (
    IncompleteScanConfiguration,
)
from geopatcher._src.spatial.sampler import (
    AlongTrack,
    Explicit,
    ExplicitCoords,
    JitteredStride,
    PoissonDisk,
    Random,
    RegularStride,
    Sampler,
)


__all__ = [
    "AlongTrack",
    "Explicit",
    "ExplicitCoords",
    "IncompleteScanConfiguration",
    "JitteredStride",
    "PoissonDisk",
    "Random",
    "RegularStride",
    "Sampler",
]
