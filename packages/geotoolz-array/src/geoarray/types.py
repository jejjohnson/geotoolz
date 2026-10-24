"""`geoarray.types` — readable names for the common carrier shapes.

A readable type is a name, a dims pattern and a geometry kind (GA-T5), not
a subclass: any carrier whose dims and kind fit matches it (`matches`),
and `check_dims` checks annotated arguments and returns at the call.

| Type | Kind and dims | Typical data |
| --- | --- | --- |
| `GeoMap` | grid ``y x`` | a matched-filter score, a mask, an XCH₄ map |
| `GeoImage` | grid ``band y x`` | one multispectral scene |
| `GeoMapSeries` | grid ``time y x`` | maps over time: SLA analyses, a daily MLD cube |
| `GeoImageSeries` | grid ``time band y x`` | frames over time: a Sentinel-2 stack |
| `GeoVolume` | grid ``level y x`` | gridded profiles on shared levels (EN4) |
| `GeoPatches` | grid ``patch band y x`` | a geopatcher batch |
| `GeoSwathImage` | swath ``band row col`` | EMIT or AVIRIS L1B in sensor geometry |
| `GeoSwathProfile` | swath ``level row col`` | TROPOMI averaging kernels and priors |
| `GeoTimeSeries` | points ``time variable obs`` | stations over shared times |
| `GeoTracks` | points ``variable obs`` + time | along-track altimetry, ship tracks |
| `GeoProfiles` | points ``variable obs`` + z | Argo, CTD, gliders |

A time series is not a video: an image series is frames on a grid; a time
series is values at fixed locations.

Each alias is ``Annotated[<carrier>, ReadableType(...)]``, so a type
checker reads ``GeoImage`` as `GeoArray`.

Example:
    >>> import numpy as np
    >>> from geoarray import GeoArray
    >>> from geoarray.types import GeoImage, check_dims
    >>> @check_dims
    ... def brightest(scene: GeoImage) -> GeoArray["y x"]:
    ...     return scene.isel(band=0)
    >>> t = (1, 0, 0, 0, -1, 0)
    >>> scene = GeoArray(np.zeros((2, 4, 4)), t, None, dims="band y x")
    >>> brightest(scene).dims
    ('y', 'x')
"""

from __future__ import annotations

from typing import Annotated

from geoarray._src.array import GeoArray
from geoarray._src.carriers import GeoPoints, GeoSwath
from geoarray._src.types import (
    DimsError,
    Kind,
    ReadableType,
    check_dims,
    kind_of,
    matches,
    register_kind_reader,
)


__all__ = [
    "DimsError",
    "GeoImage",
    "GeoImageSeries",
    "GeoMap",
    "GeoMapSeries",
    "GeoPatches",
    "GeoProfiles",
    "GeoSwathImage",
    "GeoSwathProfile",
    "GeoTimeSeries",
    "GeoTracks",
    "GeoVolume",
    "Kind",
    "ReadableType",
    "check_dims",
    "kind_of",
    "matches",
    "register_kind_reader",
]

GeoMap = Annotated[GeoArray, ReadableType("GeoMap", "y x", "grid")]
GeoImage = Annotated[GeoArray, ReadableType("GeoImage", "band y x", "grid")]
GeoMapSeries = Annotated[GeoArray, ReadableType("GeoMapSeries", "time y x", "grid")]
GeoImageSeries = Annotated[
    GeoArray, ReadableType("GeoImageSeries", "time band y x", "grid")
]
GeoVolume = Annotated[GeoArray, ReadableType("GeoVolume", "level y x", "grid")]
GeoPatches = Annotated[GeoArray, ReadableType("GeoPatches", "patch band y x", "grid")]
GeoSwathImage = Annotated[
    GeoSwath, ReadableType("GeoSwathImage", "band row col", "swath")
]
GeoSwathProfile = Annotated[
    GeoSwath, ReadableType("GeoSwathProfile", "level row col", "swath")
]
GeoTimeSeries = Annotated[
    GeoPoints, ReadableType("GeoTimeSeries", "time variable obs", "points")
]
GeoTracks = Annotated[
    GeoPoints, ReadableType("GeoTracks", "variable obs", "points", ("time",))
]
GeoProfiles = Annotated[
    GeoPoints, ReadableType("GeoProfiles", "variable obs", "points", ("z",))
]
