"""`geoops.einx` — universal tensor notation over GeoArray carriers.

Wraps [einx](https://github.com/fferflo/einx) so einstein-notation
array ops compose inside operator pipelines while the spatial-survival
rule decides what happens to geospatial metadata:

- pattern keeps the trailing bare ``y x`` axes intact → ``GeoArray``
  in, ``GeoArray`` out (transform / CRS / fill preserved);
- pattern consumes, moves, or recomposes a spatial axis → plain
  ``np.ndarray`` out;
- `SpatialPool` is the deliberate exception: it *rescales* the
  transform to the pooled grid.

einx is a core dependency — the same notation also powers the internal
linear algebra of the Tier-A primitives (covariance/Gram products,
matched-filter scoring, PCA projections, static channel-order flips) and
the shared ``(c, h, w) -> (n, c)`` pixel flattening
(``geoops._src.samples.cube_to_samples``).
The pattern-analysis helpers (`spatial_survives`, `output_axes`) are
pure string processing and live in ``geoops.einx._src.array``.
"""

from __future__ import annotations

from geoops.einx._src.array import output_axes, spatial_survives
from geoops.einx._src.operators import (
    CHWtoHWC,
    Einx,
    HWCtoCHW,
    PerBandReduce,
    SpatialPool,
)


__all__ = [
    "CHWtoHWC",
    "Einx",
    "HWCtoCHW",
    "PerBandReduce",
    "SpatialPool",
    "output_axes",
    "spatial_survives",
]
