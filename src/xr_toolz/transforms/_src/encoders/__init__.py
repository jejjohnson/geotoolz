"""Encoder primitives — coordinate-space, coordinate-time, and basis encodings.

Pure functions organized by what they encode. Each submodule is import-stable;
the top-level :mod:`xr_toolz.transforms.encoders` shim re-exports the public
names for convenience.
"""

from __future__ import annotations

from xr_toolz.transforms._src.encoders.basis import (
    cyclical_encode,
    fourier_features,
    positional_encoding,
    random_fourier_features,
)
from xr_toolz.transforms._src.encoders.coord_space import (
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
)
from xr_toolz.transforms._src.encoders.coord_time import (
    encode_time_cyclical,
    encode_time_ordinal,
    time_rescale,
    time_unrescale,
)


__all__ = [
    "cyclical_encode",
    "encode_time_cyclical",
    "encode_time_ordinal",
    "fourier_features",
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
    "positional_encoding",
    "random_fourier_features",
    "time_rescale",
    "time_unrescale",
]
