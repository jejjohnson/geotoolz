"""Public encoder surface — coord-space, coord-time, and basis encodings.

Re-exports the pure functions from :mod:`xr_toolz.transforms._src.encoders`.
Submodules :mod:`coord_space`, :mod:`coord_time`, and :mod:`basis` are
also importable as ``xr_toolz.transforms.encoders.<sub>`` for callers
who want a narrower import.
"""

from __future__ import annotations

from . import basis, coord_space, coord_time
from .basis import (
    cyclical_encode,
    fourier_features,
    positional_encoding,
    random_fourier_features,
)
from .coord_space import (
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
)
from .coord_time import (
    encode_time_cyclical,
    encode_time_ordinal,
    time_rescale,
    time_unrescale,
)


__all__ = [
    "basis",
    "coord_space",
    "coord_time",
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
