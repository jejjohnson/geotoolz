"""Deep-import shim for the encoders that moved to
:mod:`xr_toolz.transforms._src.encoders` (D8).

The public-surface :mod:`xr_toolz.geo` module emits a
:class:`DeprecationWarning` when these names are accessed via
``from xr_toolz.geo import <name>``. The functions themselves now live
under :mod:`xr_toolz.transforms.encoders`; this module is kept only so
that existing deep imports such as
``from xr_toolz.geo._src.encoders import cyclical_encode`` continue to
resolve for one release while downstream code migrates.
"""

from __future__ import annotations

from xr_toolz.transforms._src.encoders import (
    cyclical_encode,
    encode_time_cyclical,
    encode_time_ordinal,
    fourier_features,
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
    positional_encoding,
    random_fourier_features,
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
