"""Import-surface tests for :mod:`xrops.transforms.encoders` (D8).

Guards against regressions in the encoder taxonomy after the move from
the old geo package to ``xrops.transforms.encoders``.
"""

from __future__ import annotations

import importlib

import pytest


# ---- New canonical surface ------------------------------------------------


def test_transforms_encoders_root_exposes_all_names():
    from xrops.transforms.encoders import (
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

    assert callable(cyclical_encode)
    _ = (
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


def test_coord_space_submodule_imports():
    from xrops.transforms.encoders.coord_space import (
        lat_90_to_180,
        lat_180_to_90,
        lon_180_to_360,
        lon_360_to_180,
    )

    assert callable(lon_360_to_180)
    _ = (lat_90_to_180, lat_180_to_90, lon_180_to_360)


def test_coord_time_submodule_imports():
    from xrops.transforms.encoders.coord_time import (
        encode_time_cyclical,
        encode_time_ordinal,
        time_rescale,
        time_unrescale,
    )

    assert callable(time_rescale)
    _ = (encode_time_cyclical, encode_time_ordinal, time_unrescale)


def test_basis_submodule_imports():
    from xrops.transforms.encoders.basis import (
        cyclical_encode,
        fourier_features,
        positional_encoding,
        random_fourier_features,
    )

    assert callable(cyclical_encode)
    _ = (fourier_features, positional_encoding, random_fourier_features)


def test_root_and_submodule_paths_resolve_to_same_object():
    from xrops.transforms.encoders import (
        cyclical_encode as root_ce,
        lon_360_to_180 as root_lon,
        time_rescale as root_t,
    )
    from xrops.transforms.encoders.basis import cyclical_encode as sub_ce
    from xrops.transforms.encoders.coord_space import (
        lon_360_to_180 as sub_lon,
    )
    from xrops.transforms.encoders.coord_time import time_rescale as sub_t

    assert root_ce is sub_ce
    assert root_lon is sub_lon
    assert root_t is sub_t


# ---- The old xrtoolz.geo encoder paths are gone (no shims, #359) ---------


@pytest.mark.parametrize(
    "name",
    (
        "cyclical_encode",
        "fourier_features",
        "positional_encoding",
        "random_fourier_features",
        "lat_90_to_180",
        "lat_180_to_90",
        "lon_180_to_360",
        "lon_360_to_180",
        "encode_time_cyclical",
        "encode_time_ordinal",
        "time_rescale",
        "time_unrescale",
    ),
)
def test_encoders_are_not_reexported_from_geo(name):
    """Encoders live only in :mod:`xrops.transforms.encoders`."""
    import xrops.geom as geo

    assert not hasattr(geo, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xrops.geom._src.encoders")
