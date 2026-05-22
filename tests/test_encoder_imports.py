"""Import-surface tests for :mod:`xrtoolz.transforms.encoders` (D8).

Guards against regressions in the encoder taxonomy after the move from
``xrtoolz.geo.encoders`` to ``xrtoolz.transforms.encoders``.
"""

from __future__ import annotations

import importlib
import warnings

import pytest


# ---- New canonical surface ------------------------------------------------


def test_transforms_encoders_root_exposes_all_names():
    from xrtoolz.transforms.encoders import (
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
    from xrtoolz.transforms.encoders.coord_space import (
        lat_90_to_180,
        lat_180_to_90,
        lon_180_to_360,
        lon_360_to_180,
    )

    assert callable(lon_360_to_180)
    _ = (lat_90_to_180, lat_180_to_90, lon_180_to_360)


def test_coord_time_submodule_imports():
    from xrtoolz.transforms.encoders.coord_time import (
        encode_time_cyclical,
        encode_time_ordinal,
        time_rescale,
        time_unrescale,
    )

    assert callable(time_rescale)
    _ = (encode_time_cyclical, encode_time_ordinal, time_unrescale)


def test_basis_submodule_imports():
    from xrtoolz.transforms.encoders.basis import (
        cyclical_encode,
        fourier_features,
        positional_encoding,
        random_fourier_features,
    )

    assert callable(cyclical_encode)
    _ = (fourier_features, positional_encoding, random_fourier_features)


def test_root_and_submodule_paths_resolve_to_same_object():
    from xrtoolz.transforms.encoders import (
        cyclical_encode as root_ce,
        lon_360_to_180 as root_lon,
        time_rescale as root_t,
    )
    from xrtoolz.transforms.encoders.basis import cyclical_encode as sub_ce
    from xrtoolz.transforms.encoders.coord_space import (
        lon_360_to_180 as sub_lon,
    )
    from xrtoolz.transforms.encoders.coord_time import time_rescale as sub_t

    assert root_ce is sub_ce
    assert root_lon is sub_lon
    assert root_t is sub_t


# ---- Legacy deprecation surface ------------------------------------------


_DEPRECATED_ENCODER_NAMES = (
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
)


@pytest.mark.parametrize("name", _DEPRECATED_ENCODER_NAMES)
def test_legacy_geo_encoder_imports_warn_but_resolve(name):
    """Each legacy ``from xrtoolz.geo import <name>`` must keep working
    for one release with a :class:`DeprecationWarning` that points at
    :mod:`xrtoolz.transforms.encoders`.
    """
    import xrtoolz.geo as geo

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = getattr(geo, name)

    assert callable(obj)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, f"expected DeprecationWarning on legacy xrtoolz.geo.{name}"
    assert "xrtoolz.transforms.encoders" in str(deprecations[0].message)


def test_plain_import_xrtoolz_geo_is_silent_for_encoders():
    """Reloading the package itself (without naming a moved encoder) must
    not emit a :class:`DeprecationWarning` — the warning is per-name.
    """
    import xrtoolz.geo

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(xrtoolz.geo)

    encoder_deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "xrtoolz.transforms.encoders" in str(w.message)
    ]
    assert not encoder_deprecations, (
        f"plain import of xrtoolz.geo emitted unexpected encoder deprecations: "
        f"{[str(w.message) for w in encoder_deprecations]}"
    )
