"""The 0.2 → 0.3 deprecation path (#523): every 0.2 form works, with a warning."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import geoarray
from geoarray import FrozenAttrs, GeoArray
from geoarray.coords import SpectralAxis


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)


def test_rank_inferred_dims_warn_once_per_call_site() -> None:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        for _ in range(3):
            GeoArray(np.zeros((2, 4, 4)), T, None)
    assert len([w for w in seen if w.category is DeprecationWarning]) == 1
    assert seen[0].filename == __file__  # points at the caller


def test_band_names_are_read_into_a_spectral_axis() -> None:
    attrs = {"band_names": ["B11", "B12"], "wavelengths": [1610.0, 2190.0]}
    with pytest.warns(DeprecationWarning, match="SpectralAxis"):
        g = GeoArray(np.zeros((2, 4, 4)), T, None, attrs=attrs, dims="band y x")
    assert g.coord("band") == SpectralAxis(("B11", "B12"), (1610.0, 2190.0))
    assert g.attrs["band_names"] == ("B11", "B12")  # kept for 0.2 readers
    assert g.sel(band="B12").dims == ("y", "x")
    picked = g.isel(band=[1])
    assert picked.labels("band") == ("B12",) and picked.attrs["band_names"] == ("B12",)


def test_carried_frozen_attrs_do_not_warn() -> None:
    frozen = FrozenAttrs(band_names=["a", "b"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = GeoArray(np.zeros((2, 4, 4)), T, None, attrs=frozen, dims="band y x")
    assert g.labels("band") == ("a", "b")


@pytest.mark.parametrize(
    "attrs",
    [{"band_names": ["a", "a"]}, {"band_names": ["a"]}, {"band_names": "ab"}],
    ids=["repeated", "wrong length", "not a list"],
)
def test_unusable_band_names_stay_attrs_only(attrs: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = GeoArray(np.zeros((2, 4, 4)), T, None, attrs=attrs, dims="band y x")
    assert g.coord("band") is None


def test_explicit_coordinates_win() -> None:
    axis = SpectralAxis(("x1", "x2"))
    g = GeoArray(
        np.zeros((2, 4, 4)),
        T,
        None,
        attrs={"band_names": ["a", "b"]},
        dims="band y x",
        coords=(axis,),
    )
    assert g.coord("band") is axis


def test_the_0_2_forms_still_work() -> None:
    with pytest.warns(DeprecationWarning):
        g = GeoArray(np.zeros((2, 4, 4)), T, None)
    with pytest.warns(DeprecationWarning, match="dim="):
        assert geoarray.stack([g, g]).dims == ("time", "band", "y", "x")
    assert geoarray.concatenate([g, g]).shape == (4, 4, 4)
    with pytest.warns(DeprecationWarning, match="array-type"):
        assert GeoArray[np.ndarray] is GeoArray
    with pytest.warns(DeprecationWarning, match="dim="):
        assert g.expand_dims(0).dims == ("time", "band", "y", "x")
