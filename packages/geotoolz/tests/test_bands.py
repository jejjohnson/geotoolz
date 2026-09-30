"""Tests for the package-wide band-name resolver (``geotoolz._src.bands``, #150)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _helpers import toy_geotensor
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz._src.bands import (
    DEFAULT_BAND_KEYS,
    SENTINEL2_L2A_BANDS,
    band_names,
    resolve_band,
    resolve_bands,
)


NAMES = ["B02", "B03", "B04", "B08"]
# Same names, different order: a family that consulted ``descriptions``
# first would resolve "B04" to position 1 instead of 2.
DESCRIPTIONS = ["B08", "B04", "B03", "B02"]


def _conflicting() -> GeoTensor:
    """A tensor whose ``band_names`` and ``descriptions`` disagree."""
    rng = np.random.default_rng(0)
    values = rng.integers(1, 1000, size=(4, 5, 5)).astype(np.uint16)
    return toy_geotensor(
        values,
        fill_value_default=0,
        attrs={"band_names": list(NAMES), "descriptions": list(DESCRIPTIONS)},
    )


def _values(x: Any) -> np.ndarray:
    return np.asarray(x)


# Each case: (family, operator by name, the same operator by band_names positions).
_B04, _B08 = NAMES.index("B04"), NAMES.index("B08")
CASES: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
    (
        "indices",
        lambda: gz.indices.NormalizedDifference(a="B08", b="B04"),
        lambda: gz.indices.NormalizedDifference(a=_B08, b=_B04),
    ),
    (
        "spectral.SelectBands",
        lambda: gz.spectral.SelectBands(indexes=["B04", "B08"]),
        lambda: gz.spectral.SelectBands(indexes=[_B04, _B08]),
    ),
    (
        "spectral.BandRatio",
        lambda: gz.spectral.BandRatio(numerator="B08", denominator="B04"),
        lambda: gz.spectral.BandRatio(numerator=_B08, denominator=_B04),
    ),
    (
        "qa",
        lambda: gz.qa.MaskClouds(qa_band="B04", bits=[0]),
        lambda: gz.qa.MaskClouds(qa_band=_B04, bits=[0]),
    ),
    (
        "augment",
        lambda: gz.augment.BandJitter(groups={"g": ["B04", "B08"]}, seed=3),
        lambda: gz.augment.BandJitter(groups={"g": [_B04, _B08]}, seed=3),
    ),
    (
        "viz",
        lambda: gz.viz.Composite(bands=["B04", "B08"]),
        lambda: gz.viz.Composite(bands=[_B04, _B08]),
    ),
    (
        "viz.TrueColor",
        lambda: gz.viz.TrueColor(red="B04", green="B03", blue="B02"),
        lambda: gz.viz.TrueColor(red=_B04, green=1, blue=0),
    ),
    (
        "plume",
        lambda: gz.plume.SBMP(swir1="B08", swir2="B04"),
        lambda: gz.plume.SBMP(swir1=_B08, swir2=_B04),
    ),
]


@pytest.mark.parametrize(
    ("family", "by_name", "by_index"), CASES, ids=[c[0] for c in CASES]
)
def test_same_tensor_resolves_identically_across_families(
    family: str, by_name: Callable[[], Any], by_index: Callable[[], Any]
) -> None:
    """``"B04"`` is ``band_names[2]`` in every family, whatever ``descriptions`` say."""
    gt = _conflicting()
    np.testing.assert_array_equal(
        _values(by_name()(gt)), _values(by_index()(gt)), err_msg=family
    )


def test_band_jitter_by_name_leaves_other_bands_untouched() -> None:
    gt = _conflicting()
    out = np.asarray(gz.augment.BandJitter(groups={"g": ["B04", "B08"]}, seed=3)(gt))
    arr = np.asarray(gt)
    np.testing.assert_array_equal(out[:2], arr[:2])


def test_default_key_order() -> None:
    assert DEFAULT_BAND_KEYS == ("band_names", "descriptions", "bands")


def test_resolve_band_falls_through_keys() -> None:
    gt = toy_geotensor(
        np.ones((3, 2, 2), dtype=np.float32),
        attrs={"band_names": ["a", "b", "c"], "bands": ["x", "y", "z"]},
    )
    assert resolve_band(gt, "b") == 1
    assert resolve_band(gt, "z") == 2  # missing from band_names -> next key
    with pytest.raises(ValueError, match="'q'"):
        resolve_band(gt, "q")


def test_resolve_band_accepts_numpy_integers_and_rejects_others() -> None:
    arr = np.ones((3, 2, 2))
    got = resolve_band(arr, np.int64(2))
    assert got == 2 and type(got) is int
    assert resolve_bands(arr, [np.int32(1), 0]) == [1, 0]
    with pytest.raises(TypeError):
        resolve_band(arr, True)
    with pytest.raises(TypeError):
        resolve_band(arr, 1.0)


def test_resolve_band_mapping_form() -> None:
    gt = toy_geotensor(
        np.ones((2, 2, 2), dtype=np.uint16),
        attrs={"band_names": {"QA60": 1, "B04": 0}},
    )
    assert resolve_band(gt, "QA60") == 1
    assert band_names(gt) == ["B04", "QA60"]
    np.testing.assert_array_equal(
        np.asarray(gz.qa.MaskClouds(qa_band="QA60", bits=[0])(gt)),
        np.asarray(gz.qa.MaskClouds(qa_band=1, bits=[0])(gt)),
    )


def test_string_ref_on_plain_array() -> None:
    arr = np.ones((12, 2, 2))
    with pytest.raises(TypeError, match="band-name metadata"):
        resolve_band(arr, "B11")
    assert resolve_band(arr, "B11", fallback=SENTINEL2_L2A_BANDS) == 10
    with pytest.raises(ValueError, match="'B10'"):
        resolve_band(arr, "B10", fallback=SENTINEL2_L2A_BANDS)


def test_band_names_helper() -> None:
    assert band_names(_conflicting()) == NAMES
    assert band_names(np.ones((2, 2, 2))) is None
    assert band_names(toy_geotensor(np.ones((2, 2, 2)), attrs={})) is None
    only_desc = toy_geotensor(np.ones((2, 2, 2)), attrs={"descriptions": ("r", "n")})
    assert band_names(only_desc) == ["r", "n"]


def test_sparse_mapping_keeps_declared_positions() -> None:
    """``{"red": 1, "nir": 2}`` must not compact to band 0 / band 1."""
    values = np.stack([np.full((2, 2), v, dtype=np.float32) for v in (10, 20, 30, 40)])
    gt = toy_geotensor(values, attrs={"band_names": {"red": 1, "nir": 2}})
    assert band_names(gt) == [None, "red", "nir", None]

    out = gz.spectral.BandMath(expression="nir - red")(gt)
    np.testing.assert_array_equal(np.squeeze(_values(out)), np.full((2, 2), 10.0))

    parts = gz.spectral.SplitBands()(gt)
    assert len(parts) == 4

    aliases = toy_geotensor(values, attrs={"band_names": {"red": 1, "B04": 1}})
    assert band_names(aliases) == [None, "red", None, None]


def test_readers_write_only_band_names() -> None:
    from geotoolz.readers.toy_sensor import Reader

    gt = Reader("toy.tif").load()
    assert set(gt.attrs or {}) >= {"band_names"}
    assert "bands" not in (gt.attrs or {})
