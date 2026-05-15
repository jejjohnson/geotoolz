"""Tests for `geotoolz.cloud`."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.cloud import (
    SCL,
    SCL_CLOUDS,
    SCL_INVALID,
    SCL_LAND,
    ApplyMask,
    MaskFromQABits,
    MaskFromSCL,
    MaskValid,
    apply_mask,
    mask_from_qa_bits,
    mask_from_scl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_geotensor(values: np.ndarray, fill: int = -9999) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=fill,
    )


@pytest.fixture
def s2_l2a_stack() -> GeoTensor:
    """Synthetic Sentinel-2 L2A stack: (4 reflectance bands + 1 SCL) x 4 x 4.

    The SCL band has a quadrant of CLOUD_HIGH_PROBABILITY (=9), a
    quadrant of VEGETATION (=4), a quadrant of WATER (=6), and a
    quadrant of NO_DATA (=0).
    """
    rng = np.random.default_rng(0)
    reflectance = rng.uniform(0.05, 0.6, size=(4, 4, 4)).astype(np.float32)
    scl = np.array(
        [
            [9, 9, 4, 4],
            [9, 9, 4, 4],
            [6, 6, 0, 0],
            [6, 6, 0, 0],
        ],
        dtype=np.uint8,
    )
    stack = np.concatenate([reflectance, scl[None, ...].astype(np.float32)], axis=0)
    return _toy_geotensor(stack)


# ---------------------------------------------------------------------------
# Tier-A — primitive math
# ---------------------------------------------------------------------------


def test_mask_from_qa_bits_decodes_correctly() -> None:
    """Build a QA array with known bits, decode them, compare."""
    qa = np.array(
        [
            [0b00000000, 0b00001000, 0b00010000],  # 0, bit3, bit4
            [0b00011000, 0b00100000, 0b11111111],  # bits3+4, bit5, all
        ],
        dtype=np.uint16,
    )
    # Looking for bit 3 OR bit 4 (cloud OR shadow).
    out = mask_from_qa_bits(qa, bits=[3, 4])
    expected = np.array(
        [
            [False, True, True],
            [True, False, True],
        ]
    )
    np.testing.assert_array_equal(out, expected)


def test_mask_from_qa_bits_invert() -> None:
    qa = np.array([[0, 8]], dtype=np.uint16)
    out = mask_from_qa_bits(qa, bits=[3], invert=True)
    np.testing.assert_array_equal(out, [[True, False]])


def test_mask_from_qa_bits_rejects_negative_bit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mask_from_qa_bits(np.zeros(1, dtype=np.uint16), bits=[-1])


def test_mask_from_scl_classes() -> None:
    scl = np.array(
        [
            [9, 8, 4, 4],
            [10, 6, 5, 0],
        ],
        dtype=np.uint8,
    )
    # SCL_CLOUDS = {8, 9, 10}
    out = mask_from_scl(scl, list(SCL_CLOUDS))
    expected = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    np.testing.assert_array_equal(out, expected)


def test_mask_from_scl_accepts_enum_members() -> None:
    scl = np.array(
        [[int(SCL.CLOUD_HIGH_PROBABILITY), int(SCL.VEGETATION)]], dtype=np.uint8
    )
    out = mask_from_scl(scl, [SCL.CLOUD_HIGH_PROBABILITY])
    np.testing.assert_array_equal(out, [[True, False]])


def test_mask_from_scl_invert_keeps_only_listed() -> None:
    scl = np.array(
        [[SCL.VEGETATION, SCL.WATER, SCL.CLOUD_HIGH_PROBABILITY]], dtype=np.uint8
    )
    out = mask_from_scl(scl, [SCL.VEGETATION, SCL.WATER], invert=True)
    # invert=True -> True where NOT in classes -> True only for cloud.
    np.testing.assert_array_equal(out, [[False, False, True]])


def test_mask_from_scl_rejects_empty_classes() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mask_from_scl(np.zeros(1, dtype=np.uint8), [])


def test_apply_mask_fills_where_true() -> None:
    arr = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, False, True]])
    out = apply_mask(arr, mask, fill_value=-1.0)
    np.testing.assert_array_equal(out, [[-1.0, 2.0, -1.0]])


def test_apply_mask_broadcasts_over_bands() -> None:
    arr = np.array(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    )  # (2 bands, 1, 2)
    mask = np.array([[True, False]])  # (1, 2)
    out = apply_mask(arr, mask, fill_value=0.0)
    np.testing.assert_array_equal(out, [[[0.0, 2.0]], [[0.0, 4.0]]])


# ---------------------------------------------------------------------------
# SCL enum + convenience sets
# ---------------------------------------------------------------------------


def test_scl_enum_values() -> None:
    assert SCL.NO_DATA == 0
    assert SCL.VEGETATION == 4
    assert SCL.CLOUD_HIGH_PROBABILITY == 9
    assert SCL.SNOW == 11


def test_scl_convenience_sets_disjoint_where_expected() -> None:
    # Clouds and invalid are disjoint.
    assert SCL_CLOUDS.isdisjoint(SCL_INVALID)
    # Land is disjoint from clouds and from invalid.
    assert SCL_LAND.isdisjoint(SCL_CLOUDS)
    assert SCL_LAND.isdisjoint(SCL_INVALID)


# ---------------------------------------------------------------------------
# Tier-B — Operator + GeoTensor round-trip
# ---------------------------------------------------------------------------


def test_mask_from_scl_operator_preserves_metadata(s2_l2a_stack: GeoTensor) -> None:
    op = MaskFromSCL(band_idx=-1, classes=SCL_CLOUDS)
    mask = op(s2_l2a_stack)
    assert isinstance(mask, GeoTensor)
    assert mask.transform == s2_l2a_stack.transform
    assert mask.shape == (4, 4)  # band axis collapsed
    # Top-left quadrant is SCL_CLOUDS -> True.
    arr = np.asarray(mask)
    assert bool(arr[0, 0])
    assert not bool(arr[0, 2])


def test_mask_valid_uses_carrier_fill(s2_l2a_stack: GeoTensor) -> None:
    # Inject a -9999 sentinel into a known location and check it's flagged.
    arr = np.asarray(s2_l2a_stack).copy()
    arr[0, 0, 0] = -9999  # band 0, row 0, col 0
    gt = _toy_geotensor(arr)
    out = MaskValid()(gt)
    assert isinstance(out, GeoTensor)
    assert out.shape == (4, 4)
    assert bool(np.asarray(out)[0, 0])


def test_mask_valid_raises_when_no_fill() -> None:
    gt = GeoTensor(
        np.zeros((4, 4), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:4326",
        fill_value_default=None,
    )
    with pytest.raises(ValueError, match="invalid_value"):
        MaskValid()(gt)


def test_apply_mask_with_operator_argument(s2_l2a_stack: GeoTensor) -> None:
    """ApplyMask(mask=Operator(...)) should run the inner Operator first."""
    op = ApplyMask(
        mask=MaskFromSCL(band_idx=-1, classes=SCL_CLOUDS),
        fill_value=np.nan,
    )
    out = op(s2_l2a_stack)
    assert isinstance(out, GeoTensor)
    assert out.shape == s2_l2a_stack.shape
    arr = np.asarray(out)
    # Cloudy quadrant (top-left) -> NaN.
    assert np.all(np.isnan(arr[:, 0, 0]))
    # Vegetation quadrant (top-right) -> original values.
    assert not np.any(np.isnan(arr[:4, 0, 2]))  # reflectance bands


def test_apply_mask_invert(s2_l2a_stack: GeoTensor) -> None:
    op = ApplyMask(
        mask=MaskFromSCL(band_idx=-1, classes=SCL_CLOUDS),
        fill_value=0.0,
        invert=True,  # keep ONLY cloudy pixels, mask the rest to 0
    )
    out = op(s2_l2a_stack)
    arr = np.asarray(out)
    # Vegetation quadrant -> filled (0).
    assert np.all(arr[:4, 0, 2] == 0.0)
    # Cloudy quadrant -> untouched.
    assert not np.all(arr[:4, 0, 0] == 0.0)


def test_apply_mask_with_precomputed_array(s2_l2a_stack: GeoTensor) -> None:
    """A static boolean mask should work just as well."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # mask out one pixel
    op = ApplyMask(mask=mask, fill_value=-1.0)
    out = op(s2_l2a_stack)
    arr = np.asarray(out)
    assert np.all(arr[:, 0, 0] == -1.0)


def test_mask_from_qa_bits_operator_smoke() -> None:
    """End-to-end MaskFromQABits on a synthetic Landsat-like stack."""
    rng = np.random.default_rng(2)
    bands = rng.uniform(0.0, 1.0, size=(3, 4, 4)).astype(np.float32)
    qa = np.array(
        [
            [0, 0, 8, 8],
            [0, 0, 8, 8],
            [16, 16, 0, 0],
            [16, 16, 0, 0],
        ],
        dtype=np.uint16,
    )
    stack = np.concatenate([bands, qa[None].astype(np.float32)], axis=0)
    gt = _toy_geotensor(stack)
    op = MaskFromQABits(band_idx=-1, bits=[3, 4])  # cloud or shadow
    mask = op(gt)
    arr = np.asarray(mask)
    assert arr.shape == (4, 4)
    # All four quadrants except the bottom-right should be masked.
    assert bool(arr[0, 2])  # cloud quadrant
    assert bool(arr[2, 0])  # shadow quadrant
    assert not bool(arr[2, 2])  # clear quadrant


# ---------------------------------------------------------------------------
# Hydra-zen round-trip
# ---------------------------------------------------------------------------


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize(
    "op",
    [
        MaskFromQABits(band_idx=-1, bits=[2, 3, 4]),
        MaskFromSCL(band_idx=-1, classes=[8, 9, 10]),
        MaskValid(invalid_value=-9999),
    ],
)
def test_cloud_hydra_zen_roundtrip(op: object) -> None:
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_apply_mask_get_config_with_operator_mask_is_jsonable() -> None:
    """ApplyMask should nest the Operator-valued mask as {class, config}."""
    import json

    op = ApplyMask(
        mask=MaskFromSCL(band_idx=-1, classes=[8, 9, 10]),
        fill_value=float("nan"),
        invert=False,
    )
    cfg = op.get_config()
    # nan is float — json.dumps with allow_nan=True (the default) handles it.
    encoded = json.dumps(cfg)
    decoded = json.loads(encoded)
    assert decoded["mask"]["class"] == "MaskFromSCL"
    assert decoded["mask"]["config"]["band_idx"] == -1
    assert decoded["mask"]["config"]["classes"] == [8, 9, 10]
    assert decoded["invert"] is False
    assert ApplyMask.forbid_in_yaml is True  # carries an Operator/closure


def test_apply_mask_get_config_with_array_mask_emits_summary() -> None:
    """ApplyMask with a raw boolean array should serialize only the
    array's shape/dtype, not the bytes."""
    import json

    mask = np.array([[True, False], [False, True]])
    op = ApplyMask(mask=mask, fill_value=0.0)
    cfg = op.get_config()
    encoded = json.dumps(cfg)
    decoded = json.loads(encoded)
    assert decoded["mask"] == {"type": "ndarray", "shape": [2, 2], "dtype": "bool"}


def test_apply_mask_preserves_float32_dtype(s2_l2a_stack: GeoTensor) -> None:
    """Regression: fill_value=np.nan used to upcast float32 -> float64."""
    arr_f32 = np.asarray(s2_l2a_stack).astype(np.float32)
    gt_f32 = _toy_geotensor(arr_f32)
    op = ApplyMask(
        mask=MaskFromSCL(band_idx=-1, classes=SCL_CLOUDS),
        fill_value=np.nan,
    )
    out = op(gt_f32)
    assert out.dtype == np.float32  # no upcast despite NaN fill
