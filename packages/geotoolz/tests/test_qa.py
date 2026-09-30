"""Tests for `geotoolz.qa`."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor as _toy_geotensor

import geotoolz as gz
from geotoolz import qa


def test_qa_module_is_available_from_top_level() -> None:
    assert gz.qa is qa


def test_decode_bitmask_returns_named_boolean_layers() -> None:
    qa_band = np.array(
        [
            [0, 1 << 10, 1 << 11],
            [(1 << 10) | (1 << 11), 0, 1 << 10],
        ],
        dtype=np.uint16,
    )
    gt = _toy_geotensor(qa_band)

    out = qa.DecodeBitmask(bits={"cloud": [10], "cirrus": [11]})(gt)

    assert out.dtype == np.bool_
    assert out.shape == (2, 2, 3)
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.fill_value_default is False
    assert out.attrs["band_names"] == ["cloud", "cirrus"]
    np.testing.assert_array_equal(
        np.asarray(out)[0],
        [[False, True, False], [True, False, True]],
    )
    np.testing.assert_array_equal(
        np.asarray(out)[1],
        [[False, False, True], [True, False, False]],
    )


def test_decode_bitmask_selects_named_qa_band_with_default_any_mode() -> None:
    stack = np.stack(
        [
            np.ones((2, 2), dtype=np.uint16),
            np.array([[0, 1 << 10], [1 << 11, 0]], dtype=np.uint16),
        ]
    )
    gt = _toy_geotensor(stack, attrs={"band_names": ["B04", "QA60"]})

    out = qa.DecodeBitmask(bits={"bad": [10, 11]}, qa_band="QA60")(gt)

    np.testing.assert_array_equal(np.asarray(out)[0], [[False, True], [True, False]])


def test_decode_bitmask_replaces_every_band_name_key() -> None:
    """Stale ``descriptions`` no longer survive next to the new names (#144)."""
    stack = np.zeros((2, 2, 2), dtype=np.uint16)
    gt = _toy_geotensor(
        stack,
        attrs={"band_names": ["B04", "QA60"], "descriptions": ["red", "qa"], "k": 1},
    )

    out = qa.DecodeBitmask(bits={"bad": [10]}, qa_band="QA60")(gt)

    assert out.attrs == {"k": 1, "band_names": ["bad"]}
    assert out.attrs is not gt.attrs


def test_decode_bitmask_all_mode_requires_every_bit() -> None:
    gt = _toy_geotensor(np.array([[0, 1, 2, 3]], dtype=np.uint16))
    out = qa.DecodeBitmask(bits={"both": [0, 1]}, mode="all")(gt)
    np.testing.assert_array_equal(np.asarray(out)[0], [[False, False, False, True]])


def test_decode_bitmask_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        qa.DecodeBitmask(bits={})
    with pytest.raises(ValueError, match="'any' or 'all'"):
        qa.DecodeBitmask(bits={"cloud": [3]}, mode="bad")
    with pytest.raises(ValueError, match="non-negative"):
        qa.DecodeBitmask(bits={"cloud": [-1]})(_toy_geotensor(np.zeros((1, 1))))


def test_mask_clouds_uses_named_qa_band_bits() -> None:
    stack = np.stack(
        [
            np.ones((2, 3), dtype=np.uint16),
            np.array([[0, 1 << 10, 1 << 11], [0, 0, 0]], dtype=np.uint16),
        ]
    )
    gt = _toy_geotensor(stack, attrs={"band_names": ["B04", "QA60"]})

    out = qa.MaskClouds(qa_band="QA60", bits=[10, 11])(gt)

    assert out.shape == (2, 3)
    np.testing.assert_array_equal(
        np.asarray(out),
        [[False, True, True], [False, False, False]],
    )


def test_mask_clouds_rejects_missing_named_band() -> None:
    gt = _toy_geotensor(np.zeros((1, 2, 2)), attrs={"band_names": ["B04"]})
    with pytest.raises(ValueError, match="QA60"):
        qa.MaskClouds(qa_band="QA60", bits=[10])(gt)


def test_mask_shortcuts_support_categorical_values() -> None:
    scl = np.array([[4, 6, 9], [10, 11, 3]], dtype=np.uint8)
    gt = _toy_geotensor(scl)

    clouds = qa.MaskClouds(qa_band=None, values=[8, 9, 10])(gt)
    shadow = qa.MaskCloudShadow(qa_band=None, values=[3])(gt)
    cirrus = qa.MaskCirrus(qa_band=None, values=[10])(gt)
    snow = qa.MaskSnow(qa_band=None, values=[11])(gt)
    water = qa.MaskWater(qa_band=None, values=[6])(gt)

    np.testing.assert_array_equal(
        np.asarray(clouds), [[False, False, True], [True, False, False]]
    )
    np.testing.assert_array_equal(
        np.asarray(shadow), [[False, False, False], [False, False, True]]
    )
    np.testing.assert_array_equal(
        np.asarray(cirrus), [[False, False, False], [True, False, False]]
    )
    np.testing.assert_array_equal(
        np.asarray(snow), [[False, False, False], [False, True, False]]
    )
    np.testing.assert_array_equal(
        np.asarray(water), [[False, True, False], [False, False, False]]
    )


def test_mask_shortcuts_require_one_definition() -> None:
    gt = _toy_geotensor(np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="either bits or values"):
        qa.MaskClouds(qa_band=None)(gt)
    with pytest.raises(ValueError, match="only one"):
        qa.MaskClouds(qa_band=None, bits=[1], values=[1])(gt)


def test_s2qa60_preset_masks_cloud_and_cirrus() -> None:
    qa60 = np.array([[0, 1 << 10], [1 << 11, 0]], dtype=np.uint16)
    gt = _toy_geotensor(qa60[None], attrs={"band_names": ["QA60"]})

    out = qa.S2QA60()(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])
    assert out.dtype == np.bool_
    cloud_only = qa.S2QA60(targets=["cloud"])(gt)
    np.testing.assert_array_equal(
        np.asarray(cloud_only), [[False, True], [False, False]]
    )


def test_s2qa60_reads_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """S2QA60 decodes ``SENSOR_QA_REGISTRY["s2_qa60"]``, not hardcoded bits."""
    qa60 = np.array([[1 << 10, 1 << 3]], dtype=np.uint16)
    monkeypatch.setitem(
        qa.SENSOR_QA_REGISTRY,
        "s2_qa60",
        {"cloud": {"bits": (3,)}, "cirrus": {"bits": (11,)}},
    )
    out = qa.S2QA60(qa_band=None)(qa60)
    np.testing.assert_array_equal(out, [[False, True]])


def test_s2scl_targets_mask_listed_classes() -> None:
    scl = np.array([[4, 5, 6], [9, 11, 3]], dtype=np.uint8)
    gt = _toy_geotensor(scl[None], attrs={"band_names": ["SCL"]})

    out = qa.S2SCL(targets=["cloud", "snow"])(gt)

    np.testing.assert_array_equal(
        np.asarray(out),
        [[False, False, False], [True, True, False]],
    )


def test_scl_constants_agree_with_the_registry() -> None:
    """``SCL_CLOUDS`` and ``SENSOR_QA_REGISTRY["s2_scl"]`` share one source."""
    registry = qa.SENSOR_QA_REGISTRY["s2_scl"]

    def classes(*targets: str) -> set[int]:
        return {v for target in targets for v in registry[target]["values"]}

    assert set(qa.SCL_CLOUDS) == classes("cloud", "cirrus") == {8, 9, 10}
    assert set(qa.SCL_INVALID) == classes("no_data", "saturated", "cloud_shadow")
    assert set(qa.SCL_LAND) == classes("vegetation", "soil")
    assert set(qa.SCL_WATER) == classes("water")
    # Every SCL class is reachable from exactly one registry target.
    all_values = [v for entry in registry.values() for v in entry["values"]]
    assert sorted(all_values) == [int(c) for c in qa.SCL]


def test_landsat_qa_pixel_preset_targets() -> None:
    qa_pixel = np.array([[0, 1 << 3], [1 << 4, 1 << 7]], dtype=np.uint16)
    gt = _toy_geotensor(qa_pixel[None], attrs={"band_names": ["QA_PIXEL"]})

    out = qa.LandsatQA_PIXEL(targets=["cloud", "cloud_shadow"])(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_modis_state_qa_preset_targets() -> None:
    state_qa = np.array([[0, 0b01], [1 << 2, 1 << 8]], dtype=np.uint16)
    gt = _toy_geotensor(state_qa[None], attrs={"band_names": ["state_1km"]})

    out = qa.MODISStateQA(targets=["cloud", "cloud_shadow"])(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_presets_reject_unknown_targets() -> None:
    gt = _toy_geotensor(
        np.zeros((1, 1, 1), dtype=np.uint16), attrs={"band_names": ["QA_PIXEL"]}
    )
    with pytest.raises(ValueError, match="unknown landsat_qa_pixel"):
        qa.LandsatQA_PIXEL(targets=["not_a_flag"])(gt)

    scl = _toy_geotensor(
        np.zeros((1, 1, 1), dtype=np.uint8), attrs={"band_names": ["SCL"]}
    )
    with pytest.raises(ValueError, match="unknown s2_scl"):
        qa.S2SCL(targets=["not_a_class"])(scl)


def test_mask_nodata_infers_from_fill_value() -> None:
    arr = np.array(
        [
            [[1, -9999], [3, 4]],
            [[5, 6], [-9999, 8]],
        ],
        dtype=np.int16,
    )
    gt = _toy_geotensor(arr, fill_value_default=-9999)

    out = qa.MaskNoData()(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_mask_nodata_can_decode_qa_values() -> None:
    gt = _toy_geotensor(np.array([[0, 1], [2, 0]], dtype=np.uint8))
    out = qa.MaskNoData(qa_band=None, values=[0])(gt)
    np.testing.assert_array_equal(np.asarray(out), [[True, False], [False, True]])


def test_mask_nodata_requires_fill_when_not_decoding_qa() -> None:
    gt = _toy_geotensor(np.zeros((2, 2), dtype=np.float32), fill_value_default=None)
    with pytest.raises(ValueError, match="fill_value_default"):
        qa.MaskNoData()(gt)
    with pytest.raises(ValueError, match="needs bits or values"):
        qa.MaskNoData(qa_band=0)(gt)


def test_mask_nodata_is_a_qa_decoder() -> None:
    """MaskNoData is the shared `_QAMask` decoder with a fill fallback."""
    from geotoolz.qa._src.operators import _QAMask

    assert isinstance(qa.MaskNoData(), _QAMask)
    assert isinstance(qa.MaskClouds(bits=[3]), _QAMask)


def test_mask_invalid_matches_georeader_invalidmask() -> None:
    """#154: MaskInvalid == georeader ``invalidmask()`` reduced over bands."""
    rng = np.random.default_rng(1)
    arr = rng.integers(1, 100, size=(3, 5, 5)).astype(np.int16)
    arr[0, 1, 1] = -9999
    arr[2, 3, 4] = -9999
    arr[:, 0, 0] = -9999
    gt = _toy_geotensor(arr, fill_value_default=-9999)

    out = qa.MaskInvalid()(gt)

    expected = np.asarray(gt.invalidmask()).any(axis=0)
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert expected.sum() == 3
    assert out.dtype == np.bool_
    assert out.fill_value_default is False
    # MaskNoData's fill fallback is the same mask.
    np.testing.assert_array_equal(np.asarray(qa.MaskNoData()(gt)), expected)


def test_mask_invalid_detects_nan_fill() -> None:
    """A NaN fill is found although georeader's ``invalidmask()`` misses it."""
    arr = np.ones((2, 3, 3), dtype=np.float32)
    arr[:, 1, 2] = np.nan
    gt = _toy_geotensor(arr, fill_value_default=np.nan)

    out = qa.MaskInvalid()(gt)

    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 2] = True
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert not np.asarray(gt.invalidmask()).any()  # georeader's blind spot


def test_mask_invalid_explicit_value_and_plain_array() -> None:
    arr = np.array([[[1, 7], [7, 2]]], dtype=np.int16)
    np.testing.assert_array_equal(
        qa.MaskInvalid(invalid_value=7)(arr), [[False, True], [True, False]]
    )
    with pytest.raises(ValueError, match="invalid_value"):
        qa.MaskInvalid()(arr)


@pytest.mark.parametrize("fill", [-9999, np.nan], ids=["numeric", "nan"])
@pytest.mark.parametrize(
    "op", [qa.MaskInvalid(), qa.MaskNoData()], ids=["MaskInvalid", "MaskNoData"]
)
def test_fill_pixels_are_excluded(op, fill) -> None:
    """Fill pixels are flagged -- including a NaN fill (``nan == nan`` is False)."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.0, 1.0, size=(3, 4, 4)).astype(np.float32)
    gt = _toy_geotensor(arr, fill_value_default=fill, with_fill_pixels=True)

    out = op(gt)

    np.testing.assert_array_equal(np.asarray(out), fill_pixel_mask(arr.shape))
    assert out.dtype == np.bool_


def test_mask_nodata_flags_non_finite_under_numeric_fill() -> None:
    arr = np.ones((2, 2, 2), dtype=np.float32)
    arr[1, 0, 1] = np.nan
    gt = _toy_geotensor(arr, fill_value_default=-9999)
    np.testing.assert_array_equal(
        np.asarray(qa.MaskNoData()(gt)), [[False, True], [False, False]]
    )


def test_mask_saturated_infers_integer_max() -> None:
    max_uint16 = np.iinfo(np.uint16).max
    arr = np.array(
        [
            [[0, max_uint16], [2, 3]],
            [[4, 5], [max_uint16, 7]],
        ],
        dtype=np.uint16,
    )
    gt = _toy_geotensor(arr)

    out = qa.MaskSaturated()(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_mask_saturated_defaults_float_inputs_to_reflectance_ceiling() -> None:
    """Float input defaults to 1.0, as the removed ``SaturationFlag`` did."""
    refl = np.array([[[0.5, 1.0], [1.2, 0.99]]], dtype=np.float32)
    np.testing.assert_array_equal(
        qa.MaskSaturated()(refl), [[False, True], [True, False]]
    )
    with pytest.raises(ValueError, match="saturation_value"):
        qa.MaskSaturated()(np.zeros((1, 2, 2), dtype=bool))


def test_mask_saturated_is_a_threshold_with_optional_band_reduction() -> None:
    """Folds in the removed ``restore.SaturationFlag``: ``>=`` + per-band."""
    arr = np.array([[[0.25, 0.75]], [[1.5, 0.1]]])  # (2, 1, 2)

    per_band = qa.MaskSaturated(saturation_value=0.5, reduce_bands=False)(arr)
    reduced = qa.MaskSaturated(saturation_value=0.5)(arr)

    np.testing.assert_array_equal(per_band, [[[False, True]], [[True, False]]])
    np.testing.assert_array_equal(reduced, [[True, True]])
    # Band attrs survive the per-band form.
    gt = _toy_geotensor(arr, attrs={"band_names": ["B1", "B2"]})
    out = qa.MaskSaturated(saturation_value=0.5, reduce_bands=False)(gt)
    assert out.attrs["band_names"] == ["B1", "B2"]


def test_mask_saturated_never_flags_nodata() -> None:
    """A uint16 fill at the dtype max is not "saturated" (#146)."""
    counts = np.full((1, 4, 4), 100, dtype=np.uint16)
    gt = _toy_geotensor(counts, fill_value_default=65535, with_fill_pixels=True)

    out = qa.MaskSaturated()(gt)

    assert not np.asarray(out).any()
    assert out.fill_value_default is False
    # Plain arrays: NaN pixels are never saturated either.
    arr = np.array([[[np.nan, 2.0]]])
    np.testing.assert_array_equal(
        qa.MaskSaturated(saturation_value=1.0)(arr), [[False, True]]
    )


def test_ml_placeholders_are_configurable_and_explicit() -> None:
    assert qa.S2Cloudless(threshold=0.5).get_config() == {"threshold": 0.5}
    assert qa.OmniCloudMask(checkpoint="x").get_config() == {"checkpoint": "x"}
    assert qa.CloudSEN12(checkpoint="y").get_config() == {"checkpoint": "y"}
    for op in (qa.S2Cloudless(), qa.OmniCloudMask(), qa.CloudSEN12()):
        with pytest.raises(ImportError, match="optional ML mask extra"):
            op(_toy_geotensor(np.zeros((2, 2), dtype=np.float32)))


# ---------------------------------------------------------------------------
# Sensor-spec correctness: known QA values → known masks
# ---------------------------------------------------------------------------


def test_landsat8_qa_pixel_cloud_bit_3_per_lsds_1619() -> None:
    """LSDS-1619 Table 6-3: bit 3 = cloud on Landsat 8/9 QA_PIXEL."""
    # Synthetic: pixels with bits 3 (cloud), 4 (shadow), 2 (cirrus),
    # 1 (dilated), and 0 (fill) — plus a "clear" pixel.
    qa_pixel = np.array(
        [
            [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 0],
        ],
        dtype=np.uint16,
    )
    gt = _toy_geotensor(qa_pixel[None], attrs={"band_names": ["QA_PIXEL"]})
    out = qa.LandsatQA_PIXEL(targets=("cloud", "cloud_shadow", "cirrus"))(gt)
    # cirrus(bit 2), cloud(bit 3), cloud_shadow(bit 4) → True; rest False
    np.testing.assert_array_equal(
        np.asarray(out),
        [[False, False, True, True, True, False, False, False]],
    )


def test_landsat7_qa_pixel_lacks_cirrus_per_lsds_1618() -> None:
    """LSDS-1618: bit 2 is unused on Landsat 4-7 (no cirrus channel)."""
    # Bit 2 set on L7 should NOT trigger the cirrus target —
    # "cirrus" isn't in the L7 registry at all.
    gt = _toy_geotensor(
        np.array([[1 << 2]], dtype=np.uint16)[None],
        attrs={"band_names": ["QA_PIXEL"]},
    )
    with pytest.raises(ValueError, match="unknown landsat_qa_pixel_l7"):
        qa.LandsatQA_PIXEL(sensor="l7", targets=("cirrus",))(gt)
    # But cloud/shadow/etc work normally for L7.
    qa_pixel = np.array([[1 << 3, 1 << 4, 0]], dtype=np.uint16)
    gt = _toy_geotensor(qa_pixel[None], attrs={"band_names": ["QA_PIXEL"]})
    out = qa.LandsatQA_PIXEL(sensor="l7", targets=("cloud", "cloud_shadow"))(gt)
    np.testing.assert_array_equal(np.asarray(out), [[True, True, False]])


def test_landsat_qa_pixel_rejects_unknown_sensor() -> None:
    with pytest.raises(ValueError, match="sensor must be one of"):
        qa.LandsatQA_PIXEL(sensor="l11")


def test_landsat_qa_pixel_l7_default_targets_exclude_cirrus() -> None:
    """L7 default targets must omit ``cirrus`` (no cirrus bit on TM/ETM+)."""
    # Constructing with sensor="l7" and no explicit targets must not pick
    # up the global "cirrus" default — TM/ETM+ have no cirrus channel and
    # the L7 registry has no "cirrus" entry.
    op = qa.LandsatQA_PIXEL(sensor="l7")
    assert "cirrus" not in op.targets
    assert op.targets == ("cloud", "cloud_shadow")

    # And running the operator must not raise.
    qa_pixel = np.array([[1 << 3, 1 << 4, 0]], dtype=np.uint16)
    gt = _toy_geotensor(qa_pixel[None], attrs={"band_names": ["QA_PIXEL"]})
    out = op(gt)
    np.testing.assert_array_equal(np.asarray(out), [[True, True, False]])

    # Explicitly requesting cirrus on L7 still raises a clear ValueError
    # naming the unsupported target.
    gt_one = _toy_geotensor(
        np.array([[1 << 2]], dtype=np.uint16)[None],
        attrs={"band_names": ["QA_PIXEL"]},
    )
    with pytest.raises(ValueError, match=r"unknown landsat_qa_pixel_l7.*cirrus"):
        qa.LandsatQA_PIXEL(sensor="l7", targets=("cirrus",))(gt_one)


def test_modis_state_qa_decodes_cloud_field_not_individual_bits() -> None:
    """MOD09 user guide Table 12: bits [0,1] are a 2-bit cloud-state
    field (0=clear, 1=cloudy, 2=mixed, 3=not-set). OR-ing the bits
    independently — the wrong thing — would flag value 3 as cloudy. We
    flag only 1 ("cloudy") and 2 ("mixed").
    """
    # Field values: 0=clear, 1=cloudy, 2=mixed, 3=not-set.
    state_qa = np.array([[0b00, 0b01, 0b10, 0b11]], dtype=np.uint16)
    gt = _toy_geotensor(state_qa[None], attrs={"band_names": ["state_1km"]})
    out = qa.MODISStateQA(targets=("cloud",))(gt)
    # 0=clear→F, 1=cloudy→T, 2=mixed→T, 3=not-set→F
    np.testing.assert_array_equal(np.asarray(out), [[False, True, True, False]])


def test_modis_state_qa_cirrus_field_decoding() -> None:
    """Bits [8,9]: 0=none, 1=small, 2=average, 3=high. We flag 1/2/3."""
    state_qa = np.array([[0 << 8, 1 << 8, 2 << 8, 3 << 8]], dtype=np.uint16)
    gt = _toy_geotensor(state_qa[None], attrs={"band_names": ["state_1km"]})
    out = qa.MODISStateQA(targets=("cirrus",))(gt)
    np.testing.assert_array_equal(np.asarray(out), [[False, True, True, True]])


def test_s2_scl_default_keep_classes_per_sen2cor_spec() -> None:
    """Default S2SCL keeps vegetation(4), soil(5), water(6) only."""
    # One pixel per SCL class 0-11.
    scl = np.arange(12, dtype=np.uint8).reshape(1, 12)
    gt = _toy_geotensor(scl[None], attrs={"band_names": ["SCL"]})
    out = qa.S2SCL()(gt)
    # Classes 4, 5, 6 are kept → False; everything else masked → True.
    expected = np.array(
        [[True, True, True, True, False, False, False, True, True, True, True, True]],
    )
    np.testing.assert_array_equal(np.asarray(out), expected)


# ---------------------------------------------------------------------------
# Tier-A primitives — direct unit coverage
# ---------------------------------------------------------------------------


def test_mask_from_bit_field_requires_contiguous_ascending_bits() -> None:
    from geotoolz.qa._src.array import mask_from_bit_field

    arr = np.zeros((1, 1), dtype=np.uint16)
    with pytest.raises(ValueError, match="contiguous and ascending"):
        mask_from_bit_field(arr, bits=(0, 2), values=(1,))
    with pytest.raises(ValueError, match="contiguous and ascending"):
        mask_from_bit_field(arr, bits=(2, 1), values=(1,))
    with pytest.raises(ValueError, match="must not be empty"):
        mask_from_bit_field(arr, bits=(), values=(1,))
    with pytest.raises(ValueError, match="values"):
        mask_from_bit_field(arr, bits=(0,), values=())
    with pytest.raises(ValueError, match="non-negative"):
        mask_from_bit_field(arr, bits=(-1,), values=(1,))


def test_mask_from_bit_field_invert_flips_result() -> None:
    from geotoolz.qa._src.array import mask_from_bit_field

    arr = np.array([[0b00, 0b01, 0b10, 0b11]], dtype=np.uint16)
    out = mask_from_bit_field(arr, bits=(0, 1), values=(1,), invert=True)
    # value==1 → False; everything else → True.
    np.testing.assert_array_equal(np.asarray(out), [[True, False, True, True]])


def test_mask_from_qa_bits_decodes_and_inverts() -> None:
    qa_arr = np.array(
        [
            [0b00000000, 0b00001000, 0b00010000],  # 0, bit3, bit4
            [0b00011000, 0b00100000, 0b11111111],  # bits3+4, bit5, all
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(
        qa.mask_from_qa_bits(qa_arr, bits=[3, 4]),
        [[False, True, True], [True, False, True]],
    )
    np.testing.assert_array_equal(
        qa.mask_from_qa_bits(np.array([[0, 8]], dtype=np.uint16), [3], invert=True),
        [[True, False]],
    )
    with pytest.raises(ValueError, match="non-negative"):
        qa.mask_from_qa_bits(np.zeros(1, dtype=np.uint16), bits=[-1])


def test_mask_from_scl_classes_enum_members_and_invert() -> None:
    scl = np.array([[9, 8, 4, 4], [10, 6, 5, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(
        qa.mask_from_scl(scl, list(qa.SCL_CLOUDS)),
        [[True, True, False, False], [True, False, False, False]],
    )
    np.testing.assert_array_equal(
        qa.mask_from_scl(scl[:1, :3], [qa.SCL.CLOUD_HIGH_PROBABILITY]),
        [[True, False, False]],
    )
    np.testing.assert_array_equal(
        qa.mask_from_scl(scl[:1, 1:3], [qa.SCL.VEGETATION], invert=True),
        [[True, False]],
    )
    with pytest.raises(ValueError, match="must not be empty"):
        qa.mask_from_scl(np.zeros(1, dtype=np.uint8), [])


def test_scl_enum_values_and_disjoint_sets() -> None:
    assert qa.SCL.NO_DATA == 0
    assert qa.SCL.VEGETATION == 4
    assert qa.SCL.CLOUD_HIGH_PROBABILITY == 9
    assert qa.SCL.SNOW == 11
    assert qa.SCL_CLOUDS.isdisjoint(qa.SCL_INVALID)
    assert qa.SCL_LAND.isdisjoint(qa.SCL_CLOUDS | qa.SCL_INVALID)


def test_removed_duplicates_are_gone() -> None:
    """#154 removed the duplicate spellings outright (no aliases)."""
    for name in ("MaskFromQABits", "MaskFromSCL", "MaskValid", "reduce_bit_masks"):
        assert not hasattr(qa, name)
        assert not hasattr(gz, name)
    assert not hasattr(gz.restore, "SaturationFlag")
    assert not hasattr(gz, "SaturationFlag")
    with pytest.raises(ImportError):
        import geotoolz.cloud  # noqa: F401


def test_generic_decoder_defaults_to_carrier_as_qa_band() -> None:
    """``qa_band=None`` (default) treats the carrier as the QA band."""
    qa_arr = np.array([[0, 1 << 3], [1 << 4, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(
        qa.MaskClouds(bits=[3])(qa_arr), [[False, True], [False, False]]
    )
    keep_veg = qa.MaskClouds(values=[qa.SCL.VEGETATION], invert=True)
    np.testing.assert_array_equal(
        keep_veg(np.array([[4, 9]], dtype=np.uint8)), [[False, True]]
    )


def test_apply_mask_with_qa_operator_on_scl_stack() -> None:
    """ApplyMask(mask=MaskClouds(...)) runs the decoder on the input first."""
    from geotoolz.mask import ApplyMask

    rng = np.random.default_rng(0)
    refl = rng.uniform(0.05, 0.6, size=(2, 2, 2)).astype(np.float32)
    scl = np.array([[9, 4], [6, 0]], dtype=np.float32)
    stack = _toy_geotensor(
        np.concatenate([refl, scl[None]]), attrs={"band_names": ["B4", "B8", "SCL"]}
    )
    op = ApplyMask(
        mask=qa.MaskClouds(qa_band="SCL", values=sorted(qa.SCL_CLOUDS)),
        fill_value=np.nan,
    )
    out = np.asarray(op(stack))
    assert np.all(np.isnan(out[:, 0, 0]))
    assert not np.any(np.isnan(out[:2, 0, 1]))


# ---------------------------------------------------------------------------
# Plain-ndarray carrier support
# ---------------------------------------------------------------------------


_QA_BITS_2X2 = np.array([[0, 1 << 3], [1 << 4, (1 << 3) | (1 << 4)]], dtype=np.uint16)
_SCL_2X2 = np.array([[0, 4], [8, 9]], dtype=np.uint8)


def _plain_array_cases() -> list[tuple[object, np.ndarray]]:
    return [
        (qa.DecodeBitmask(bits={"cloud": [3], "shadow": [4]}), _QA_BITS_2X2),
        (qa.MaskClouds(qa_band=None, bits=[3, 4]), _QA_BITS_2X2),
        (qa.MaskNoData(qa_band=None, values=[0]), _SCL_2X2),
        (
            qa.MaskSaturated(),
            np.array([[0, 65535], [1, 2]], dtype=np.uint16),
        ),
        (
            qa.S2QA60(qa_band=0),
            np.array([[[0, 1 << 10], [1 << 11, 0]]], dtype=np.uint16),
        ),
        (qa.S2SCL(qa_band=0), _SCL_2X2[None]),
        (qa.MaskInvalid(invalid_value=0), _SCL_2X2),
        (qa.LandsatQA_PIXEL(qa_band=0), _QA_BITS_2X2[None]),
        (
            qa.MODISStateQA(qa_band=0),
            np.array([[[0b01, 0b10], [1 << 2, 0]]], dtype=np.uint16),
        ),
    ]


@pytest.mark.parametrize(
    ("op", "arr"), _plain_array_cases(), ids=lambda case: type(case).__name__
)
def test_qa_operators_accept_plain_ndarray(op, arr) -> None:
    """Plain ndarray in -> plain ndarray out, values match the GeoTensor path."""
    out = op(arr)
    assert type(out) is np.ndarray
    assert out.dtype == np.bool_
    gt_out = op(_toy_geotensor(arr))
    np.testing.assert_array_equal(out, np.asarray(gt_out))


def test_string_qa_band_selector_requires_geotensor_metadata() -> None:
    with pytest.raises(TypeError, match="band-name metadata"):
        qa.MaskClouds(qa_band="QA60", bits=[10])(np.zeros((2, 2, 2), dtype=np.uint16))


def test_mask_nodata_fill_mode_rejects_plain_array() -> None:
    with pytest.raises(ValueError, match="fill_value_default"):
        qa.MaskNoData()(np.zeros((2, 2), dtype=np.int16))


# ---------------------------------------------------------------------------
# Tier-B operator round-trip (get_config preserves behaviour)
# ---------------------------------------------------------------------------


def _qa_test_ops() -> list[object]:
    return [
        qa.DecodeBitmask(bits={"cloud": [3], "cirrus": [2]}, qa_band=0),
        qa.MaskClouds(qa_band=0, bits=[10, 11]),
        qa.MaskCloudShadow(qa_band=0, values=[3]),
        qa.MaskCirrus(qa_band=0, bits=[11]),
        qa.MaskSnow(qa_band=0, bits=[5]),
        qa.MaskWater(qa_band=0, bits=[7]),
        qa.MaskNoData(qa_band=0, values=[0]),
        qa.MaskNoData(),
        qa.MaskInvalid(invalid_value=-9999),
        qa.MaskSaturated(saturation_value=65535),
        qa.MaskSaturated(saturation_value=0.5, reduce_bands=False),
        qa.S2QA60(qa_band="QA60"),
        qa.S2QA60(targets=["cirrus"]),
        qa.S2SCL(qa_band="SCL", targets=["cloud", "cirrus"]),
        qa.LandsatQA_PIXEL(targets=["cloud", "cloud_shadow"], sensor="l89"),
        qa.LandsatQA_PIXEL(targets=["cloud"], sensor="l7"),
        qa.MODISStateQA(targets=["cloud", "cloud_shadow", "cirrus"]),
        qa.S2Cloudless(threshold=0.42),
        qa.OmniCloudMask(checkpoint="ckpt"),
        qa.CloudSEN12(checkpoint="ckpt"),
    ]


@pytest.mark.parametrize("op", _qa_test_ops())
def test_qa_operator_get_config_roundtrip(op: object) -> None:
    """Reconstructing from get_config() preserves behaviour."""
    cfg = op.get_config()  # type: ignore[attr-defined]
    clone = type(op)(**cfg)
    assert clone.get_config() == cfg  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Hydra-zen round-trip
# ---------------------------------------------------------------------------


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize("op", _qa_test_ops())
def test_qa_hydra_zen_roundtrip(op: object) -> None:
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_4d_time_stack() -> None:
    """QA decoders select the QA band along -3, per frame (#147)."""
    qa_frame = np.zeros((2, 4, 4), dtype=np.uint16)
    qa_frame[1, 0, 0] = 1 << 3
    stack_values = np.stack([qa_frame, np.roll(qa_frame, 1, axis=-1)])
    stack = _toy_geotensor(stack_values, fill_value_default=None)
    mask = qa.MaskClouds(qa_band=1, bits=[3])(stack)
    assert mask.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(np.asarray(mask)[:, 0], stack_values[:, 1] > 0)
    layers = qa.DecodeBitmask(bits={"cloud": [3], "none": [5]}, qa_band=1)(stack)
    assert layers.shape == (2, 2, 4, 4)
    assert layers.attrs["band_names"] == ["cloud", "none"]
