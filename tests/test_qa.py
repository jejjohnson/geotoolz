"""Tests for `geotoolz.qa`."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz import qa


def _toy_geotensor(
    values: np.ndarray,
    *,
    attrs: dict[str, object] | None = None,
    fill: int | float | None = -9999,
) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=fill,
        attrs=attrs,
    )


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


def test_s2scl_keep_masks_everything_else() -> None:
    scl = np.array([[4, 5, 6], [9, 11, 3]], dtype=np.uint8)
    gt = _toy_geotensor(scl[None], attrs={"band_names": ["SCL"]})

    out = qa.S2SCL(keep=["vegetation"])(gt)

    np.testing.assert_array_equal(
        np.asarray(out),
        [[False, True, True], [True, True, True]],
    )


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
        qa.S2SCL(keep=["not_a_class"])(scl)


def test_mask_nodata_infers_from_fill_value() -> None:
    arr = np.array(
        [
            [[1, -9999], [3, 4]],
            [[5, 6], [-9999, 8]],
        ],
        dtype=np.int16,
    )
    gt = _toy_geotensor(arr, fill=-9999)

    out = qa.MaskNoData()(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_mask_nodata_can_decode_qa_values() -> None:
    gt = _toy_geotensor(np.array([[0, 1], [2, 0]], dtype=np.uint8))
    out = qa.MaskNoData(qa_band=None, values=[0])(gt)
    np.testing.assert_array_equal(np.asarray(out), [[True, False], [False, True]])


def test_mask_nodata_requires_fill_when_not_decoding_qa() -> None:
    gt = _toy_geotensor(np.zeros((2, 2), dtype=np.float32), fill=None)
    with pytest.raises(ValueError, match="fill_value_default"):
        qa.MaskNoData()(gt)


def test_mask_saturated_infers_integer_max() -> None:
    arr = np.array(
        [
            [[0, np.iinfo(np.uint16).max], [2, 3]],
            [[4, 5], [np.iinfo(np.uint16).max, 7]],
        ],
        dtype=np.uint16,
    )
    gt = _toy_geotensor(arr)

    out = qa.MaskSaturated()(gt)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_mask_saturated_requires_value_for_float_inputs() -> None:
    gt = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="saturation_value"):
        qa.MaskSaturated()(gt)


def test_ml_placeholders_are_configurable_and_explicit() -> None:
    gt = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
    assert qa.S2Cloudless(threshold=0.5).get_config() == {"threshold": 0.5}
    assert qa.OmniCloudMask(checkpoint="x").get_config() == {"checkpoint": "x"}
    assert qa.CloudSEN12(checkpoint="y").get_config() == {"checkpoint": "y"}
    with pytest.raises(ImportError, match="optional ML mask extra"):
        qa.S2Cloudless()(gt)
