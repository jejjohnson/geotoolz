"""Tests for `geotoolz.augment`."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz import augment
from geotoolz.core import Operator


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=0,
        attrs={
            "band_names": ["B02", "B03", "B04", "B08"],
            "wavelengths_nm": [490.0, 560.0, 665.0, 842.0],
            "solar_zenith_angle": 30.0,
        },
    )


@pytest.fixture
def patch() -> GeoTensor:
    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) / 100.0
    return _toy_geotensor(arr)


def _xy(transform: rasterio.Affine, col: int, row: int) -> tuple[float, float]:
    return transform * (col, row)


def test_imports_augment_module() -> None:
    assert gz.augment is augment
    assert hasattr(gz.augment, "RandomFlip")


def test_random_flip_noop_and_forced_transform(patch: GeoTensor) -> None:
    no_op = augment.RandomFlip(p_horizontal=0.0, p_vertical=0.0, seed=0)(patch)
    assert no_op is patch

    flipped = augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)(patch)
    np.testing.assert_array_equal(np.asarray(flipped), np.flip(np.asarray(patch), -1))
    assert flipped.crs == patch.crs
    assert flipped.dtype == patch.dtype
    assert _xy(flipped.transform, 0, 0) == _xy(patch.transform, patch.width - 1, 0)


def test_random_rotate90_matches_numpy_and_updates_transform(patch: GeoTensor) -> None:
    seed = 3
    rng = np.random.default_rng(seed)
    rng.random()
    k = int(rng.integers(1, 4))

    out = augment.RandomRotate90(p=1.0, seed=seed)(patch)

    np.testing.assert_array_equal(
        np.asarray(out), np.rot90(np.asarray(patch), k=k, axes=(-2, -1))
    )
    assert out.shape[-2:] == np.rot90(np.asarray(patch)[0], k=k).shape
    if k == 1:
        assert _xy(out.transform, 0, 0) == _xy(patch.transform, patch.width - 1, 0)
    elif k == 2:
        assert _xy(out.transform, 0, 0) == _xy(
            patch.transform, patch.width - 1, patch.height - 1
        )
    else:
        assert _xy(out.transform, 0, 0) == _xy(patch.transform, 0, patch.height - 1)


def test_random_crop_and_shift_update_spatial_metadata(patch: GeoTensor) -> None:
    cropped = augment.RandomCrop(size=(3, 4), seed=0)(patch)
    assert cropped.shape == (4, 3, 4)
    assert cropped.crs == patch.crs
    assert cropped.transform != patch.transform

    shifted = augment.RandomShift(max_shift=(1, 1), seed=0)(patch)
    assert shifted.shape == patch.shape
    assert shifted.crs == patch.crs
    assert shifted.dtype == patch.dtype


def test_random_crop_rejects_invalid_size(patch: GeoTensor) -> None:
    with pytest.raises(ValueError, match="positive"):
        augment.RandomCrop(size=(0, 4))
    with pytest.raises(ValueError, match="fit"):
        augment.RandomCrop(size=(patch.height + 1, patch.width))(patch)


def test_seed_override_is_reproducible(patch: GeoTensor) -> None:
    op = augment.GaussianNoise(sigma=(0.01, 0.02), seed=1)
    first = op(patch, seed=42)
    second = op(patch, seed=42)
    third = op(patch, seed=43)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(third))


def test_brightness_jitter_per_band_and_shared_factor(patch: GeoTensor) -> None:
    per_band = augment.BrightnessJitter(factor=(0.9, 1.1), per_band=True, seed=0)(patch)
    shared = augment.BrightnessJitter(factor=(0.9, 1.1), per_band=False, seed=0)(patch)

    source = np.asarray(patch)
    per_band_ratio = np.asarray(per_band)[:, 1, 1] / source[:, 1, 1]
    shared_ratio = np.asarray(shared)[:, 1, 1] / source[:, 1, 1]
    assert len(np.unique(np.round(per_band_ratio, 6))) > 1
    np.testing.assert_allclose(shared_ratio, shared_ratio[0], rtol=1e-6)
    assert per_band.dtype == patch.dtype
    assert per_band.shape == patch.shape


def test_brightness_jitter_statistical_midpoint(patch: GeoTensor) -> None:
    factors = []
    pixel = float(np.asarray(patch)[0, 1, 1])
    # The feature request asks for a 1000-sample statistical check.
    for seed in range(1000):
        out = augment.BrightnessJitter(factor=(0.8, 1.2), per_band=False, seed=seed)(
            patch
        )
        factors.append(float(np.asarray(out)[0, 1, 1]) / pixel)
    assert np.mean(factors) == pytest.approx(1.0, abs=0.01)


def test_contrast_jitter_noise_and_speckle_preserve_shape_dtype(
    patch: GeoTensor,
) -> None:
    ops = [
        augment.ContrastJitter(factor=(0.95, 1.05), seed=0),
        augment.GaussianNoise(sigma=0.01, per_band=False, seed=0),
        augment.SpeckleNoise(sigma=(0.01, 0.02), seed=0),
    ]
    for op in ops:
        out = op(patch)
        assert out.shape == patch.shape
        assert out.dtype == patch.dtype
        assert out.transform == patch.transform
        assert out.crs == patch.crs


def test_negative_noise_parameters_raise(patch: GeoTensor) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        augment.GaussianNoise(sigma=-0.1)(patch)
    with pytest.raises(ValueError, match="non-negative"):
        augment.SpeckleNoise(sigma=-0.1)(patch)


def test_band_dropout_identity_and_all_fill(patch: GeoTensor) -> None:
    identity = augment.BandDropout(p=0.0, fill=-1, seed=0)(patch)
    filled = augment.BandDropout(p=1.0, fill=-1, seed=0)(patch)

    np.testing.assert_array_equal(np.asarray(identity), np.asarray(patch))
    np.testing.assert_array_equal(np.asarray(filled), np.full(patch.shape, -1.0))
    assert filled.dtype == patch.dtype


def test_band_jitter_disabled_and_grouped_permutation(patch: GeoTensor) -> None:
    disabled = augment.BandJitter()(patch)
    assert disabled is patch

    jittered = augment.BandJitter(groups={"visible": ["B02", "B03", "B04"]}, seed=1)(
        patch
    )
    source = np.asarray(patch)
    out = np.asarray(jittered)
    assert np.array_equal(out[3], source[3])
    assert {tuple(b.ravel()) for b in out[:3]} == {tuple(b.ravel()) for b in source[:3]}


def test_band_jitter_requires_names(patch: GeoTensor) -> None:
    unnamed = GeoTensor(
        np.asarray(patch),
        patch.transform,
        patch.crs,
        patch.fill_value_default,
        attrs={},
    )
    with pytest.raises(ValueError, match="band names"):
        augment.BandJitter(groups={"g": ["B02", "B03"]})(unnamed)


def test_sun_angle_haze_and_cloud_identity_cases(patch: GeoTensor) -> None:
    sun = augment.SunAngleJitter(delta_sza_deg=0.0, seed=0)(patch)
    haze = augment.AtmosphericHaze(intensity=0.0, seed=0)(patch)
    clouds = augment.SimulatedClouds(coverage=0.0, seed=0)(patch)

    np.testing.assert_array_equal(np.asarray(sun), np.asarray(patch))
    np.testing.assert_array_equal(np.asarray(haze), np.asarray(patch))
    np.testing.assert_array_equal(np.asarray(clouds), np.asarray(patch))


def test_haze_uses_inverse_fourth_power_spectral_weights(patch: GeoTensor) -> None:
    out = augment.AtmosphericHaze(intensity=0.05, seed=0)(patch)
    delta = np.asarray(out) - np.asarray(patch)
    assert delta[0].mean() > delta[-1].mean()
    assert out.shape == patch.shape
    assert out.dtype == patch.dtype


def test_simulated_clouds_changes_pixels_but_preserves_metadata(
    patch: GeoTensor,
) -> None:
    out = augment.SimulatedClouds(coverage=0.5, feather=1, seed=0)(patch)
    assert out.shape == patch.shape
    assert out.dtype == patch.dtype
    assert out.transform == patch.transform
    assert not np.array_equal(np.asarray(out), np.asarray(patch))


def test_cutmix_probability_shape_and_mismatch(patch: GeoTensor) -> None:
    donor = _toy_geotensor(np.full(patch.shape, 9.0, dtype=np.float32))

    identity = augment.CutMix(pool=[donor], p=0.0, seed=0)(patch)
    mixed = augment.CutMix(pool=[donor], p=1.0, seed=0)(patch)

    assert identity is patch
    assert mixed.shape == patch.shape
    assert mixed.transform == patch.transform
    assert np.any(np.asarray(mixed) == 9.0)

    bad = _toy_geotensor(np.zeros((4, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="match"):
        augment.CutMix(pool=[bad], p=1.0, seed=0)(patch)


class _AffineTestOp(Operator):
    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        del seed
        return gt.array_as_geotensor(np.asarray(gt) * self.scale + self.offset)


def test_compose_applies_in_order_and_respects_probability(patch: GeoTensor) -> None:
    composed = augment.Compose(
        [_AffineTestOp(scale=2.0, offset=0.0), _AffineTestOp(scale=1.0, offset=3.0)]
    )
    out = composed(patch, seed=0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(patch) * 2.0 + 3.0)

    skipped = augment.Compose([_AffineTestOp(scale=2.0, offset=0.0)], p=0.0)(
        patch, seed=0
    )
    assert skipped is patch


def test_random_flip_is_an_involution(patch: GeoTensor) -> None:
    """Two horizontal flips must reproduce the original pixels and transform."""
    op = augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)
    once = op(patch)
    twice = op(once)
    np.testing.assert_array_equal(np.asarray(twice), np.asarray(patch))
    assert twice.transform == patch.transform
    assert twice.crs == patch.crs


def test_random_rotate90_known_transform_on_unit_raster() -> None:
    """For each k in {1, 2, 3}, output (0,0) maps to the expected corner."""
    arr = np.arange(2 * 2, dtype=np.float32).reshape(1, 2, 2)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )

    # Exercise public API across many seeds so we cover all three k values
    # without depending on RNG internals.
    seen_corners = set()
    for seed in range(50):
        rng = np.random.default_rng(seed)
        rng.random()  # mirror the p-check draw inside _apply
        k = int(rng.integers(1, 4))

        out = augment.RandomRotate90(p=1.0, seed=seed)(gt)
        np.testing.assert_array_equal(
            np.asarray(out)[0], np.rot90(np.asarray(gt)[0], k=k)
        )

        expected = {
            1: gt.transform * (gt.width - 1, 0),
            2: gt.transform * (gt.width - 1, gt.height - 1),
            3: gt.transform * (0, gt.height - 1),
        }[k]
        assert out.transform * (0, 0) == expected
        seen_corners.add(k)
    assert seen_corners == {1, 2, 3}


def test_random_crop_translation_matches_origin(patch: GeoTensor) -> None:
    """Cropping shifts the transform translation to the new origin pixel."""
    seed = 0
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, patch.height - 3 + 1))
    left = int(rng.integers(0, patch.width - 4 + 1))

    out = augment.RandomCrop(size=(3, 4), seed=seed)(patch)

    expected = patch.transform * (left, top)
    assert out.transform * (0, 0) == expected
    assert out.shape == (4, 3, 4)


def test_cutmix_rejects_mismatched_crs_and_resolution(patch: GeoTensor) -> None:
    arr = np.full(patch.shape, 9.0, dtype=np.float32)
    different_crs = GeoTensor(
        values=arr,
        transform=patch.transform,
        crs="EPSG:4326",
        fill_value_default=0,
    )
    different_res = GeoTensor(
        values=arr,
        transform=rasterio.Affine(20.0, 0.0, 500_000.0, 0.0, -20.0, 4_000_000.0),
        crs=patch.crs,
        fill_value_default=0,
    )

    with pytest.raises(ValueError, match="CRS"):
        augment.CutMix(pool=[different_crs], p=1.0, seed=0)(patch)
    with pytest.raises(ValueError, match="resolution"):
        augment.CutMix(pool=[different_res], p=1.0, seed=0)(patch)


def test_get_config_is_json_safe(patch: GeoTensor) -> None:
    """Every public augmentation's config should serialise to JSON."""
    donor = _toy_geotensor(np.full(patch.shape, 9.0, dtype=np.float32))
    ops: list[Operator] = [
        augment.RandomFlip(seed=0),
        augment.RandomRotate90(seed=0),
        augment.RandomCrop(size=(2, 2), seed=0),
        augment.RandomShift(max_shift=(1, 1), seed=0),
        augment.BrightnessJitter(seed=0),
        augment.ContrastJitter(seed=0),
        augment.GaussianNoise(sigma=(0.0, 0.1), seed=0),
        augment.SpeckleNoise(sigma=(0.0, 0.1), seed=0),
        augment.BandDropout(seed=0),
        augment.BandJitter(seed=0),
        augment.SunAngleJitter(seed=0),
        augment.AtmosphericHaze(seed=0),
        augment.SimulatedClouds(seed=0),
        augment.CutMix(pool=[donor], seed=0),
        augment.Compose([augment.RandomFlip(seed=0)], seed=0),
    ]
    for op in ops:
        # Should not raise — tuples become lists, no live objects leak through.
        json.dumps(op.get_config())


def test_compose_and_cutmix_are_forbid_in_yaml() -> None:
    assert augment.Compose([]).forbid_in_yaml is True
    assert augment.CutMix(pool=[]).forbid_in_yaml is True
