"""Tests for `geotoolz.augment`."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import rasterio
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor
from pipekit import Operator, Sequential

import geotoolz as gz
from geotoolz import augment


def _toy_geotensor(values: np.ndarray, **kwargs: Any) -> GeoTensor:
    # The ``patch`` data starts at 0.0, so a 0 fill would silently mark
    # pixel (0, 0) as nodata; use a fill the toy data never takes.
    kwargs.setdefault("fill_value_default", -9999.0)
    return toy_geotensor(
        values,
        **kwargs,
        attrs={
            "band_names": ["B02", "B03", "B04", "B08"],
            "wavelengths": [490.0, 560.0, 665.0, 842.0],
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
    # The mirrored origin is the far pixel edge, so the extent is unchanged.
    assert _xy(flipped.transform, 0, 0) == _xy(patch.transform, patch.width, 0)
    assert flipped.bounds == patch.bounds


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
        assert _xy(out.transform, 0, 0) == _xy(patch.transform, patch.width, 0)
    elif k == 2:
        assert _xy(out.transform, 0, 0) == _xy(
            patch.transform, patch.width, patch.height
        )
    else:
        assert _xy(out.transform, 0, 0) == _xy(patch.transform, 0, patch.height)
    assert out.bounds == patch.bounds


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
    identity = augment.BandDropout(p=0.0, fill_value=-1, seed=0)(patch)
    filled = augment.BandDropout(p=1.0, fill_value=-1, seed=0)(patch)

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

    identity = augment.CutMix(p=0.0, seed=0)(patch, donor)
    mixed = augment.CutMix(p=1.0, seed=0)(patch, donor)

    assert identity is patch
    assert mixed.shape == patch.shape
    assert mixed.transform == patch.transform
    assert np.any(np.asarray(mixed) == 9.0)

    bad = _toy_geotensor(np.zeros((4, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"CutMix.*match"):
        augment.CutMix(p=1.0, seed=0)(patch, bad)


def test_cutmix_pool_is_positional(patch: GeoTensor) -> None:
    """Donors are positional carriers, alone or as one list (#141)."""
    donors = [
        _toy_geotensor(np.full(patch.shape, v, dtype=np.float32)) for v in (7.0, 9.0)
    ]
    spread = augment.CutMix(p=1.0, seed=0)(patch, *donors)
    listed = augment.CutMix(p=1.0, seed=0)(patch, donors)
    np.testing.assert_array_equal(np.asarray(spread), np.asarray(listed))
    # No donors: the input passes through.
    assert augment.CutMix(p=1.0, seed=0)(patch) is patch
    # Every donor is grid-checked, not only the one drawn.
    bad = _toy_geotensor(np.zeros((4, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="CutMix: the donor 1 pixel grid"):
        augment.CutMix(p=0.0, seed=0)(patch, donors[0], bad)
    with pytest.raises(TypeError, match="pool"):
        augment.CutMix(pool=donors)  # ty: ignore[unknown-argument]


class _AffineTestOp(Operator):
    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        del seed
        return gt.array_as_geotensor(np.asarray(gt) * self.scale + self.offset)


def test_compose_applies_in_order_and_respects_probability(patch: GeoTensor) -> None:
    composed = augment.Compose(
        augmentations=[
            _AffineTestOp(scale=2.0, offset=0.0),
            _AffineTestOp(scale=1.0, offset=3.0),
        ]
    )
    out = composed(patch, seed=0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(patch) * 2.0 + 3.0)

    skipped = augment.Compose(
        augmentations=[_AffineTestOp(scale=2.0, offset=0.0)], p=0.0
    )(patch, seed=0)
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
            1: gt.transform * (gt.width, 0),
            2: gt.transform * (gt.width, gt.height),
            3: gt.transform * (0, gt.height),
        }[k]
        assert out.transform * (0, 0) == expected
        assert out.bounds == gt.bounds
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


def test_cutmix_rejects_donors_off_the_input_grid(patch: GeoTensor) -> None:
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

    # Same resolution and CRS but a different origin: pasting it pixel-for-
    # pixel would put the donor's content at the wrong place.
    shifted = GeoTensor(
        values=arr,
        transform=patch.transform * rasterio.Affine.translation(2, 0),
        crs=patch.crs,
        fill_value_default=0,
    )

    for donor in (different_crs, different_res, shifted):
        with pytest.raises(ValueError, match="pixel grid"):
            augment.CutMix(p=1.0, seed=0)(patch, donor)


def test_get_config_is_json_safe(patch: GeoTensor) -> None:
    """Every public augmentation's config should serialise to JSON."""
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
        augment.CutMix(seed=0),
        augment.Compose(augmentations=[augment.RandomFlip(seed=0)], seed=0),
    ]
    for op in ops:
        # Should not raise — tuples become lists, no live objects leak through.
        json.dumps(op.get_config())


def test_cutmix_and_compose_round_trip() -> None:
    # CutMix takes its donors at call time (#141), so it holds no runtime
    # rasters; Compose only nests operators, which pipekit's container rule
    # leaves to the children (#140).
    assert augment.CutMix().forbid_in_yaml is False
    op = augment.CutMix(p=0.3, seed=1)
    assert Operator.from_state(json.loads(json.dumps(op.state))).get_config() == (
        op.get_config()
    )
    assert augment.Compose(augmentations=[]).forbid_in_yaml is False


def test_simulated_clouds_rejects_invalid_coverage_at_construction() -> None:
    """Coverage outside [0, 1] must fail at construction, not at apply time."""
    with pytest.raises(ValueError, match="coverage"):
        augment.SimulatedClouds(coverage=(0.5, 1.5))
    with pytest.raises(ValueError, match="coverage"):
        augment.SimulatedClouds(coverage=(-0.1, 0.5))
    with pytest.raises(ValueError, match="coverage"):
        augment.SimulatedClouds(coverage=1.5)
    with pytest.raises(ValueError, match="coverage"):
        augment.SimulatedClouds(coverage=(0.5, 0.2))


def test_range_ops_reject_reversed_range_at_construction() -> None:
    """Tuples with hi < lo must raise at construction, not silently swap."""
    with pytest.raises(ValueError, match="lo <= hi"):
        augment.BrightnessJitter(factor=(1.1, 0.9))
    with pytest.raises(ValueError, match="lo <= hi"):
        augment.ContrastJitter(factor=(1.1, 0.9))
    with pytest.raises(ValueError, match="lo <= hi"):
        augment.GaussianNoise(sigma=(0.2, 0.1))
    with pytest.raises(ValueError, match="lo <= hi"):
        augment.SpeckleNoise(sigma=(0.2, 0.1))


def test_sun_angle_jitter_requires_sza_in_attrs(patch: GeoTensor) -> None:
    """SunAngleJitter must not silently fall back to a hard-coded 30 degrees."""
    no_sza = GeoTensor(
        np.asarray(patch),
        patch.transform,
        patch.crs,
        patch.fill_value_default,
        attrs={"band_names": ["B02", "B03", "B04", "B08"]},
    )
    with pytest.raises(ValueError, match="solar_zenith_angle"):
        augment.SunAngleJitter(delta_sza_deg=1.0, seed=0)(no_sza)


class _NonStochasticOp(Operator):
    """Plain Operator whose ``__init__`` does not accept ``seed``."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(np.asarray(gt) * self.scale)


def test_compose_forwards_seed_only_to_stochastic_children(patch: GeoTensor) -> None:
    """Mixing a non-stochastic child into Compose must not raise TypeError."""
    composed = augment.Compose(
        augmentations=[
            _NonStochasticOp(scale=2.0),
            augment.GaussianNoise(sigma=0.0, seed=0),
        ],
        seed=0,
    )
    # Should not raise even though `_NonStochasticOp._apply` rejects `seed`.
    out = composed(patch, seed=7)
    assert out.shape == patch.shape
    assert out.dtype == patch.dtype


@pytest.mark.parametrize(
    "op",
    [
        augment.BrightnessJitter(seed=0),
        augment.ContrastJitter(seed=0),
        augment.GaussianNoise(sigma=0.01, seed=0),
        augment.SpeckleNoise(sigma=0.02, seed=0),
        augment.BandDropout(p=0.5, fill_value=-1.0, seed=0),
        augment.AtmosphericHaze(intensity=0.05, seed=0),
        augment.SimulatedClouds(coverage=0.5, feather=1, seed=0),
        augment.RandomFlip(p_horizontal=1.0, p_vertical=1.0, seed=0),
        augment.RandomRotate90(p=1.0, seed=0),
        augment.RandomCrop(size=(2, 3), seed=0),
    ],
    ids=lambda op: type(op).__name__,
)
def test_plain_ndarray_in_plain_ndarray_out(op: Operator) -> None:
    """Plain ndarray in -> plain ndarray out, values equal to the GeoTensor path."""
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 100.0
    gt = toy_geotensor(arr)  # no wavelength attrs: both paths use defaults

    # Independent copies: a seeded op's stream advances on every call.
    plain_out = copy.deepcopy(op)(arr)
    geo_out = copy.deepcopy(op)(gt)

    assert type(plain_out) is np.ndarray
    assert isinstance(geo_out, GeoTensor)
    np.testing.assert_array_equal(plain_out, np.asarray(geo_out))


def test_metadata_dependent_ops_reject_plain_arrays() -> None:
    """Geo/attrs-dependent augmentations fail clearly on plain arrays."""
    arr = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="georeferenced GeoTensor"):
        augment.RandomShift(max_shift=(1, 1), seed=0)(arr)
    with pytest.raises(TypeError, match="band-name metadata"):
        augment.BandJitter(groups={"g": ["B02", "B03"]}, seed=0)(arr)
    with pytest.raises(ValueError, match="solar_zenith_angle"):
        augment.SunAngleJitter(delta_sza_deg=1.0, seed=0)(arr)


def test_cutmix_and_compose_support_plain_arrays() -> None:
    arr = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) / 100.0
    donor = np.full_like(arr, 9.0)

    mixed = augment.CutMix(p=1.0, seed=0)(arr, donor)
    assert type(mixed) is np.ndarray
    assert np.any(mixed == 9.0)

    composed = augment.Compose(
        augmentations=[augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)],
        seed=0,
    )
    out = composed(arr)
    assert type(out) is np.ndarray
    np.testing.assert_array_equal(out, np.flip(arr, axis=-1))


def test_compose_rejects_non_operator_children() -> None:
    """A reloaded nested payload fails at construction, not at apply."""
    payload = augment.Compose(augmentations=[augment.RandomFlip()]).get_config()[
        "augmentations"
    ]
    with pytest.raises(TypeError, match="must be an Operator"):
        augment.Compose(augmentations=payload)


@pytest.mark.parametrize(
    ("p_horizontal", "p_vertical"),
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ids=["horizontal", "vertical", "both"],
)
def test_flip_preserves_footprint(
    patch: GeoTensor, p_horizontal: float, p_vertical: float
) -> None:
    """A flip mirrors pixels in place: same bounds, same footprint (#126)."""
    out = augment.RandomFlip(p_horizontal=p_horizontal, p_vertical=p_vertical)(patch)
    assert out.bounds == patch.bounds
    assert out.footprint().equals(patch.footprint())
    # Mirrored pixels keep their world coordinates.
    unflipped = np.asarray(out)
    if p_horizontal:
        unflipped = np.flip(unflipped, -1)
    if p_vertical:
        unflipped = np.flip(unflipped, -2)
    np.testing.assert_array_equal(unflipped, np.asarray(patch))
    assert not np.shares_memory(np.asarray(out), np.asarray(patch))


def _seed_for_k(k: int) -> int:
    """Per-call seed whose RandomRotate90(p=1) draw is ``k`` quarter-turns."""
    for seed in range(100):
        rng = np.random.default_rng(seed)
        rng.random()  # mirror the p-check draw inside _apply
        if int(rng.integers(1, 4)) == k:
            return seed
    raise AssertionError(f"no seed draws k={k}")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_preserves_footprint(patch: GeoTensor, k: int) -> None:
    """Every quarter-turn keeps the input bounds and footprint (#126)."""
    out = augment.RandomRotate90(p=1.0)(patch, seed=_seed_for_k(k))
    np.testing.assert_array_equal(
        np.asarray(out), np.rot90(np.asarray(patch), k=k, axes=(-2, -1))
    )
    assert out.bounds == pytest.approx(patch.bounds)
    assert out.footprint().equals(patch.footprint())
    assert not np.shares_memory(np.asarray(out), np.asarray(patch))


def test_geometric_ops_do_not_alias_plain_array_input() -> None:
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    flipped = augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0)(arr)
    rotated = augment.RandomRotate90(p=1.0, seed=0)(arr)
    assert not np.shares_memory(flipped, arr)
    assert not np.shares_memory(rotated, arr)


_SEEDED_OPS: list[Any] = [
    lambda: augment.RandomFlip(seed=0),
    lambda: augment.RandomRotate90(p=1.0, seed=0),
    lambda: augment.RandomCrop(size=(2, 3), seed=0),
    lambda: augment.RandomShift(max_shift=(1, 1), seed=0),
    lambda: augment.BrightnessJitter(seed=0),
    lambda: augment.ContrastJitter(seed=0),
    lambda: augment.GaussianNoise(sigma=0.01, seed=0),
    lambda: augment.SpeckleNoise(sigma=0.05, seed=0),
    lambda: augment.BandDropout(p=0.5, seed=0),
    lambda: augment.BandJitter(groups={"vis": ["B02", "B03", "B04"]}, seed=0),
    lambda: augment.SunAngleJitter(seed=0),
    lambda: augment.AtmosphericHaze(seed=0),
    lambda: augment.SimulatedClouds(coverage=(0.1, 0.5), feather=1, seed=0),
    lambda: augment.CutMix(seed=0),
    lambda: augment.Compose(
        augmentations=[augment.RandomFlip(), augment.GaussianNoise(sigma=0.01)], seed=0
    ),
]


@pytest.mark.parametrize("make_op", _SEEDED_OPS, ids=lambda make: type(make()).__name__)
def test_seeded_operator_varies_across_calls_and_is_reproducible(
    patch: GeoTensor, make_op: Callable[[], Operator]
) -> None:
    """A seeded op draws a reproducible *sequence*, not one repeated draw (#134)."""
    # CutMix draws from a positional donor pool (#141).
    extra = (
        (_toy_geotensor(np.full((4, 5, 6), 9.0, dtype=np.float32)),)
        if isinstance(make_op(), augment.CutMix)
        else ()
    )
    op = make_op()
    draws = [np.asarray(op(patch, *extra)).copy() for _ in range(8)]
    # Successive calls apply different augmentations ...
    assert any(
        a.shape != draws[0].shape or not np.array_equal(a, draws[0]) for a in draws[1:]
    )
    # ... and a second instance with the same seed replays the same sequence.
    replay = make_op()
    for expected in draws:
        np.testing.assert_array_equal(np.asarray(replay(patch, *extra)), expected)
    # A reload from config restarts the stream from the constructor seed.
    if not op.forbid_in_yaml and not isinstance(op, augment.Compose):
        clone = Operator.from_state(json.loads(json.dumps(op.state)))
        np.testing.assert_array_equal(np.asarray(clone(patch, *extra)), draws[0])


def test_per_call_seed_is_one_off_and_leaves_stream_untouched(
    patch: GeoTensor,
) -> None:
    op = augment.GaussianNoise(sigma=0.01, seed=5)
    reference = augment.GaussianNoise(sigma=0.01, seed=5)
    np.testing.assert_array_equal(
        np.asarray(op(patch, seed=42)),
        np.asarray(augment.GaussianNoise(sigma=0.01)(patch, seed=42)),
    )
    # The one-off draw did not advance the instance stream.
    np.testing.assert_array_equal(np.asarray(op(patch)), np.asarray(reference(patch)))


def test_unseeded_compose_honours_seeded_children(patch: GeoTensor) -> None:
    pipe = augment.Compose(augmentations=[augment.GaussianNoise(sigma=0.01, seed=3)])
    expected = augment.GaussianNoise(sigma=0.01, seed=3)
    for _ in range(3):
        np.testing.assert_array_equal(
            np.asarray(pipe(patch)), np.asarray(expected(patch))
        )


def test_integer_carriers_round_instead_of_truncating() -> None:
    """Integer DNs are rounded to nearest, not truncated toward zero (#134)."""
    arr = np.full((1, 2, 2), 3, dtype=np.uint16)
    out = augment.BrightnessJitter(factor=(1.3, 1.3), per_band=False, seed=0)(arr)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, np.full((1, 2, 2), 4, dtype=np.uint16))


def test_boolean_carriers_pass_geometry_but_reject_radiometry() -> None:
    mask = np.zeros((1, 3, 4), dtype=bool)
    mask[0, 0, 0] = True
    flipped = augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0)(mask)
    assert flipped.dtype == np.bool_
    np.testing.assert_array_equal(flipped, np.flip(mask, -1))
    with pytest.raises(TypeError, match="boolean"):
        augment.GaussianNoise(sigma=0.01, seed=0)(mask)


_FILL = -9999.0
_RADIOMETRIC_OPS: list[Any] = [
    lambda: augment.BrightnessJitter(factor=(1.1, 1.1), seed=0),
    lambda: augment.BrightnessJitter(seed=0),
    lambda: augment.ContrastJitter(factor=(0.5, 1.5), seed=0),
    lambda: augment.GaussianNoise(sigma=0.5, seed=0),
    lambda: augment.SpeckleNoise(sigma=0.2, seed=0),
    lambda: augment.BandDropout(p=0.5, fill_value=-1.0, seed=2),
    lambda: augment.SunAngleJitter(delta_sza_deg=(10.0, 20.0), seed=0),
    lambda: augment.AtmosphericHaze(intensity=(0.5, 1.0), seed=0),
    lambda: augment.SimulatedClouds(coverage=(0.3, 0.6), feather=1, seed=0),
]


@pytest.mark.parametrize(
    "make_op", _RADIOMETRIC_OPS, ids=lambda make: type(make()).__name__
)
def test_fill_pixels_are_excluded(make_op: Callable[[], Operator]) -> None:
    """Radiometric ops leave nodata as nodata and ignore it in statistics (#145)."""
    clean = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6) / 10.0 + 1.0
    fill = fill_pixel_mask(clean.shape)
    valid = ~fill
    gt = _toy_geotensor(clean, fill_value_default=_FILL, with_fill_pixels=True)
    op = make_op()

    out = np.asarray(op(gt))

    # Fill pixels hold the output fill in every band (not e.g. -10998.9).
    np.testing.assert_array_equal(out[:, fill], _FILL)

    # Valid pixels match the same op (same seed -> same draws) run on data
    # without fill pixels.
    if isinstance(op, augment.ContrastJitter):
        # Put the valid-pixel band mean at the fill locations, so the plain
        # spatial mean of the reference equals the valid-only mean.
        reference_in = clean.copy()
        reference_in[:, fill] = clean[:, valid].mean(axis=1, keepdims=True)
        expected = np.asarray(make_op()(_toy_geotensor(reference_in)))
    elif isinstance(op, augment.SimulatedClouds):
        # The cloud brightness percentile must ignore nodata: compare with the
        # fill pixels absent (NaN, which the percentile already skipped).
        reference_in = clean.copy()
        reference_in[:, fill] = np.nan
        expected = make_op()(reference_in)
    else:
        expected = np.asarray(make_op()(_toy_geotensor(clean)))
    np.testing.assert_allclose(out[:, valid], expected[:, valid])


def test_fill_pixels_are_excluded_nan_fill_contrast() -> None:
    """One NaN pixel must not poison ContrastJitter's band mean (#145)."""
    clean = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    arr = clean.copy()
    arr[:, 1, 2] = np.nan
    valid = np.ones(arr.shape[-2:], dtype=bool)
    valid[1, 2] = False

    out = augment.ContrastJitter(factor=(0.5, 0.5), seed=0)(arr)

    assert np.isnan(out[:, 1, 2]).all()
    mean = clean[:, valid].mean(axis=1)[:, None]
    np.testing.assert_allclose(out[:, valid], (clean[:, valid] - mean) * 0.5 + mean)


def test_fill_pixels_are_excluded_cutmix_donor_holes() -> None:
    """Donor nodata pasted by CutMix becomes the input's fill value (#145)."""
    base = _toy_geotensor(np.ones((4, 5, 6), dtype=np.float32))
    donor = _toy_geotensor(
        np.full((4, 5, 6), 9.0, dtype=np.float32),
        fill_value_default=-1.0,
        with_fill_pixels=True,
    )
    # Draw until the pasted rectangle covers a donor fill pixel.
    for seed in range(100):
        out = np.asarray(augment.CutMix(p=1.0)(base, donor, seed=seed))
        if np.any(out == -1.0) or np.any(out == _FILL):
            break
    else:  # pragma: no cover - the draw space makes this unreachable
        pytest.fail("no CutMix draw covered a donor fill pixel")
    assert not np.any(out == -1.0)
    assert np.any(out == _FILL)
    assert set(np.unique(out)) <= {1.0, 9.0, _FILL}


def test_4d_time_stack() -> None:
    """The band axis of a (T, C, H, W) stack is -3, not time (#147)."""
    from _helpers import time_stack

    stack = time_stack()
    values = np.asarray(stack)

    # Per-band factors: one per band, shared by every frame.
    out = np.asarray(gz.augment.BrightnessJitter(factor=(0.5, 1.5), seed=0)(stack))
    ratio = out / values
    per_band = ratio[..., 0, 0]  # (T, C)
    np.testing.assert_allclose(
        ratio, np.broadcast_to(per_band[..., None, None], ratio.shape)
    )
    np.testing.assert_allclose(per_band[0], per_band[1])
    assert len(np.unique(np.round(per_band[0], 12))) == 3

    # Dropout removes bands of every frame, never whole frames.
    dropped = np.asarray(gz.augment.BandDropout(p=0.5, fill_value=0.0, seed=3)(stack))
    band_dropped = (dropped == 0.0).all(axis=(0, 2, 3))
    assert band_dropped.any() and not band_dropped.all()
    np.testing.assert_array_equal(dropped[:, ~band_dropped], values[:, ~band_dropped])

    # Jitter permutes bands (the same permutation in every frame).
    jitter = gz.augment.BandJitter(groups={"all": ["b0", "b1", "b2"]}, seed=1)
    permuted = np.asarray(jitter(stack))
    order = [
        next(j for j in range(3) if np.array_equal(permuted[0, c], values[0, j]))
        for c in range(3)
    ]
    assert sorted(order) == [0, 1, 2]
    np.testing.assert_array_equal(permuted, values[:, order])

    # Haze reads one wavelength per band (3), not per frame (2).
    hazed = np.asarray(gz.augment.AtmosphericHaze(intensity=0.1, seed=0)(stack))
    haze = hazed - values
    np.testing.assert_allclose(haze[0], haze[1])
    assert haze[0, 0, 0, 0] > haze[0, 1, 0, 0] > haze[0, 2, 0, 0]


# --- #155: Compose is a gated pipekit.Sequential ---------------------------


class _InitSeededOp(Operator):
    """Seed is constructor-only (like ``patch_ops.StratifiedSample``)."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def _apply(self, gt: np.ndarray) -> np.ndarray:
        return np.asarray(gt) + 1.0


class _ApplySeededOp(Operator):
    """Takes the per-call seed in ``_apply`` but has no ``seed`` in ``__init__``."""

    def __init__(self) -> None:
        self.seen: list[int | None] = []

    def _apply(self, gt: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        self.seen.append(seed)
        return np.asarray(gt)


class _KwargsSeededOp(Operator):
    """Consumes the per-call seed through ``**kwargs``."""

    def __init__(self) -> None:
        self.seen: list[int | None] = []

    def _apply(self, gt: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.seen.append(kwargs.get("seed"))
        return np.asarray(gt)


def test_compose_forwards_seed_to_var_keyword_apply() -> None:
    child = _KwargsSeededOp()
    augment.Compose(augmentations=[child], seed=0)(
        np.zeros((1, 2, 2), dtype=np.float32)
    )
    assert len(child.seen) == 1
    assert isinstance(child.seen[0], int)


def test_compose_forwards_seed_only_when_apply_accepts_it() -> None:
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    init_seeded = _InitSeededOp(seed=3)
    apply_seeded = _ApplySeededOp()
    pipe = augment.Compose(augmentations=[init_seeded, apply_seeded], seed=0)

    # Used to raise TypeError: `seed` was forwarded because __init__ took one.
    out = pipe(arr)
    np.testing.assert_array_equal(out, arr + 1.0)
    out = pipe(arr, seed=5)
    assert len(apply_seeded.seen) == 2
    assert all(isinstance(seed, int) for seed in apply_seeded.seen)

    # The same top-level seed derives the same child seed.
    again = _ApplySeededOp()
    augment.Compose(augmentations=[_InitSeededOp(), again], seed=0)(arr)
    assert again.seen == apply_seeded.seen[:1]

    # An unseeded Compose forwards nothing.
    unseeded = _ApplySeededOp()
    augment.Compose(augmentations=[unseeded])(arr)
    assert unseeded.seen == [None]


def test_compose_is_a_gated_sequential(patch: GeoTensor) -> None:
    flip = augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)
    pipe = augment.Compose(augmentations=[flip], p=0.0, seed=0)
    assert isinstance(pipe, Sequential)
    assert pipe.augmentations is pipe.operators
    assert pipe.get_config() == {
        "augmentations": Sequential([flip]).get_config()["operators"],
        "p": 0.0,
        "seed": 0,
    }
    assert "Compose(augmentations=" in repr(pipe)
    # `compose | op` keeps the gate: p=0 skips the flip, Identity passes through.
    chained = pipe | gz.Identity()
    assert isinstance(chained, Sequential)
    assert chained.operators[0] is pipe
    assert chained(patch) is patch
    with pytest.raises(TypeError, match="must be an Operator"):
        augment.Compose(augmentations=[{"class": "x"}])  # type: ignore[list-item]


def test_compose_is_not_a_top_level_export() -> None:
    # `gz.compose`-style names belong to pipekit's right-to-left composer.
    assert not hasattr(gz, "Compose")
    assert gz.augment.Compose is augment.Compose


# ---------------------------------------------------------------------------
# Non-default grids (``_helpers.TOY_GRIDS``)
# ---------------------------------------------------------------------------


def _pixel_centres(gt: GeoTensor) -> dict[float, tuple[float, float]]:
    """World coordinate of every pixel centre, keyed by its (unique) value."""
    rows, cols = np.mgrid[0 : gt.shape[-2], 0 : gt.shape[-1]]
    xs, ys = gt.transform * (cols + 0.5, rows + 0.5)
    values = np.asarray(gt)[0]
    return {
        float(v): (float(x), float(y))
        for v, x, y in zip(values.ravel(), xs.ravel(), ys.ravel(), strict=True)
    }


@pytest.mark.parametrize("grid", ["non_square", "rotated", "sheared", "geographic"])
@pytest.mark.parametrize(
    "make_op",
    [
        lambda: augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0),
        lambda: augment.RandomFlip(p_horizontal=0.0, p_vertical=1.0),
        # Seeds 0 / 1 / 2 draw k = 2 / 3 / 1 quarter turns.
        *(
            (lambda seed=seed: augment.RandomRotate90(p=1.0, seed=seed))
            for seed in (0, 1, 2)
        ),
    ],
)
def test_geometric_augments_keep_every_pixel_in_place(
    grid: str, make_op: Callable[[], Operator]
) -> None:
    """Each output pixel sits where its value sat on the input grid (#126).

    A rigid flip / rotation only re-indexes pixels; on anisotropic, rotated,
    sheared and geographic grids the transform must follow exactly.
    """
    values = np.arange(4 * 6, dtype=np.float64).reshape(1, 4, 6) + 1.0
    patch = toy_geotensor(values, grid=grid)
    out = make_op()(patch)
    before = _pixel_centres(patch)
    for value, xy in _pixel_centres(out).items():
        np.testing.assert_allclose(xy, before[value], rtol=0, atol=1e-9)
