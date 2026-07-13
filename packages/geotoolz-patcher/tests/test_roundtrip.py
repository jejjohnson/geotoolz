"""Hypothesis property tests for split/merge round-trip — issue #21.

Four orthogonal axes (Geometry x Sampler x Window x Aggregation) yield
a huge combinatorial space; example-based tests miss interactions.
This module pins down round-trip properties that must hold for
*every* valid configuration, with Hypothesis driving the input space
so failing cases shrink to a minimal `(shape, stride, seed)` triple.

The properties (with the operator fixed to identity):

- **Disjoint Sum reconstruction.** `SpatialRectangular x boxcar x Sum`
  with non-overlapping anchors yields `merge(split(f)) == f` exactly
  on the touched cells.
- **Disjoint Mean reconstruction.** Same setup with `SpatialMean`.
- **Disjoint OverlapAdd reconstruction.** Same setup with
  `SpatialOverlapAdd`; the boxcar trivially satisfies COLA at
  stride==patch_size, so reconstruction is bit-exact.
- **Constant-field idempotence.** For any (geometry, sampler) and
  aggregations that preserve constants (`Max`, `Min`, `Mean`), the
  touched region of a constant field maps to the same constant.
- **Anchor-count contract.** `SpatialRegularStride.anchors(...)` with
  `boundary="drop"` returns exactly the integer lattice count
  determined by the domain shape, patch size, and stride.
"""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from hypothesis import HealthCheck, given, settings, strategies as st

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialMax,
    SpatialMean,
    SpatialMin,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialSum,
)


def _identity(arr):
    return arr


def _field(shape: tuple[int, int], fill: float | None = None) -> RasterField:
    if fill is None:
        # Non-repeating ramp so any indexing bug shows up.
        arr = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    else:
        arr = np.full(shape, fill, dtype=np.float64)
    # Stay in float64 end-to-end — float32 storage was underflowing
    # Hypothesis-generated tiny fills (~3e-120) to zero, masking the
    # constant-field property.
    return RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )


# Strategy: a `(domain_size, patch_size)` pair where the domain divides
# evenly by the patch size — so stride==patch_size produces a complete
# non-overlapping tiling and reconstruction is exact (no edge residual
# under boundary="drop").
@st.composite
def _aligned_shape_and_patch(draw) -> tuple[int, int]:
    patch = draw(st.integers(min_value=4, max_value=16))
    tiles = draw(st.integers(min_value=2, max_value=8))
    return (patch * tiles, patch)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sp=_aligned_shape_and_patch())
def test_disjoint_sum_reconstructs_field(sp: tuple[int, int]) -> None:
    domain, patch = sp
    field = _field((domain, domain))
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(patch, patch)),
        sampler=SpatialRegularStride(step=patch),
        window=SpatialBoxcar(),
        aggregation=SpatialSum(),
    )
    patches = [
        type(p)(
            data=_identity(p.data.values),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    result = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(
        result, np.asarray(field.reader.values, dtype=np.float64)
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sp=_aligned_shape_and_patch())
def test_disjoint_mean_reconstructs_field(sp: tuple[int, int]) -> None:
    # Disjoint tiling → each cell is touched by exactly one patch, so
    # Mean(one-element) == that element == the original field.
    domain, patch = sp
    field = _field((domain, domain))
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(patch, patch)),
        sampler=SpatialRegularStride(step=patch),
        window=SpatialBoxcar(),
        aggregation=SpatialMean(),
    )
    patches = [
        type(p)(
            data=_identity(p.data.values),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    result = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(
        result, np.asarray(field.reader.values, dtype=np.float64)
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sp=_aligned_shape_and_patch())
def test_disjoint_overlap_add_reconstructs_field(sp: tuple[int, int]) -> None:
    # SpatialBoxcar satisfies COLA trivially at stride == patch_size
    # (constant-overlap = 1 everywhere). With identity op, OverlapAdd
    # = sum w*x / sum w = field on every touched cell.
    domain, patch = sp
    field = _field((domain, domain))
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(patch, patch)),
        sampler=SpatialRegularStride(step=patch),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    patches = [
        type(p)(
            data=_identity(p.data.values),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    result = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(
        result, np.asarray(field.reader.values, dtype=np.float64)
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    sp=_aligned_shape_and_patch(),
    agg=st.sampled_from([SpatialMax(), SpatialMin(), SpatialMean()]),
    fill=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
)
def test_constant_field_preserves_constant(
    sp: tuple[int, int], agg, fill: float
) -> None:
    domain, patch = sp
    field = _field((domain, domain), fill=fill)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(patch, patch)),
        sampler=SpatialRegularStride(step=patch),
        window=SpatialBoxcar(),
        aggregation=agg,
    )
    patches = [
        type(p)(
            data=_identity(p.data.values),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    result = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(
        result, np.full((domain, domain), fill, dtype=np.float64)
    )


@settings(max_examples=80, deadline=None)
@given(
    h=st.integers(min_value=2, max_value=64),
    w=st.integers(min_value=2, max_value=64),
    patch=st.integers(min_value=2, max_value=12),
    stride=st.integers(min_value=2, max_value=8),
)
def test_regular_stride_anchor_count_matches_formula(
    h: int, w: int, patch: int, stride: int
) -> None:
    # The drop-mode formula: floor((D - P) / S) + 1 along each axis,
    # clamped to >= 1 when P > D. Patch / stride / domain bounds
    # deliberately overlap (patch up to 12, domain down to 2) so the
    # patch > domain branch is part of the property — the clamp comment
    # would be a lie otherwise.
    arr = np.zeros((h, w), dtype=np.float32)
    field = RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )
    geom = SpatialRectangular(size=(patch, patch))
    sampler = SpatialRegularStride(step=stride)
    expected_rows = max((h - patch) // stride + 1, 1)
    expected_cols = max((w - patch) // stride + 1, 1)
    expected = expected_rows * expected_cols
    actual = sum(1 for _ in sampler.anchors(field.domain, geom))
    assert actual == expected


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_random_sampler_split_is_deterministic_under_seed(seed: int) -> None:
    # End-to-end determinism: under a fixed int seed, two split() calls
    # of a `SpatialRandom`-driven patcher yield the same patch
    # sequence by anchor identity. The contract #21 leans on for
    # shrinking and replay.
    field = _field((32, 32))
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRandom(n_samples=10, seed=seed),
        window=SpatialBoxcar(),
        aggregation=SpatialSum(),
    )
    a = [p.anchor for p in patcher.split(field)]
    b = [p.anchor for p in patcher.split(field)]
    assert a == b
