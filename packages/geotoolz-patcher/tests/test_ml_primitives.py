"""Primitives that make user-rolled torch / JAX / Grain datasets ergonomic.

The three additions are tightly coupled by intent — they let a user
write a torch `Dataset`, a Grain `RandomAccessDataSource`, or a JAX
vmap pipeline without reaching into the patcher's internals:

- ``patcher.anchors(field) -> list[Anchor]`` is the indexable anchor
  list (``__len__`` + ``__getitem__`` on the dataset side).
- ``patcher.patch_at(field, anchor) -> Patch`` is the lazy single-read
  the dataset's ``__getitem__`` calls per index.
- ``geopatcher.stack_patches(patches)`` returns
  ``(N, *patch_shape)`` for `vmap`-style batched models.

The contract that ties them together: **the patch returned by
`patch_at(field, anchors(field)[i])` must equal the i-th patch from
`split(field)` bit-for-bit.** Without that, dataset replay between
runs and dataset/split parity in tests both silently break.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialMean,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    TemporalCausalBoxcar,
    TemporalFixedLookback,
    TemporalMean,
    TemporalMultiScale,
    TemporalPatcher,
    TemporalRegularStride,
    stack_patches,
)


@pytest.fixture
def spatial_patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


@pytest.fixture
def series() -> np.ndarray:
    return np.arange(100, dtype=np.float64)


@pytest.fixture
def temporal_patcher() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=5),
        sampler=TemporalRegularStride(step=10),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )


# ---------------------------------------------------------------------------
# Spatial: anchors / patch_at / split-parity
# ---------------------------------------------------------------------------


class TestSpatialAnchors:
    def test_matches_split_anchor_sequence(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        from_anchors = spatial_patcher.anchors(field)
        from_split = [p.anchor for p in spatial_patcher.split(field)]
        assert from_anchors == from_split

    def test_is_indexable_and_supports_len(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        # The torch / Grain dataset contract: len() + __getitem__.
        anchors = spatial_patcher.anchors(field)
        assert len(anchors) == spatial_patcher.n_anchors(field)
        first, last = anchors[0], anchors[-1]
        assert isinstance(first, tuple)
        assert isinstance(last, tuple)


class TestSpatialPatchAt:
    def test_matches_corresponding_split_patch(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        # Every i-th patch from patch_at must equal the i-th patch
        # from split — the parity that makes torch/Grain replay work.
        from_split = list(spatial_patcher.split(field))
        anchors = spatial_patcher.anchors(field)
        for i, anchor in enumerate(anchors):
            via_patch_at = spatial_patcher.patch_at(field, anchor)
            assert via_patch_at.anchor == from_split[i].anchor
            np.testing.assert_array_equal(
                np.asarray(via_patch_at.data),
                np.asarray(from_split[i].data),
            )
            assert via_patch_at.indices == from_split[i].indices

    def test_returns_a_patch_instance(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        anchors = spatial_patcher.anchors(field)
        result = spatial_patcher.patch_at(field, anchors[0])
        assert isinstance(result, Patch)

    def test_independent_of_iteration_state(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        # patch_at must not advance any shared iterator — repeated
        # reads of the same anchor must return the same patch.
        anchors = spatial_patcher.anchors(field)
        first = spatial_patcher.patch_at(field, anchors[3])
        second = spatial_patcher.patch_at(field, anchors[3])
        np.testing.assert_array_equal(np.asarray(first.data), np.asarray(second.data))

    def test_works_with_seeded_random_sampler(self, field: RasterField) -> None:
        # Regression for the "stochastic samplers" caveat in
        # SpatialPatcher.n_anchors: with a fixed int seed, patch_at
        # at the i-th anchor still matches the i-th patch from split.
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRandom(n_samples=8, seed=42),
            window=SpatialBoxcar(),
            aggregation=SpatialMean(),
        )
        anchors = patcher.anchors(field)
        from_split = list(patcher.split(field))
        for i, anchor in enumerate(anchors):
            p = patcher.patch_at(field, anchor)
            assert p.anchor == from_split[i].anchor
            np.testing.assert_array_equal(
                np.asarray(p.data), np.asarray(from_split[i].data)
            )


# ---------------------------------------------------------------------------
# Async mirror — sanity check (one happy path)
# ---------------------------------------------------------------------------


class _SyncAsyncField:
    """Adapter — wraps a sync `RasterField` so `select` is awaitable."""

    def __init__(self, sync_field: RasterField) -> None:
        self._sync = sync_field

    @property
    def domain(self):
        return self._sync.domain

    async def select(self, window):
        return self._sync.select(window)

    def with_data(self, array):
        return self._sync.with_data(array)


def test_async_patch_at_matches_async_split(field: RasterField) -> None:
    # Driven via asyncio.run() to keep the dev dep graph clean — no
    # pytest-asyncio required just for one happy-path async test.
    import asyncio

    from geopatcher import AsyncSpatialPatcher

    # `_SyncAsyncField` already satisfies the AsyncField protocol
    # (domain, async select, with_data) — no need to re-wrap in
    # AsyncRasterField, which would just double-shim and break domain
    # dispatch.
    async_field = _SyncAsyncField(field)
    patcher = AsyncSpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )

    async def _run() -> None:
        anchors = patcher.anchors(async_field)
        from_split = [p async for p in patcher.split(async_field)]
        for i, anchor in enumerate(anchors):
            p = await patcher.patch_at(async_field, anchor)
            assert p.anchor == from_split[i].anchor
            np.testing.assert_array_equal(
                np.asarray(p.data), np.asarray(from_split[i].data)
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Temporal: patches_at handles multi-scale; anchors matches split
# ---------------------------------------------------------------------------


class TestTemporalAnchors:
    def test_matches_split_anchor_sequence(
        self, temporal_patcher: TemporalPatcher, series: np.ndarray
    ) -> None:
        # len(anchors) == n_anchors for single-slice geometries
        # (multi-scale is tested separately below).
        from_anchors = temporal_patcher.anchors(series)
        from_split = [p.anchor for p in temporal_patcher.split(series)]
        assert from_anchors == from_split


class TestTemporalPatchesAt:
    def test_single_slice_returns_list_of_one(
        self, temporal_patcher: TemporalPatcher, series: np.ndarray
    ) -> None:
        anchors = temporal_patcher.anchors(series)
        patches = temporal_patcher.patches_at(series, anchors[0])
        assert isinstance(patches, list)
        assert len(patches) == 1

    def test_multi_scale_returns_one_per_scale(self, series: np.ndarray) -> None:
        # Regression for the multi-scale fix in #37: TemporalMultiScale
        # emits len(scales) patches per anchor, and patches_at must
        # return the full list — not just the first slice.
        tp = TemporalPatcher(
            geometry=TemporalMultiScale(scales=[5, 20, 50]),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        anchors = tp.anchors(series)
        patches = tp.patches_at(series, anchors[0])
        assert len(patches) == 3

    def test_parity_with_split(
        self, temporal_patcher: TemporalPatcher, series: np.ndarray
    ) -> None:
        # Concatenating patches_at across anchors must reproduce split
        # exactly — the contract the torch/Grain dataset path leans on.
        from_split = list(temporal_patcher.split(series))
        anchors = temporal_patcher.anchors(series)
        reconstructed = [
            p for anchor in anchors for p in temporal_patcher.patches_at(series, anchor)
        ]
        assert len(reconstructed) == len(from_split)
        for a, b in zip(reconstructed, from_split, strict=True):
            np.testing.assert_array_equal(a.data, b.data)
            assert a.anchor == b.anchor
            assert a.indices == b.indices


# ---------------------------------------------------------------------------
# stack_patches: shape, dtype, attr, error modes
# ---------------------------------------------------------------------------


class TestStackPatches:
    def test_stacks_data_to_n_plus_shape(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        patches = list(spatial_patcher.split(field))
        stacked = stack_patches(patches)
        # 4x4 lattice = 16 anchors, each patch is 16x16.
        assert stacked.shape == (16, 16, 16)
        # First patch in the stack matches the first split patch.
        np.testing.assert_array_equal(stacked[0], np.asarray(patches[0].data))

    def test_attr_weights_stacks_window_weights(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        patches = list(spatial_patcher.split(field))
        stacked = stack_patches(patches, attr="weights")
        assert stacked.shape == (16, 16, 16)
        # SpatialBoxcar yields all-ones weights.
        np.testing.assert_array_equal(stacked, 1.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty input"):
            stack_patches([])

    def test_none_attribute_raises(self) -> None:
        # A patch with weights=None must give a clear error when
        # attr="weights" — without this, np.stack would either succeed
        # (object dtype) or fail with a numpy-internal message.
        from rasterio.windows import Window

        p = Patch(
            data=np.zeros((4, 4)),
            anchor=(0, 0),
            indices=Window(col_off=0, row_off=0, width=4, height=4),
            weights=None,
        )
        with pytest.raises(ValueError, match="weights=None"):
            stack_patches([p], attr="weights")

    def test_shape_mismatch_raises_with_helpful_message(self) -> None:
        # Surfaces ragged geometries early. The error should name the
        # first mismatching patch and gesture at SpatialRadiusGraph /
        # SpatialPolygonIntersection as the typical culprits.
        from rasterio.windows import Window

        p1 = Patch(
            data=np.zeros((4, 4)),
            anchor=(0, 0),
            indices=Window(col_off=0, row_off=0, width=4, height=4),
            weights=None,
        )
        p2 = Patch(
            data=np.zeros((5, 5)),
            anchor=(4, 4),
            indices=Window(col_off=4, row_off=4, width=5, height=5),
            weights=None,
        )
        with pytest.raises(ValueError, match=r"patch 1.*shape"):
            stack_patches([p1, p2])

    def test_dataset_round_trip(
        self, spatial_patcher: SpatialPatcher, field: RasterField
    ) -> None:
        # The realistic flow: enumerate anchors, build a Dataset-style
        # list comprehension of patch_at, stack for vmap. This is the
        # idiom the recipe notebooks lean on; pin it here so a future
        # change to any one primitive can't break the chain silently.
        anchors = spatial_patcher.anchors(field)
        patches = [spatial_patcher.patch_at(field, a) for a in anchors]
        batch = stack_patches(patches)
        assert batch.shape[0] == len(anchors)
