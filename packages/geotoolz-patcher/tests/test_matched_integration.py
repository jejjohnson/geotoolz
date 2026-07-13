"""Integration tests for the matched-field pipeline on real numeric data.

The Round 1 PR landed `MatchedField` / `MatchedSpatialPatcher` with
stub-driven unit tests in ``test_matched_e2e.py``. This module
exercises the **full** pipeline with concrete `RasterField`s,
a real `SpatialPatcher` (geometry + sampler + window +
aggregation), and numeric data — proving the carriers behave
correctly when the per-source data is genuine numpy arrays.

Coverage focus:

* Identity-coreg pipeline produces members equal to direct
  field reads (no metadata corruption).
* A non-identity coreg callable (the kind of thing
  ``geotoolz.geom.coregister.RasterToRasterLike`` would do)
  actually transforms the secondary data.
* `valid_mask=True` populates per-source masks from real NaN
  data — the P1 fix from the earlier review.
* Multi-secondary `MatchedField` fans out correctly and
  per-source aggregators reduce each member independently.
* Merge round-trip: split into patches, run the operator on
  each, merge back — the reconstructed field shape and grid
  match the primary's.

No external geotoolz dependency: we model "coregistration" with
plain Python callables to avoid pulling in rasterio's warp
machinery (which would dominate the test runtime and add a
network-or-build burden).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher._src.fields.raster import RasterField
from geopatcher._src.matched import (
    MatchedField,
    MatchedPatch,
    MatchedSpatialPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY
from geopatcher._src.spatial.aggregation import SpatialMean, SpatialSum
from geopatcher._src.spatial.geometry import SpatialRectangular
from geopatcher._src.spatial.patcher import SpatialPatcher
from geopatcher._src.spatial.sampler import SpatialRegularStride
from geopatcher._src.spatial.window import SpatialBoxcar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(values: np.ndarray) -> GeoTensor:
    """10m-pixel UTM 29N GeoTensor."""
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=np.nan,
    )


def _checkerboard(shape: tuple[int, int], stride: int = 2) -> np.ndarray:
    """Distinctive pattern so we can read back which pixel each patch holds."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = float(i * 100 + j) + (
                10.0 if (i // stride + j // stride) % 2 else 0.0
            )
    return out


# ---------------------------------------------------------------------------
# Pipeline shape — identity coreg, single secondary
# ---------------------------------------------------------------------------


class TestIdentityCoregFullPipeline:
    """An identity coreg callable returns the secondary's raw data
    unchanged; after split/merge we expect both members to match
    their source field bit-for-bit."""

    def _build(self) -> tuple[MatchedSpatialPatcher, MatchedField, GeoTensor]:
        primary_tensor = _gt(_checkerboard((16, 16)))
        secondary_tensor = _gt(_checkerboard((16, 16)) + 1000.0)
        mf = MatchedField(
            primary=RasterField(primary_tensor),
            secondaries={"sec": RasterField(secondary_tensor)},
            coreg={"sec": lambda raw, prim: raw},
        )
        primary_patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(4, 4)),
            sampler=SpatialRegularStride(step=(4, 4)),
            window=SpatialBoxcar(),
            aggregation=SpatialSum(),
        )
        msp = MatchedSpatialPatcher(
            primary=primary_patcher,
            secondary_aggregators={"sec": SpatialSum()},
        )
        return msp, mf, primary_tensor

    def test_yields_4x4_grid_of_patches(self) -> None:
        msp, mf, _ = self._build()
        patches = list(msp.split(mf))
        # 16x16 field with 4x4 patches at stride 4 → 4x4 = 16 anchors.
        assert len(patches) == 16
        for mp in patches:
            assert isinstance(mp, MatchedPatch)
            assert set(mp.members) == {PRIMARY_KEY, "sec"}

    def test_member_values_come_from_correct_field(self) -> None:
        msp, mf, _primary_tensor = self._build()
        for mp in msp.split(mf):
            primary_arr = np.asarray(mp.members[PRIMARY_KEY].data)
            secondary_arr = np.asarray(mp.members["sec"].data)
            # Identity coreg → secondary is the same pattern as
            # primary + 1000 across every pixel.
            np.testing.assert_array_equal(secondary_arr, primary_arr + 1000.0)

    def test_inner_patch_indices_mirror_outer(self) -> None:
        msp, mf, _ = self._build()
        for mp in msp.split(mf):
            for member_patch in mp.members.values():
                assert member_patch.anchor == mp.anchor
                assert member_patch.indices is not None


# ---------------------------------------------------------------------------
# Non-identity coreg — verify the callable actually runs
# ---------------------------------------------------------------------------


class TestNonIdentityCoreg:
    """A coreg callable that applies a known transform proves the
    output of `MatchedField.select` reflects the callable's work.

    Mimics what ``geotoolz.geom.coregister.RasterToRasterLike`` would
    do (apply a warp / resample) without dragging rasterio in.
    """

    def test_coreg_offset_applied_per_patch(self) -> None:
        # Primary is *non-zero* and varies across patches so the
        # coreg's `shift_by_primary_mean` produces a different
        # output per patch — proves the callable actually runs
        # (a no-op pipeline that skipped coreg entirely would
        # produce all 7s and silently pass an all-zero-primary
        # version of this test).
        primary_values = np.fromfunction(
            lambda i, _j: i.astype(np.float32), (8, 8)
        )  # row index 0..7 in every column
        primary_tensor = _gt(primary_values)
        secondary_tensor = _gt(np.ones((8, 8), dtype=np.float32) * 7.0)

        def shift_by_primary_mean(raw: Any, primary: Any) -> Any:
            primary_arr = np.asarray(primary)
            return np.asarray(raw) + primary_arr.mean()

        mf = MatchedField(
            primary=RasterField(primary_tensor),
            secondaries={"sec": RasterField(secondary_tensor)},
            coreg={"sec": shift_by_primary_mean},
        )
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            )
        )
        # Expected per-patch primary means:
        # - top patches (rows 0-3):    mean = 1.5 → secondary = 8.5
        # - bottom patches (rows 4-7): mean = 5.5 → secondary = 12.5
        observed = {
            mp.anchor[0]: float(np.asarray(mp.members["sec"].data).mean())
            for mp in msp.split(mf)
        }
        # Anchor row 0 = top patches, row 4 = bottom patches.
        assert observed[0] == pytest.approx(8.5)
        assert observed[4] == pytest.approx(12.5)
        # If coreg had been skipped entirely, both would equal 7.0.
        assert observed[0] != 7.0
        assert observed[4] != 7.0

    def test_coreg_callable_called_with_correct_arity(self) -> None:
        primary_tensor = _gt(np.ones((4, 4), dtype=np.float32) * 3.0)
        secondary_tensor = _gt(np.ones((4, 4), dtype=np.float32) * 5.0)

        calls: list[tuple[float, float]] = []

        def record_means(raw: Any, primary: Any) -> Any:
            # The contract: coreg receives (secondary_raw, primary).
            # If swapped, we'd record (3, 5) instead of (5, 3).
            calls.append(
                (float(np.asarray(raw).mean()), float(np.asarray(primary).mean()))
            )
            return raw

        mf = MatchedField(
            primary=RasterField(primary_tensor),
            secondaries={"sec": RasterField(secondary_tensor)},
            coreg={"sec": record_means},
        )
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            )
        )
        list(msp.split(mf))
        assert calls == [(5.0, 3.0)]  # (secondary_raw, primary)


# ---------------------------------------------------------------------------
# valid_mask on real NaN-bearing data (the P1 review fix)
# ---------------------------------------------------------------------------


class TestValidMaskOnRealData:
    def test_nan_pixels_marked_invalid(self) -> None:
        # Primary has NaN in a 2x2 corner block; secondary has NaN
        # in a different block. After split, each MatchedPatch's
        # valid_mask should mirror the NaN positions per source.
        primary_values = np.ones((8, 8), dtype=np.float32)
        primary_values[0:2, 0:2] = np.nan
        secondary_values = np.ones((8, 8), dtype=np.float32) * 2.0
        secondary_values[6:8, 6:8] = np.nan

        mf = MatchedField(
            primary=RasterField(_gt(primary_values)),
            secondaries={"sec": RasterField(_gt(secondary_values))},
            coreg={"sec": lambda raw, prim: raw},
        )
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            )
        )
        patches = list(msp.split(mf))
        # 4 patches: top-left (has primary NaN), top-right, bottom-left,
        # bottom-right (has secondary NaN).
        assert len(patches) == 4

        # Locate the patch whose anchor is at the top-left corner.
        top_left = next(p for p in patches if p.anchor == (0, 0))
        assert top_left.valid_mask is not None
        primary_mask = top_left.valid_mask[PRIMARY_KEY]
        assert primary_mask.shape == (4, 4)
        # The 2x2 NaN block sits in the top-left of this 4x4 patch.
        assert primary_mask[0:2, 0:2].sum() == 0
        # Rest of the patch should be valid.
        assert primary_mask[2:, :].all() and primary_mask[:, 2:].all()
        # Secondary mask in this patch is all-valid (NaN is elsewhere).
        assert top_left.valid_mask["sec"].all()

        # The bottom-right patch should be the secondary NaN one.
        bot_right = next(p for p in patches if p.anchor == (4, 4))
        sec_mask = bot_right.valid_mask["sec"]
        assert sec_mask[2:4, 2:4].sum() == 0
        assert bot_right.valid_mask[PRIMARY_KEY].all()

    def test_valid_mask_false_disables_computation(self) -> None:
        # When the user opts out, no masks are computed even for
        # NaN-bearing data. Lets downstream code skip the mask
        # dict allocation entirely.
        primary_values = np.full((4, 4), np.nan, dtype=np.float32)
        mf = MatchedField(
            primary=RasterField(_gt(primary_values)),
            valid_mask=False,
        )
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            )
        )
        patches = list(msp.split(mf))
        for mp in patches:
            assert mp.valid_mask is None


# ---------------------------------------------------------------------------
# Multi-secondary fan-out — each role aggregated independently
# ---------------------------------------------------------------------------


class TestMultiSecondaryFanOut:
    def test_three_member_pipeline(self) -> None:
        # Primary is zeros, s2 is 2s, landsat is 3s. Identity coreg
        # for both secondaries. After split/merge with SpatialSum,
        # each role's reconstructed array reflects the role's
        # constant value x patch count.
        primary = _gt(np.zeros((8, 8), dtype=np.float32))
        s2 = _gt(np.full((8, 8), 2.0, dtype=np.float32))
        landsat = _gt(np.full((8, 8), 3.0, dtype=np.float32))

        mf = MatchedField(
            primary=RasterField(primary),
            secondaries={
                "s2": RasterField(s2),
                "landsat": RasterField(landsat),
            },
            coreg={
                "s2": lambda raw, prim: raw,
                "landsat": lambda raw, prim: raw,
            },
        )

        # Non-overlapping 4x4 patches → SpatialSum reduces to one
        # value per pixel (no accumulation across patches at the
        # global field level).
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            ),
            secondary_aggregators={
                "s2": SpatialSum(),
                "landsat": SpatialSum(),
            },
        )

        patches = list(msp.split(mf))
        assert len(patches) == 4  # 2x2 stride grid
        merged = msp.merge(patches, mf)
        assert set(merged) == {PRIMARY_KEY, "s2", "landsat"}

        # The merged primary is all zeros; s2 is all 2s; landsat
        # is all 3s — because we used non-overlapping patches with
        # boxcar windows + SpatialSum, every pixel is touched once.
        primary_merged = np.asarray(merged[PRIMARY_KEY])
        s2_merged = np.asarray(merged["s2"])
        landsat_merged = np.asarray(merged["landsat"])
        assert primary_merged.shape == (8, 8)
        np.testing.assert_array_equal(primary_merged, 0.0)
        np.testing.assert_array_equal(s2_merged, 2.0)
        np.testing.assert_array_equal(landsat_merged, 3.0)

    def test_different_aggregators_per_source(self) -> None:
        # Primary uses Sum; secondary uses Mean. Both inputs identical
        # so reconstructed values differ predictably.
        constant = _gt(np.full((8, 8), 5.0, dtype=np.float32))
        mf = MatchedField(
            primary=RasterField(constant),
            secondaries={"sec": RasterField(constant)},
            coreg={"sec": lambda raw, prim: raw},
        )
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            ),
            secondary_aggregators={"sec": SpatialMean()},
        )
        patches = list(msp.split(mf))
        merged = msp.merge(patches, mf)
        # Sum and Mean on non-overlapping patches reduce to the
        # same answer (the local value, since each pixel is touched
        # exactly once with weight 1.0). Both should yield 5.0
        # everywhere.
        np.testing.assert_array_equal(np.asarray(merged[PRIMARY_KEY]), 5.0)
        np.testing.assert_array_equal(np.asarray(merged["sec"]), 5.0)


# ---------------------------------------------------------------------------
# Round-trip identity — split then merge reconstructs the field
# ---------------------------------------------------------------------------


class TestRoundTripIdentity:
    def test_non_overlapping_tiles_reconstruct_exactly(self) -> None:
        # With non-overlapping patches and SpatialSum on a primary
        # with arbitrary content, the reconstructed field equals
        # the original (each pixel is read once and accumulated
        # once with no overlap).
        original = _gt(_checkerboard((16, 16)))
        mf = MatchedField(primary=RasterField(original))
        msp = MatchedSpatialPatcher(
            primary=SpatialPatcher(
                geometry=SpatialRectangular(size=(4, 4)),
                sampler=SpatialRegularStride(step=(4, 4)),
                window=SpatialBoxcar(),
                aggregation=SpatialSum(),
            )
        )
        patches = list(msp.split(mf))
        merged = msp.merge(patches, mf)
        reconstructed = np.asarray(merged[PRIMARY_KEY])
        np.testing.assert_array_equal(reconstructed, np.asarray(original))
