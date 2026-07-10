"""Unit tests for `RasterToRasterLike` and `StackMatched`.

Both operators are the first concrete implementations from
``docs/design/query-matchup.md`` §5. They underpin the geocatalog/
geopatcher matchup workflow: `RasterToRasterLike` is the typical
default for `MatchedField.coreg` on raster↔raster pairs, and
`StackMatched` fuses the per-source patches returned by a
`MatchedSpatialPatcher` into a single multi-band tensor.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from _helpers import toy_geotensor
from georeader.geotensor import GeoTensor
from pipekit import Operator

import geotoolz as gz
from geotoolz.compositing import StackMatched
from geotoolz.geom.coregister import RasterToRasterLike


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt(
    values: np.ndarray,
    *,
    transform: rasterio.Affine | None = None,
    crs: str = "EPSG:32629",
) -> GeoTensor:
    return toy_geotensor(
        values, transform=transform, crs=crs, fill_value_default=np.nan
    )


# ---------------------------------------------------------------------------
# RasterToRasterLike
# ---------------------------------------------------------------------------


class TestRasterToRasterLike:
    def test_is_operator_with_config(self) -> None:
        op = RasterToRasterLike(resampling="cubic")
        assert isinstance(op, Operator)
        assert op.get_config() == {"resampling": "cubic"}

    def test_same_grid_passes_through_unchanged(self) -> None:
        # When src and like share grid + CRS, ReprojectLike (which
        # this op delegates to) is a no-op warp — the pixel values
        # come through bit-equal because no resampling is applied.
        src = _gt(np.arange(64, dtype=np.float32).reshape(8, 8))
        like = _gt(np.zeros((8, 8), dtype=np.float32))
        out = RasterToRasterLike()(src, like)
        assert out.shape == like.shape
        assert out.crs == like.crs
        assert out.transform == like.transform
        np.testing.assert_array_equal(np.asarray(out), np.asarray(src))

    def test_aligns_to_like_grid(self) -> None:
        # `src` is on a 5 m grid in the lower-left corner; `like` is
        # on a 10 m grid covering the same area. After RTRL, the
        # output should share `like`'s shape + transform.
        src = _gt(
            np.ones((10, 10), dtype=np.float32),
            transform=rasterio.Affine(5.0, 0.0, 500_000.0, 0.0, -5.0, 4_000_000.0),
        )
        like = _gt(np.zeros((5, 5), dtype=np.float32))
        out = RasterToRasterLike(resampling="bilinear")(src, like)
        assert out.shape == like.shape
        assert out.crs == like.crs
        assert out.transform == like.transform

    def test_plain_array_inputs_rejected(self) -> None:
        # Coregistration is geo-dependent on both sides; a plain array
        # (no transform/CRS) must raise a clear TypeError.
        arr = np.zeros((4, 4), dtype=np.float32)
        like = _gt(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(TypeError, match="GeoTensor"):
            RasterToRasterLike()(arr, like)
        with pytest.raises(TypeError, match="GeoTensor"):
            RasterToRasterLike()(like, arr)

    def test_resampling_kwarg_propagates(self) -> None:
        # Build a checkerboard-ish input that gives a different result
        # under nearest vs bilinear downsampling — confirms the
        # `resampling` kwarg reaches the underlying warp.
        rng = np.random.default_rng(0)
        src_values = rng.uniform(0, 100, size=(20, 20)).astype(np.float32)
        src = _gt(
            src_values,
            transform=rasterio.Affine(5.0, 0.0, 500_000.0, 0.0, -5.0, 4_000_000.0),
        )
        like = _gt(np.zeros((10, 10), dtype=np.float32))

        nearest = RasterToRasterLike(resampling="nearest")(src, like)
        bilinear = RasterToRasterLike(resampling="bilinear")(src, like)

        # Same target grid; different resampling kernels should yield
        # numerically different results on a non-degenerate input.
        assert nearest.shape == bilinear.shape == like.shape
        assert not np.allclose(np.asarray(nearest), np.asarray(bilinear))


# ---------------------------------------------------------------------------
# StackMatched
# ---------------------------------------------------------------------------


class TestStackMatchedBasics:
    def test_is_operator_with_config(self) -> None:
        op = StackMatched(order=["modis", "s2"])
        assert isinstance(op, Operator)
        cfg = op.get_config()
        assert cfg == {"order": ["modis", "s2"]}

    def test_namespace_export(self) -> None:
        assert gz.compositing.StackMatched is StackMatched

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            StackMatched()([])


class TestStackMatchedSequence:
    def test_two_2d_tensors_stack_to_2_bands(self) -> None:
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        out = StackMatched()([a, b])
        assert out.shape == (2, 4, 4)
        np.testing.assert_array_equal(np.asarray(out)[0], 1.0)
        np.testing.assert_array_equal(np.asarray(out)[1], 2.0)
        # Geo-metadata copies from the first input.
        assert out.crs == a.crs
        assert out.transform == a.transform

    def test_3d_tensors_concatenated_along_bands(self) -> None:
        # MODIS (4 bands) + S2 (3 bands) → 7-band output.
        a = _gt(np.zeros((4, 8, 8), dtype=np.float32))
        b = _gt(np.ones((3, 8, 8), dtype=np.float32))
        out = StackMatched()([a, b])
        assert out.shape == (7, 8, 8)
        np.testing.assert_array_equal(np.asarray(out)[:4], 0.0)
        np.testing.assert_array_equal(np.asarray(out)[4:], 1.0)

    def test_mixed_2d_and_3d_promoted_to_3d(self) -> None:
        a = _gt(np.zeros((8, 8), dtype=np.float32))  # 2-D → promoted to (1, 8, 8)
        b = _gt(np.ones((2, 8, 8), dtype=np.float32))
        out = StackMatched()([a, b])
        assert out.shape == (3, 8, 8)

    def test_higher_dim_input_rejected(self) -> None:
        # 4-D (T, C, H, W) isn't a valid GeoTensor shape for this op.
        a = _gt(np.zeros((1, 2, 8, 8), dtype=np.float32))
        with pytest.raises(ValueError, match=r"2-D .* or 3-D"):
            StackMatched()([a])

    def test_grid_mismatch_rejected(self) -> None:
        # Same shape, different transform → must fail loudly. A silent
        # pass would produce a misregistered multi-band stack.
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        b = _gt(
            np.zeros((4, 4), dtype=np.float32),
            transform=rasterio.Affine(10.0, 0.0, 600_000.0, 0.0, -10.0, 4_000_000.0),
        )
        with pytest.raises(ValueError, match="share spatial shape, transform"):
            StackMatched()([a, b])

    def test_crs_mismatch_rejected(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32), crs="EPSG:32629")
        b = _gt(np.zeros((4, 4), dtype=np.float32), crs="EPSG:32630")
        with pytest.raises(ValueError, match="share spatial shape"):
            StackMatched()([a, b])


class TestStackMatchedMapping:
    def test_mapping_input_with_order(self) -> None:
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        # `order` controls concatenation; the dict's iteration order
        # would be {modis, s2} but `order=["s2", "modis"]` flips it.
        out = StackMatched(order=["s2", "modis"])({"modis": a, "s2": b})
        np.testing.assert_array_equal(np.asarray(out)[0], 2.0)
        np.testing.assert_array_equal(np.asarray(out)[1], 1.0)

    def test_mapping_input_default_order_uses_dict_iteration(self) -> None:
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        out = StackMatched()({"modis": a, "s2": b})
        # Python dicts preserve insertion order — modis first.
        np.testing.assert_array_equal(np.asarray(out)[0], 1.0)
        np.testing.assert_array_equal(np.asarray(out)[1], 2.0)

    def test_order_with_missing_key_raises(self) -> None:
        a = _gt(np.zeros((4, 4), dtype=np.float32))
        # `order` references "landsat" but the dict only has "modis".
        with pytest.raises(KeyError, match="missing from input"):
            StackMatched(order=["modis", "landsat"])({"modis": a})

    def test_order_subset_of_keys_raises(self) -> None:
        # Strict by design: `order` must cover every input key. A
        # silent drop would let a new source added to
        # `MatchedPatch.members` disappear from the fused stack
        # without any error. Slice the dict beforehand if you really
        # want a subset.
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        c = _gt(np.full((4, 4), 3.0, dtype=np.float32))
        with pytest.raises(KeyError, match="extra in input but not in order"):
            StackMatched(order=["a", "c"])({"a": a, "b": b, "c": c})

    def test_pre_sliced_dict_works(self) -> None:
        # The documented workaround: slice the dict yourself if you
        # want only a subset of matched sources in the stack.
        a = _gt(np.full((4, 4), 1.0, dtype=np.float32))
        b = _gt(np.full((4, 4), 2.0, dtype=np.float32))
        c = _gt(np.full((4, 4), 3.0, dtype=np.float32))
        members = {"a": a, "b": b, "c": c}
        subset = {k: members[k] for k in ("a", "c")}
        out = StackMatched(order=["a", "c"])(subset)
        assert out.shape == (2, 4, 4)
        np.testing.assert_array_equal(np.asarray(out)[0], 1.0)
        np.testing.assert_array_equal(np.asarray(out)[1], 3.0)


# ---------------------------------------------------------------------------
# Skeleton coverage: confirm the rest of the coregister surface still
# raises NotImplementedError (unchanged from the scaffolding PR).
# ---------------------------------------------------------------------------


class TestRemainingScaffoldingUntouched:
    def test_swath_to_grid_still_scaffolding(self) -> None:
        from geotoolz.geom.coregister import SwathToGrid

        op = SwathToGrid(target_crs="EPSG:32629", target_res=(500.0, 500.0))
        with pytest.raises(NotImplementedError):
            op(object())  # type: ignore[arg-type]
