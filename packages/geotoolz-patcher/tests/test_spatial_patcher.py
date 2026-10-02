"""Tests for `SpatialPatcher` — split/merge end-to-end."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from _helpers import make_rasterio_reader_field

from geopatcher import (
    Patch,
    PatchErrorRecord,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


class FlakyRasterField:
    """RasterField wrapper that fails selected anchors before succeeding.

    `failures_by_anchor[(row, col)] = n` means the first `n` reads for that
    anchor raise `exception_type`, then later reads delegate to the wrapped
    field. This keeps retry/skip/mask tests deterministic.
    """

    def __init__(
        self,
        wrapped: RasterField,
        failures_by_anchor: dict[tuple[int, int], int],
        exception_type: type[Exception] = OSError,
    ) -> None:
        self.wrapped = wrapped
        self.failures_by_anchor = dict(failures_by_anchor)
        self.exception_type = exception_type
        self.attempts: dict[tuple[int, int], int] = {}

    @property
    def domain(self) -> Any:
        return self.wrapped.domain

    def select(self, indices):
        anchor = (int(indices.row_off), int(indices.col_off))
        self.attempts[anchor] = self.attempts.get(anchor, 0) + 1
        if self.attempts[anchor] <= self.failures_by_anchor.get(anchor, 0):
            raise self.exception_type(f"flaky read at {anchor}")
        return self.wrapped.select(indices)


# The shared `field` fixture (tests/conftest.py) is 2-D so the (row, col)
# slicer from _resolve_indices matches the domain shape exactly. The 3-D
# channels-first case is exercised in test_ops.py.


class TestSplit:
    def test_returns_iterator(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        result = patcher.split(field)
        assert isinstance(result, Iterator)
        patches = list(result)
        assert len(patches) == 16  # 4x4 tiles
        assert all(isinstance(p, Patch) for p in patches)

    def test_data_matches_indices(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        for patch in patcher.split(field):
            assert patch.data.shape[-2:] == (16, 16)

    def test_n_anchors_matches_split_length(self, field: RasterField) -> None:
        # ADR-001: `split` is an iterator (no len()); `n_anchors` is the
        # cheap substitute that walks the sampler without touching the field.
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        n = patcher.n_anchors(field)
        assert n == 16  # 4x4 lattice
        assert n == sum(1 for _ in patcher.split(field))

    def test_on_error_skip_omits_failed_patch(self, field: RasterField) -> None:
        flaky = FlakyRasterField(field, failures_by_anchor={(0, 16): 1})
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="skip",
        )

        patches = list(patcher.split(flaky))

        assert len(patches) == 15
        assert (0, 16) not in {p.anchor for p in patches}
        assert len(patcher.errors) == 1
        assert isinstance(patcher.errors[0], PatchErrorRecord)
        assert patcher.errors[0].anchor == (0, 16)
        assert patcher.errors[0].kind == "OSError"

    def test_on_error_retry_succeeds_after_transient_failures(
        self, field: RasterField
    ) -> None:
        flaky = FlakyRasterField(field, failures_by_anchor={(0, 16): 2})
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="retry",
            max_retries=2,
            # Cover class-name config; the exhausted-retry test covers classes.
            retry_on=("OSError",),
        )

        patches = list(patcher.split(flaky))

        assert len(patches) == 16
        assert (0, 16) in {p.anchor for p in patches}
        assert flaky.attempts[(0, 16)] == 3
        assert [err.retry_count for err in patcher.errors] == [0, 1]

    def test_on_error_retry_skips_after_retries_exhausted(
        self, field: RasterField
    ) -> None:
        flaky = FlakyRasterField(field, failures_by_anchor={(0, 16): 3})
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="retry",
            max_retries=1,
            # Cover class objects; the transient-success test covers names.
            retry_on=(OSError,),
        )

        patches = list(patcher.split(flaky))

        assert len(patches) == 15
        assert flaky.attempts[(0, 16)] == 2
        assert [err.retry_count for err in patcher.errors] == [0, 1]

    def test_on_error_retry_reraises_non_matching_exception(
        self, field: RasterField
    ) -> None:
        flaky = FlakyRasterField(
            field,
            failures_by_anchor={(0, 16): 1},
            exception_type=ValueError,
        )
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="retry",
            max_retries=2,
            retry_on=(OSError,),
        )

        with pytest.raises(ValueError, match="flaky read"):
            list(patcher.split(flaky))

        assert flaky.attempts[(0, 16)] == 1
        assert patcher.errors[0].kind == "ValueError"

    def test_on_error_mask_emits_nan_patch(self, field: RasterField) -> None:
        flaky = FlakyRasterField(field, failures_by_anchor={(0, 16): 1})
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="mask",
        )

        patches = list(patcher.split(flaky))

        assert len(patches) == 16
        masked = next(p for p in patches if p.anchor == (0, 16))
        assert masked.data.shape == (16, 16)
        assert np.isnan(masked.data).all()
        recon = patcher.merge(patches, field.domain)
        # The masked tile is a hole (the NaN fill); every other cell is data.
        hole = np.zeros(recon.shape, dtype=bool)
        hole[0:16, 16:32] = True
        assert np.isnan(recon[hole]).all()
        assert not np.isnan(recon[~hole]).any()
        assert patcher.errors[0].kind == "OSError"

    def test_invalid_on_error_policy_raises(self, field: RasterField) -> None:
        with pytest.raises(ValueError, match="invalid on_error policy"):
            SpatialPatcher(
                geometry=SpatialRectangular(size=(16, 16)),
                sampler=SpatialRegularStride(step=16),
                window=SpatialBoxcar(),
                aggregation=SpatialOverlapAdd(),
                on_error="ignore",  # type: ignore[arg-type]
            )

    def test_capture_traceback_false_skips_formatted_traceback(
        self, field: RasterField
    ) -> None:
        """`capture_traceback=False` keeps `errors` lean for bulk skip workloads."""
        flaky = FlakyRasterField(field, failures_by_anchor={(0, 16): 1})
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
            on_error="skip",
            capture_traceback=False,
        )

        list(patcher.split(flaky))

        assert len(patcher.errors) == 1
        assert patcher.errors[0].traceback == ""
        # Still captures kind / message so callers can inspect failure modes.
        assert patcher.errors[0].kind == "OSError"
        assert "flaky read" in patcher.errors[0].message


class TestSplitMergeRoundtrip:
    def test_identity_with_boxcar_no_overlap(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = list(patcher.split(field))
        recon = patcher.aggregation.merge(patches, field.reader)
        np.testing.assert_allclose(recon, np.asarray(field.reader))


class TestGetConfig:
    def test_records_each_axis(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        cfg = patcher.get_config()
        assert cfg["geometry"]["class"] == "SpatialRectangular"
        assert cfg["sampler"]["class"] == "SpatialRegularStride"
        assert cfg["window"]["class"] == "SpatialBoxcar"
        assert cfg["aggregation"]["class"] == "SpatialOverlapAdd"


def test_rasterio_reader_field_split_merge_pad(tmp_path: Any) -> None:
    """File-backed `RasterioReader` field: split → merge → pad (issue #177).

    Every chip must be a materialised `GeoTensor` matching rasterio's own
    read (pixels, nodata fill past the edge, window transform), and an
    overlap-add merge must reproduce the file exactly.
    """
    import rasterio
    from georeader.geotensor import GeoTensor
    from rasterio.windows import Window, transform as window_transform

    path = tmp_path / "scene.tif"
    field = make_rasterio_reader_field(path, size=(70, 70), bands=2, nodata=-1.0)
    with rasterio.open(path) as src:
        src_transform = src.transform
        reference = src.read()

        # split → merge: 14 px chips tile the 70x70 scene exactly.
        tiler = SpatialPatcher(
            geometry=SpatialRectangular(size=(14, 14)),
            sampler=SpatialRegularStride(step=14),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        tiles = list(tiler.split(field))
        assert len(tiles) == 25
        assert all(isinstance(t.data, GeoTensor) for t in tiles)
        merged = tiler.merge(tiles, field.domain)
        np.testing.assert_array_equal(merged, reference)

        # pad: 16 px chips, the 64 anchors overflow the edge by 10 px and
        # must be filled with the file's nodata, exactly as rasterio does.
        padder = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16), boundary="pad"),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        chips = list(padder.split(field))
        assert len(chips) == 25
        for chip in chips:
            assert isinstance(chip.data, GeoTensor)
            row, col = chip.anchor
            window = Window(col_off=col, row_off=row, width=16, height=16)
            expected = src.read(window=window, boundless=True, fill_value=-1.0)
            np.testing.assert_array_equal(np.asarray(chip.data), expected)
            assert chip.data.transform == window_transform(window, src_transform)

    out = field.with_data(merged)
    assert out.fill_value_default == -1.0
    assert out.transform == src_transform


@pytest.mark.parametrize("adapter", ["raster", "reproject"])
def test_with_data_preserves_nodata_and_attrs(adapter: str) -> None:
    """`with_data` keeps the source's ``fill_value_default`` and ``attrs``."""
    import rasterio
    from georeader.geotensor import GeoTensor

    from geopatcher import ReprojectingRasterField

    source = GeoTensor(
        values=np.ones((8, 8), dtype=np.float32),
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0),
        crs="EPSG:32630",
        fill_value_default=-1.0,
        attrs={"k": 1},
    )
    field: Any = (
        RasterField(source)
        if adapter == "raster"
        else ReprojectingRasterField(source, dst_crs="EPSG:3857")
    )
    out = field.with_data(np.zeros((8, 8), dtype=np.float32))
    assert out.fill_value_default == -1.0
    assert out.attrs == {"k": 1}
    # A copy — mutating the output must not leak back into the source.
    out.attrs["k"] = 2
    assert source.attrs == {"k": 1}


def test_window_type_error_is_not_swallowed() -> None:
    # #187: `_safe_base_weights` caught every TypeError, so a bug inside
    # a `SpatialCustom` fn silently turned into "no weights".
    from _helpers import make_raster_field

    from geopatcher import SpatialCustom

    def broken(geometry: Any) -> np.ndarray:
        raise TypeError("bug in user code")

    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialCustom(fn=broken),
        aggregation=SpatialOverlapAdd(),
    )
    with pytest.raises(TypeError, match="bug in user code"):
        list(patcher.split(make_raster_field(16)))


def test_ragged_geometry_gets_no_base_weights() -> None:
    from geopatcher import SpatialKNNGraph
    from geopatcher._src.spatial.patcher import _safe_base_weights

    assert _safe_base_weights(SpatialBoxcar(), SpatialKNNGraph(k=2)) is None
