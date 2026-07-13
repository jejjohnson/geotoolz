"""Tests for `SpatialPatcher` — split/merge end-to-end."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

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
        assert not np.isnan(recon).any()
        np.testing.assert_allclose(recon[0:16, 16:32], 0.0)
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
