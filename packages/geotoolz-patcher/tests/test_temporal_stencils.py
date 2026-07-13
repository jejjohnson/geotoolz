"""End-to-end integration test: TemporalPatcher driven by a TimeStencil."""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher._src.fields.xarray import XarrayField
from geopatcher._src.time.stencils import (
    build_sampling_slices,
    valid_origin_points,
)
from geopatcher.time import (
    TemporalCausalBoxcar,
    TemporalForecast,
    TemporalPatcher,
    TemporalStencilGeometry,
    TemporalStencilSampler,
    TimeStencil,
)


def _three_hourly_series(start: str = "2020-01-01", end: str = "2020-01-05"):
    time = np.arange(start, end, dtype="datetime64[h]")[::3].astype("datetime64[ns]")
    data = np.arange(time.size, dtype=np.float32)
    return xr.DataArray(data, dims=("time",), coords={"time": time})


def _stencil() -> TimeStencil:
    # 9h of lookback + 3h horizon at the source 3-hourly cadence — 5 points
    # inclusive (-9h, -6h, -3h, 0h, +3h).
    return TimeStencil(start="-9h", stop="3h", step="3h", closed="both")


def _patcher(stencil: TimeStencil) -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalStencilGeometry(
            stencil=stencil,
            source_step=np.timedelta64(3, "h"),
        ),
        sampler=TemporalStencilSampler(stencil=stencil),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalForecast(horizon=1),
    )


def test_patch_count_matches_valid_origin_points() -> None:
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()
    stencil = _stencil()
    expected = valid_origin_points(coord, stencil)
    patcher = _patcher(stencil)
    patches = list(patcher.split(da.values, coord=coord))
    assert len(patches) == len(expected)


def test_every_patch_has_full_stencil_length() -> None:
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()
    stencil = _stencil()
    patcher = _patcher(stencil)
    expected_length = len(stencil.points)  # 5 for (-9h..3h, 3h step, closed='both')
    for patch in patcher.split(da.values, coord=coord):
        assert patch.data.shape == (expected_length,)


def test_slices_fall_on_coord_boundaries() -> None:
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()
    stencil = _stencil()
    origins = valid_origin_points(coord, stencil)
    expected_slices = build_sampling_slices(coord, origins, stencil)
    patcher = _patcher(stencil)
    actual_slices = [p.indices for p in patcher.split(da.values, coord=coord)]
    for actual, expected in zip(actual_slices, expected_slices, strict=True):
        assert actual.start == expected.start
        assert actual.stop == expected.stop


def test_n_anchors_agrees_with_split_length() -> None:
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()
    patcher = _patcher(_stencil())
    n = patcher.n_anchors(da.values, coord=coord)
    assert n == sum(1 for _ in patcher.split(da.values, coord=coord))


def test_anchors_method_returns_integer_indices() -> None:
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()
    patcher = _patcher(_stencil())
    anchors = patcher.anchors(da.values, coord=coord)
    assert all(isinstance(a, int) for a in anchors)
    # Each anchor index points to a valid origin coordinate.
    expected = valid_origin_points(coord, _stencil())
    np.testing.assert_array_equal(coord[anchors], expected)


def test_missing_coord_raises_at_entry() -> None:
    da = _three_hourly_series()
    patcher = _patcher(_stencil())
    with pytest.raises(ValueError, match="requires coord="):
        list(patcher.split(da.values))


def test_coord_length_mismatch_raises_at_entry() -> None:
    # Covers the mixed-pipeline case Codex flagged: coord-aware geometry
    # + integer sampler would otherwise accept any non-None coord and
    # silently resolve slices against the wrong timeline.
    da = _three_hourly_series()
    coord = XarrayField(da).time_coord()[:5]  # truncated relative to series
    patcher = _patcher(_stencil())
    with pytest.raises(ValueError, match="coord length must equal"):
        list(patcher.split(da.values, coord=coord))


def test_cadence_swap_does_not_change_stencil_length() -> None:
    # The same TimeStencil against a 1-hourly source produces windows of
    # the same physical extent (-9h .. +3h, step 3h → 5 points). The
    # source_step changes; the patcher rejects stride > 1 at construction,
    # so this also exercises the v0.1 stride guard against a different
    # source cadence.
    stencil = _stencil()
    with pytest.raises(ValueError, match="stride-1 stencils only"):
        TemporalStencilGeometry(stencil=stencil, source_step=np.timedelta64(1, "h"))
