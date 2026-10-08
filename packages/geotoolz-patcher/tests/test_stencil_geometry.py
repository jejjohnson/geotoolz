"""Tests for `temporal.geometry.StencilGeometry` — coordinate-aware time geometry."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import temporal
from geopatcher._src.temporal.stencils import Stencil
from geopatcher.temporal.stencils import TimeStencil


class TestNeedsCoordFlag:
    def test_flag_is_true_on_class(self) -> None:
        assert temporal.geometry.StencilGeometry.needs_coord is True


class TestStrideGuard:
    def test_stride_one_constructs_cleanly(self) -> None:
        temporal.geometry.StencilGeometry(
            stencil=TimeStencil("-9h", "3h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        )

    def test_stride_greater_than_one_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="stride-1 stencils only"):
            temporal.geometry.StencilGeometry(
                stencil=TimeStencil("-6h", "6h", "3h", closed="both"),
                source_step=np.timedelta64(1, "h"),
            )

    def test_no_source_step_defers_stride_check_to_resolve(self) -> None:
        # No source_step supplied → construction succeeds, but window_coord
        # raises if the stencil step exceeds the actual source step.
        g = temporal.geometry.StencilGeometry(
            stencil=Stencil(start=-4, stop=4, step=2, closed="both"),
        )
        # Source step is 1 here → stride would be 2 → raise.
        coord = np.arange(20)
        with pytest.raises(ValueError, match="stride-1 stencils only"):
            g.window_coord(coord, 10)


class TestWindowCoord:
    def test_resolves_to_expected_slice(self) -> None:
        g = temporal.geometry.StencilGeometry(
            stencil=TimeStencil("-3h", "3h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        )
        coord = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")
        # Anchor at index 5 (= 05:00) → window covers 02:00..08:00 inclusive.
        sl = g.window_coord(coord, 5)
        assert sl == slice(2, 9)
        np.testing.assert_array_equal(
            coord[sl],
            np.arange("2020-01-01T02", "2020-01-01T09", dtype="datetime64[h]"),
        )

    def test_returned_slice_has_step_none(self) -> None:
        # The patcher path relies on `s.stop - s.start` being the realised
        # window length; ensure we strip the explicit step=1.
        g = temporal.geometry.StencilGeometry(
            stencil=Stencil(start=-1, stop=1, step=1, closed="both"),
        )
        coord = np.arange(10)
        sl = g.window_coord(coord, 5)
        assert sl.step is None


class TestIntegerWindowGuard:
    def test_calling_window_raises_typeerror(self) -> None:
        g = temporal.geometry.StencilGeometry(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
        )
        with pytest.raises(TypeError, match="coordinate-aware"):
            g.window(24, 5)


class TestGetConfig:
    def test_round_trip_friendly_payload(self) -> None:
        g = temporal.geometry.StencilGeometry(
            stencil=TimeStencil("-9h", "3h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        )
        cfg = g.get_config()
        assert cfg["stencil"]["class"] == "temporal.stencils.TimeStencil"
        assert cfg["stencil"]["config"]["start"] == {"value": -9, "unit": "h"}
        assert cfg["source_step"] == {"value": 1, "unit": "h"}

    def test_round_trip_with_numeric_stencil(self) -> None:
        g = temporal.geometry.StencilGeometry(
            stencil=Stencil(start=-2, stop=2, step=1, closed="both"),
        )
        cfg = g.get_config()
        assert cfg["stencil"] == {
            "class": "temporal.stencils.Stencil",
            "config": {"start": -2, "stop": 2, "step": 1, "closed": "both"},
        }
        assert cfg["source_step"] is None


# ---------------------------------------------------------------------------
# #190 — one coordinate validation per split, arithmetic windows, boundary
# ---------------------------------------------------------------------------


def test_no_quadratic_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression (#190): window_coord re-validated the whole coordinate per
    # anchor (O(N^2)): ~114 s for a 20-year hourly (175 320-step) axis. Now
    # the coordinate is validated a constant number of times per split and
    # each window is O(1) arithmetic.
    from time import perf_counter

    import geopatcher._src.temporal.geometry as geometry_mod
    import geopatcher._src.temporal.patcher as patcher_mod
    from geopatcher import TemporalPatcher

    calls: list[int] = []
    real = geometry_mod.coord_step

    def counting(points: np.ndarray) -> object:
        calls.append(len(points))
        return real(points)

    monkeypatch.setattr(geometry_mod, "coord_step", counting)
    monkeypatch.setattr(patcher_mod, "coord_step", counting)

    t0 = np.datetime64("2000-01-01T00")
    coord = np.arange(t0, t0 + np.timedelta64(175_320, "h"), np.timedelta64(1, "h"))
    stencil = TimeStencil("-9h", "3h", "1h", closed="both")
    geometry = temporal.geometry.StencilGeometry(stencil=stencil)
    tp = TemporalPatcher(
        geometry,
        temporal.sampler.StencilSampler(stencil=stencil),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Fold(fold_fn=lambda n, _: n + 1, initial_state=0),
    )
    series = np.zeros(coord.shape[0], dtype=np.float32)
    start = perf_counter()
    windows = [geometry.window_coord(coord, i) for i in range(9, 175_320 - 3)]
    assert perf_counter() - start < 2.0
    assert windows[0] == slice(0, 13)
    assert windows[-1] == slice(175_320 - 13, 175_320)

    calls.clear()
    n = tp.merge(tp.split(series, coord=coord))
    assert n == coord.shape[0] - 12
    # One check in TemporalPatcher._require_coord, one memoised in the
    # geometry — never one per anchor.
    assert len(calls) <= 2


def test_window_coord_matches_build_sampling_slices() -> None:
    from geopatcher.temporal.stencils import build_sampling_slices

    coord = np.arange(0.0, 30.0, 0.5)
    for stencil in (
        Stencil(-2.0, 1.0, 0.5, closed="both"),
        Stencil(-2.0, 1.0, 0.5, closed="left"),
        Stencil(-2.0, 1.0, 0.5, closed="right"),
        Stencil(-2.0, 1.0, 0.5, closed="neither"),
        Stencil(0.0, 0.0, 0.0, closed="both"),
    ):
        g = temporal.geometry.StencilGeometry(stencil=stencil)
        for i in (4, 10, 57):
            (ref,) = build_sampling_slices(coord, coord[[i]], stencil)
            assert g.window_coord(coord, i) == slice(ref.start, ref.stop)


def test_window_coord_rejects_bad_coords() -> None:
    g = temporal.geometry.StencilGeometry(stencil=Stencil(-1, 1, 1, closed="both"))
    with pytest.raises(ValueError, match="strictly increasing"):
        g.window_coord(np.arange(10)[::-1], 5)
    with pytest.raises(ValueError, match="evenly spaced"):
        g.window_coord(np.array([0, 1, 2, 4, 5, 6]), 3)
    with pytest.raises(ValueError, match="1-D"):
        g.window_coord(np.zeros((3, 3)), 1)


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [("drop", None), ("shrink", slice(0, 3))],
)
def test_boundary_at_axis_start(boundary: str, expected: slice | None) -> None:
    g = temporal.geometry.StencilGeometry(
        stencil=Stencil(-2, 2, 1, closed="both"),
        boundary=boundary,  # type: ignore[arg-type]
    )
    assert g.window_coord(np.arange(10), 0) == expected


def test_boundary_raise() -> None:
    g = temporal.geometry.StencilGeometry(
        stencil=Stencil(-2, 2, 1, closed="both"), boundary="raise"
    )
    assert g.window_coord(np.arange(10), 5) == slice(3, 8)
    with pytest.raises(ValueError, match="overflows the time axis"):
        g.window_coord(np.arange(10), 9)


def test_source_step_string_is_parsed() -> None:
    # Regression (#190): "1h" was accepted, then failed on first use.
    g = temporal.geometry.StencilGeometry(
        stencil=TimeStencil("-3h", "3h", "1h", closed="both"), source_step="1h"
    )
    assert g.source_step == np.timedelta64(1, "h")
    with pytest.raises(TypeError, match="timedelta64"):
        temporal.geometry.StencilGeometry(
            stencil=TimeStencil("-3h", "3h", "1h", closed="both"), source_step=1
        )


def test_pickles_after_use() -> None:
    import pickle

    g = temporal.geometry.StencilGeometry(stencil=Stencil(-1, 1, 1, closed="both"))
    coord = np.arange(10)
    assert g.window_coord(coord, 4) == slice(3, 6)
    clone = pickle.loads(pickle.dumps(g))
    assert clone.window_coord(coord, 4) == slice(3, 6)
