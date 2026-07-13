"""Tests for `TemporalStencilGeometry` — coordinate-aware time geometry."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher._src.time.stencils import Stencil
from geopatcher.time import (
    TemporalStencilGeometry,
    TimeStencil,
)


class TestNeedsCoordFlag:
    def test_flag_is_true_on_class(self) -> None:
        assert TemporalStencilGeometry.needs_coord is True


class TestStrideGuard:
    def test_stride_one_constructs_cleanly(self) -> None:
        TemporalStencilGeometry(
            stencil=TimeStencil("-9h", "3h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        )

    def test_stride_greater_than_one_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="stride-1 stencils only"):
            TemporalStencilGeometry(
                stencil=TimeStencil("-6h", "6h", "3h", closed="both"),
                source_step=np.timedelta64(1, "h"),
            )

    def test_no_source_step_defers_stride_check_to_resolve(self) -> None:
        # No source_step supplied → construction succeeds, but window_coord
        # raises if the stencil step exceeds the actual source step.
        g = TemporalStencilGeometry(
            stencil=Stencil(start=-4, stop=4, step=2, closed="both"),
        )
        # Source step is 1 here → stride would be 2 → raise.
        coord = np.arange(20)
        with pytest.raises(ValueError, match="stride-1 stencils only"):
            g.window_coord(coord, 10)


class TestWindowCoord:
    def test_resolves_to_expected_slice(self) -> None:
        g = TemporalStencilGeometry(
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
        g = TemporalStencilGeometry(
            stencil=Stencil(start=-1, stop=1, step=1, closed="both"),
        )
        coord = np.arange(10)
        sl = g.window_coord(coord, 5)
        assert sl.step is None


class TestIntegerWindowGuard:
    def test_calling_window_raises_typeerror(self) -> None:
        g = TemporalStencilGeometry(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
        )
        with pytest.raises(TypeError, match="coordinate-aware"):
            g.window(24, 5)


class TestGetConfig:
    def test_round_trip_friendly_payload(self) -> None:
        g = TemporalStencilGeometry(
            stencil=TimeStencil("-9h", "3h", "1h", closed="both"),
            source_step=np.timedelta64(1, "h"),
        )
        cfg = g.get_config()
        assert cfg["stencil"]["start"] == "-9 hours"
        assert cfg["source_step"] == "1 hours"

    def test_round_trip_with_numeric_stencil(self) -> None:
        g = TemporalStencilGeometry(
            stencil=Stencil(start=-2, stop=2, step=1, closed="both"),
        )
        cfg = g.get_config()
        assert cfg["stencil"] == {
            "start": -2,
            "stop": 2,
            "step": 1,
            "closed": "both",
        }
        assert cfg["source_step"] is None
