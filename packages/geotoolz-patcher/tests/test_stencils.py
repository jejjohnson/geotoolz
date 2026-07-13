# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright 2026 J. Emmanuel Johnson, distributed under the
# MIT license that covers the rest of geopatcher. Modifications:
#   - port from absl.testing to pytest;
#   - drop `assertRaisesWithLiteralMatch` checks of the upstream
#     error-message string (we changed the message to name operands);
#   - add geopatcher-specific tests covering get_config round-trip,
#     labelled-error messages, and the valid_origin_points contract.
"""Tests for `geopatcher._src.time.stencils`."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher._src.time.stencils import (
    Stencil,
    TimeStencil,
    _to_timedelta64,
    build_sampling_slices,
    divide_evenly,
    valid_origin_points,
)


class TestStencilPoints:
    @pytest.mark.parametrize(
        ("start", "stop", "step", "closed", "expected"),
        [
            (0, 1, 0.5, "both", [0.0, 0.5, 1.0]),
            (0, 1, 0.5, "left", [0.0, 0.5]),
            (0, 1, 0.5, "right", [0.5, 1.0]),
            (0, 1, 0.5, "neither", [0.5]),
            (-1, 1, 0.5, "both", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ],
    )
    def test_points_match_closedness(self, start, stop, step, closed, expected) -> None:
        stencil = Stencil(start, stop, step, closed=closed)
        np.testing.assert_allclose(stencil.points, expected)

    def test_default_closed_is_left(self) -> None:
        assert Stencil(0, 1, 0.5).closed == "left"

    def test_single_value_stencil(self) -> None:
        # Degenerate stencil with one point at the origin.
        s = Stencil(0, 0, 0, closed="both")
        np.testing.assert_array_equal(s.points, [0])


class TestStencilValidation:
    def test_start_greater_than_stop_raises(self) -> None:
        with pytest.raises(ValueError, match="start must not be greater than stop"):
            Stencil(0, -1, 1)

    def test_single_value_invalid_closed(self) -> None:
        with pytest.raises(ValueError, match='must be "both"'):
            Stencil(0, 0, 0, closed="left")

    def test_single_value_nonzero_step(self) -> None:
        with pytest.raises(ValueError, match="must equal zero"):
            Stencil(0, 0, 1, closed="both")

    def test_invalid_closed_value(self) -> None:
        with pytest.raises(ValueError, match="invalid value for closed"):
            Stencil(0, 1, 0.5, closed="invalid")  # type: ignore[arg-type]

    def test_non_divisible_step_raises_on_points(self) -> None:
        s = Stencil(0, 1, 0.3, closed="both")
        with pytest.raises(ValueError, match="must evenly divide"):
            _ = s.points


class TestTimeStencil:
    def test_constructs_from_strings(self) -> None:
        s = TimeStencil("-3h", "+2h", "1h", closed="both")
        assert s.start == np.timedelta64(-3, "h")
        assert s.stop == np.timedelta64(2, "h")
        assert s.step == np.timedelta64(1, "h")
        expected = np.array([-3, -2, -1, 0, 1, 2], dtype="timedelta64[h]")
        np.testing.assert_array_equal(s.points, expected)

    def test_repr_includes_units(self) -> None:
        s = TimeStencil("-9h", "3h", "1h", closed="both")
        assert repr(s) == (
            "TimeStencil(start='-9 hours', stop='3 hours', step='1 hours',"
            " closed='both')"
        )

    def test_hashable_and_equal_by_value(self) -> None:
        a = Stencil(start=0, stop=10, step=1)
        b = Stencil(start=0, stop=10, step=1)
        c = Stencil(start=0, stop=10, step=2)
        assert hash(a) == hash(b)
        assert hash(a) != hash(c)
        d = {a: "value"}
        assert d[b] == "value"

        ts1 = TimeStencil(start="-1h", stop="1h", step="1h")
        ts2 = TimeStencil(start="-1h", stop="1h", step="1h")
        assert hash(ts1) == hash(ts2)


class TestToTimedelta64:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1h", np.timedelta64(1, "h")),
            ("-1h", np.timedelta64(-1, "h")),
            ("+1h", np.timedelta64(1, "h")),
            ("1hr", np.timedelta64(1, "h")),
            ("1hour", np.timedelta64(1, "h")),
            ("1hours", np.timedelta64(1, "h")),
            ("1m", np.timedelta64(1, "m")),
            ("1min", np.timedelta64(1, "m")),
            ("1minute", np.timedelta64(1, "m")),
            ("1minutes", np.timedelta64(1, "m")),
            ("1s", np.timedelta64(1, "s")),
            ("1sec", np.timedelta64(1, "s")),
            ("1second", np.timedelta64(1, "s")),
            ("1seconds", np.timedelta64(1, "s")),
            ("1D", np.timedelta64(1, "D")),
            ("1day", np.timedelta64(1, "D")),
            ("1days", np.timedelta64(1, "D")),
            ("24 hours", np.timedelta64(24, "h")),
        ],
    )
    def test_parses_known_units(self, value, expected) -> None:
        assert _to_timedelta64(value) == expected

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid time delta string"):
            _to_timedelta64("invalid")

    def test_unsupported_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported time unit"):
            _to_timedelta64("1w")


class TestBuildSamplingSlices:
    @pytest.mark.parametrize(
        ("source_points", "sample_origins", "stencil", "expected"),
        [
            (
                [0, 1, 2, 3, 4],
                [1, 3],
                Stencil(start=-1, stop=1, step=1, closed="both"),
                [slice(0, 3, 1), slice(2, 5, 1)],
            ),
            (
                [0, 1, 2, 3, 4],
                [1, 3],
                Stencil(start=-1, stop=1, step=1, closed="left"),
                [slice(0, 2, 1), slice(2, 4, 1)],
            ),
            (
                [0, 1, 2, 3, 4],
                [1, 3],
                Stencil(start=-1, stop=1, step=1, closed="right"),
                [slice(1, 3, 1), slice(3, 5, 1)],
            ),
            (
                [0, 1, 2, 3, 4],
                [1, 3],
                Stencil(start=-1, stop=1, step=1, closed="neither"),
                [slice(1, 2, 1), slice(3, 4, 1)],
            ),
            (
                [0, 2, 4, 6, 8],
                [2, 6],
                Stencil(start=-2, stop=2, step=2, closed="both"),
                [slice(0, 3, 1), slice(2, 5, 1)],
            ),
            (
                np.arange(10).tolist(),
                [2],
                Stencil(start=-2, stop=4, step=2, closed="left"),
                [slice(0, 6, 2)],
            ),
        ],
    )
    def test_matches_expected_slices(
        self, source_points, sample_origins, stencil, expected
    ) -> None:
        actual = build_sampling_slices(source_points, sample_origins, stencil)
        assert actual == expected

    def test_slice_round_trip_matches_stencil_points(self) -> None:
        # The first slice over source_points must recover stencil.points
        # shifted by the origin. Direct port of the upstream round-trip.
        source_points = np.arange(10)
        sample_origins = [3]
        stencil = Stencil(start=-1, stop=1, step=1, closed="both")
        (s,) = build_sampling_slices(source_points, sample_origins, stencil)
        np.testing.assert_array_equal(
            source_points[s], stencil.points + sample_origins[0]
        )

    def test_datetime_inputs(self) -> None:
        source_points = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")
        sample_origins = np.array(
            [
                np.datetime64("2020-01-01T02"),
                np.datetime64("2020-01-01T06"),
            ]
        )
        stencil = Stencil(
            start=np.timedelta64(-1, "h"),
            stop=np.timedelta64(2, "h"),
            step=np.timedelta64(1, "h"),
            closed="both",
        )
        actual = build_sampling_slices(source_points, sample_origins, stencil)
        assert actual == [slice(1, 5, 1), slice(5, 9, 1)]

    def test_invalid_source_points_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="source_points must be 1D"):
            build_sampling_slices(
                source_points=[[0, 1], [2, 3]],
                sample_origins=[1],
                stencil=Stencil(0, 1, 1),
            )

    def test_invalid_sample_origins_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_origins must be 1D"):
            build_sampling_slices(
                source_points=[0, 1],
                sample_origins=[[1], [2]],
                stencil=Stencil(0, 1, 1),
            )

    def test_unsorted_source_points_raises(self) -> None:
        with pytest.raises(ValueError, match="source_points must be sorted"):
            build_sampling_slices(
                source_points=[1, 0],
                sample_origins=[1],
                stencil=Stencil(0, 1, 1),
            )

    def test_unsorted_sample_origins_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_origins must be sorted"):
            build_sampling_slices(
                source_points=[0, 1],
                sample_origins=[1, 0],
                stencil=Stencil(0, 1, 1),
            )

    def test_non_constant_source_step_raises(self) -> None:
        with pytest.raises(ValueError, match="source_points must have constant step"):
            build_sampling_slices(
                source_points=[0, 1, 3],
                sample_origins=[1],
                stencil=Stencil(0, 1, 1),
            )

    def test_origin_before_first_source_point_raises(self) -> None:
        with pytest.raises(ValueError, match="at or after the first source point"):
            build_sampling_slices(
                source_points=[0, 1, 2],
                sample_origins=[-1],
                stencil=Stencil(0, 1, 1),
            )

    def test_origin_past_last_source_point_raises(self) -> None:
        with pytest.raises(ValueError, match="at or before the last source point"):
            build_sampling_slices(
                source_points=[0, 1, 2],
                sample_origins=[2],
                stencil=Stencil(0, 2, 1),
            )


class TestValidOriginPoints:
    @pytest.mark.parametrize(
        ("source_points", "stencil", "expected"),
        [
            (
                [0, 1, 2, 3, 4],
                Stencil(start=-1, stop=1, step=1, closed="both"),
                [1, 2, 3],
            ),
            (
                [0, 1, 2, 3, 4],
                Stencil(start=-2, stop=2, step=1, closed="both"),
                [2],
            ),
            (
                [0, 2, 4, 6, 8],
                Stencil(start=-2, stop=2, step=2, closed="both"),
                [2, 4, 6],
            ),
            (
                [0, 1, 2, 3, 4],
                Stencil(start=-1, stop=1, step=1, closed="left"),
                [1, 2, 3, 4],
            ),
            (
                [0, 1, 2, 3, 4],
                Stencil(start=-1, stop=1, step=1, closed="right"),
                [0, 1, 2, 3],
            ),
            (
                [0, 1, 2, 3, 4],
                Stencil(start=-1, stop=1, step=1, closed="neither"),
                [0, 1, 2, 3, 4],
            ),
        ],
    )
    def test_matches_expected(self, source_points, stencil, expected) -> None:
        actual = valid_origin_points(source_points, stencil)
        np.testing.assert_array_equal(actual, expected)

    def test_datetime_inputs(self) -> None:
        source = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")
        stencil = Stencil(
            start=np.timedelta64(-2, "h"),
            stop=np.timedelta64(3, "h"),
            step=np.timedelta64(1, "h"),
            closed="both",
        )
        expected = np.arange("2020-01-01T02", "2020-01-01T21", dtype="datetime64[h]")
        np.testing.assert_array_equal(valid_origin_points(source, stencil), expected)

    def test_no_truncation(self) -> None:
        # For every valid origin, the resolved slice must fit in-record.
        source = np.arange(0, 100)
        stencil = Stencil(start=-3, stop=4, step=1, closed="both")
        origins = valid_origin_points(source, stencil)
        slices = build_sampling_slices(source, origins, stencil)
        for s in slices:
            assert 0 <= s.start < s.stop <= len(source)
            assert (s.stop - s.start) == len(stencil.points)


class TestDivideEvenly:
    def test_exact_division_returns_quotient(self) -> None:
        assert divide_evenly(12, 3) == 4

    def test_uneven_division_raises_with_label(self) -> None:
        # The labelled error names what `y` represents — geopatcher-specific.
        with pytest.raises(ValueError, match="step"):
            divide_evenly(10, 3, label="step")

    def test_timedelta_inputs(self) -> None:
        # 9h / 3h = 3 exactly; auto-unit-promotion is fine.
        result = divide_evenly(
            np.timedelta64(9, "h"), np.timedelta64(3, "h"), label="source_step"
        )
        assert int(result) == 3

    def test_float_inputs_within_epsilon(self) -> None:
        # 1.0 / 0.5 = 2 within float fuzz.
        assert int(divide_evenly(1.0, 0.5)) == 2


class TestGetConfig:
    def test_numeric_round_trip(self) -> None:
        s = Stencil(start=-2, stop=2, step=0.5, closed="both")
        assert s.get_config() == {
            "start": -2,
            "stop": 2,
            "step": 0.5,
            "closed": "both",
        }

    def test_time_stencil_yaml_friendly(self) -> None:
        ts = TimeStencil("-9h", "3h", "1h", closed="both")
        cfg = ts.get_config()
        # Values must be JSON/YAML-serialisable strings/scalars.
        for key, value in cfg.items():
            assert isinstance(value, (str, int, float)), (
                f"{key!r} -> {value!r} is not YAML-friendly"
            )
