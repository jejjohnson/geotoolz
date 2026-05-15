"""Tests for Tap, Snapshot, ShapeTrace."""

from __future__ import annotations

import numpy as np
import pytest

from geotoolz.core import Sequential, ShapeTrace, Snapshot, Tap


class TestTap:
    def test_passes_value_through(self) -> None:
        seen: list[int] = []
        out = Tap(seen.append)(42)
        assert out == 42
        assert seen == [42]

    def test_fn_return_value_is_ignored(self) -> None:
        # The callback returns something — Tap must not propagate it.
        out = Tap(lambda x: 999)(1)
        assert out == 1

    def test_forbid_in_yaml(self) -> None:
        assert Tap.forbid_in_yaml is True

    def test_in_sequential(self) -> None:
        # Tap should not change the chain's output.
        seen: list[int] = []
        pipe = Sequential([Tap(seen.append), Tap(seen.append)])
        out = pipe(7)
        assert out == 7
        assert seen == [7, 7]


class TestSnapshot:
    def test_stores_by_key(self) -> None:
        snap = Snapshot()
        pipe = Sequential([snap.at("first"), snap.at("second")])
        out = pipe("hello")
        assert out == "hello"
        assert snap["first"] == "hello"
        assert snap["second"] == "hello"
        assert "first" in snap
        assert "missing" not in snap

    def test_keys_and_items(self) -> None:
        snap = Snapshot()
        pipe = Sequential([snap.at("a"), snap.at("b")])
        pipe(1)
        assert set(snap.keys()) == {"a", "b"}
        assert dict(snap.items()) == {"a": 1, "b": 1}

    def test_clear(self) -> None:
        snap = Snapshot()
        Sequential([snap.at("k")])(0)
        assert "k" in snap
        snap.clear()
        assert "k" not in snap

    def test_at_returns_passthrough_operator(self) -> None:
        snap = Snapshot()
        op = snap.at("k")
        assert op(123) == 123


class TestShapeTrace:
    def test_logs_each_call(self) -> None:
        lines: list[str] = []
        op = ShapeTrace(printer=lines.append)
        arr = np.zeros((3, 4, 5), dtype=np.int16)
        out = op(arr)
        assert out is arr
        assert len(lines) == 1
        assert "(3, 4, 5)" in lines[0]
        assert "int16" in lines[0]

    def test_diff_only_suppresses_repeats(self) -> None:
        lines: list[str] = []
        op = ShapeTrace(printer=lines.append, mode="diff_only")
        arr = np.zeros((3, 4, 5), dtype=np.int16)
        op(arr)
        op(arr)
        op(arr)
        # Three calls, identical descriptions — only one printed.
        assert len(lines) == 1

    def test_diff_only_prints_on_change(self) -> None:
        lines: list[str] = []
        op = ShapeTrace(printer=lines.append, mode="diff_only")
        op(np.zeros((1, 2, 3), dtype=np.int16))
        op(np.zeros((1, 2, 3), dtype=np.int16))  # same — silent
        op(np.zeros((1, 2, 3), dtype=np.float32))  # dtype change
        assert len(lines) == 2

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            ShapeTrace(mode="bogus")

    def test_not_forbid_in_yaml(self) -> None:
        assert ShapeTrace.forbid_in_yaml is False
