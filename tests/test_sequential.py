"""Tests for Sequential."""

from __future__ import annotations

import pytest

from geotoolz.core import Identity, Operator, Sequential


class Add(Operator):
    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}


class Writer(Operator):
    """A terminal operator (only valid at the end of a Sequential)."""

    _terminal = True

    def _apply(self, _x: int) -> None:
        return None


class TestBasicChain:
    def test_eager_chain(self) -> None:
        pipe = Sequential([Add(1), Add(10), Add(100)])
        assert pipe(0) == 111

    def test_empty_sequential_is_identity(self) -> None:
        assert Sequential([])(42) == 42

    def test_empty_sequential_requires_input(self) -> None:
        with pytest.raises(TypeError, match="requires an input"):
            Sequential([])()

    def test_single_op(self) -> None:
        assert Sequential([Add(5)])(0) == 5


class TestValidation:
    def test_rejects_non_operator(self) -> None:
        with pytest.raises(TypeError, match="expected Operator"):
            Sequential([Add(1), "not an operator"])  # type: ignore[list-item]

    def test_rejects_terminal_in_non_last_position(self) -> None:
        with pytest.raises(TypeError, match="terminal operator"):
            Sequential([Writer(), Add(1)])

    def test_allows_terminal_at_end(self) -> None:
        pipe = Sequential([Add(1), Writer()])
        # Should not raise.
        assert pipe is not None


class TestGetConfig:
    def test_config_recurses(self) -> None:
        pipe = Sequential([Add(1), Add(2)])
        cfg = pipe.get_config()
        assert cfg == {
            "operators": [
                {"class": "Add", "config": {"n": 1}},
                {"class": "Add", "config": {"n": 2}},
            ]
        }


class TestOrFlattening:
    def test_or_appends(self) -> None:
        pipe = Sequential([Add(1), Add(2)]) | Add(3)
        assert isinstance(pipe, Sequential)
        assert len(pipe) == 3
        assert pipe(0) == 6

    def test_or_flattens_right_sequential(self) -> None:
        pipe = Sequential([Add(1), Add(2)]) | Sequential([Add(3), Add(4)])
        assert isinstance(pipe, Sequential)
        assert len(pipe) == 4
        assert pipe(0) == 10


class TestRepr:
    def test_repr_lists_operators(self) -> None:
        pipe = Sequential([Add(1), Identity()])
        assert repr(pipe) == "Sequential([Add(n=1), Identity()])"

    def test_empty_repr(self) -> None:
        assert repr(Sequential([])) == "Sequential([])"


class TestIndexingAndLen:
    def test_len(self) -> None:
        assert len(Sequential([Add(1), Add(2), Add(3)])) == 3

    def test_index(self) -> None:
        pipe = Sequential([Add(1), Add(2)])
        first = pipe[0]
        assert isinstance(first, Add)
        assert first.n == 1

    def test_slice_returns_sequential(self) -> None:
        pipe = Sequential([Add(1), Add(2), Add(3)])
        sliced = pipe[1:]
        assert isinstance(sliced, Sequential)
        assert len(sliced) == 2
