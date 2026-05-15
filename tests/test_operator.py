"""Tests for the Operator base class."""

from __future__ import annotations

import pytest

from geotoolz.core import Identity, Input, Node, Operator, Sequential


class Double(Operator):
    """Trivial test operator that doubles its scalar input."""

    def __init__(self, factor: int = 2) -> None:
        self.factor = factor

    def _apply(self, x: int) -> int:
        return x * self.factor

    def get_config(self) -> dict:
        return {"factor": self.factor}


class TestEagerDispatch:
    def test_call_runs_apply_for_scalar(self) -> None:
        assert Double(3)(4) == 12

    def test_unimplemented_apply_raises(self) -> None:
        class Bare(Operator):
            pass

        with pytest.raises(NotImplementedError):
            Bare()(1)


class TestGraphDispatch:
    def test_call_with_input_returns_node(self) -> None:
        op = Double()
        inp = Input("x")
        out = op(inp)
        assert isinstance(out, Node)
        assert out.operator is op
        assert out.parents == (inp,)

    def test_call_with_node_returns_node(self) -> None:
        op = Double()
        inp = Input("x")
        first = op(inp)
        second = op(first)
        assert isinstance(second, Node)
        assert second.parents == (first,)


class TestRepr:
    def test_repr_uses_get_config(self) -> None:
        assert repr(Double(7)) == "Double(factor=7)"

    def test_repr_empty_config(self) -> None:
        assert repr(Identity()) == "Identity()"


class TestPipeOperator:
    def test_or_creates_sequential(self) -> None:
        result = Double(2) | Double(3)
        assert isinstance(result, Sequential)
        assert len(result.operators) == 2
        assert result(4) == 4 * 2 * 3

    def test_or_flattens_right_sequential(self) -> None:
        right = Sequential([Double(2), Double(3)])
        merged = Double(5) | right
        # Should be three ops, not nested
        assert isinstance(merged, Sequential)
        assert len(merged.operators) == 3
        assert merged(1) == 1 * 5 * 2 * 3

    def test_or_chain_associativity(self) -> None:
        left = Double(2) | Double(3) | Double(5)
        assert isinstance(left, Sequential)
        assert len(left.operators) == 3
        assert left(1) == 30


class TestFlags:
    def test_default_not_terminal(self) -> None:
        assert Operator._terminal is False
        assert Double()._terminal is False

    def test_default_not_forbid_in_yaml(self) -> None:
        assert Operator.forbid_in_yaml is False
        assert Double().forbid_in_yaml is False


class TestPostApplyHookIsNoOp:
    def test_hook_does_not_raise(self) -> None:
        # The reserved hook-dispatch must not interfere with eager dispatch.
        op = Double()
        assert op(7) == 14
