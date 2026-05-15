"""Tests for Graph, Input, Node."""

from __future__ import annotations

import pytest

from geotoolz.core import Graph, Input, Node, Operator


class Inc(Operator):
    """+1."""

    def _apply(self, x: int) -> int:
        return x + 1

    def get_config(self) -> dict:
        return {}


class Sum2(Operator):
    """Sum two inputs."""

    def _apply(self, a: int, b: int) -> int:
        return a + b

    def get_config(self) -> dict:
        return {}


class TestSingleInputSingleOutput:
    def test_eval(self) -> None:
        x = Input("x")
        y = Inc()(x)
        g = Graph(inputs={"x": x}, outputs={"y": y})
        assert g(x=10) == {"y": 11}


class TestMultiInput:
    def test_two_input_fusion(self) -> None:
        a, b = Input("a"), Input("b")
        out = Sum2()(a, b)
        g = Graph(inputs={"a": a, "b": b}, outputs={"sum": out})
        assert g(a=3, b=4) == {"sum": 7}


class TestBranchingOutputs:
    def test_two_outputs_share_input(self) -> None:
        x = Input("x")
        y1 = Inc()(x)
        y2 = Inc()(y1)  # x + 2
        g = Graph(inputs={"x": x}, outputs={"a": y1, "b": y2})
        result = g(x=0)
        assert result == {"a": 1, "b": 2}


class TestDiamond:
    def test_diamond_dependency(self) -> None:
        # x → a → c
        # x → b → c (c sums both)
        x = Input("x")
        a = Inc()(x)  # x + 1
        b = Inc()(a)  # x + 2
        c = Sum2()(a, b)  # (x+1) + (x+2) = 2x + 3
        g = Graph(inputs={"x": x}, outputs={"c": c})
        assert g(x=5) == {"c": 13}


class TestPassthrough:
    def test_output_is_input_directly(self) -> None:
        x = Input("x")
        g = Graph(inputs={"x": x}, outputs={"echo": x})
        assert g(x="hi") == {"echo": "hi"}


class TestErrors:
    def test_missing_input_raises(self) -> None:
        x = Input("x")
        g = Graph(inputs={"x": x}, outputs={"y": Inc()(x)})
        with pytest.raises(ValueError, match="missing required input"):
            g()

    def test_unknown_input_in_node_raises(self) -> None:
        x = Input("x")
        rogue = Input("rogue")
        # `rogue` referenced but not declared in inputs.
        with pytest.raises(ValueError, match="not declared"):
            Graph(inputs={"x": x}, outputs={"y": Inc()(rogue)})


class TestGetConfig:
    def test_config_lists_inputs_and_outputs(self) -> None:
        x = Input("x")
        y = Inc()(x)
        g = Graph(inputs={"x": x}, outputs={"y": y})
        cfg = g.get_config()
        assert cfg["inputs"] == ["x"]
        assert "y" in cfg["outputs"]
        assert cfg["outputs"]["y"]["class"] == "Inc"


class TestNodeIdentity:
    def test_distinct_nodes_for_same_op_on_same_input(self) -> None:
        # Each call to op(input) produces a new Node — graph topology
        # is built by Python identity, not by hashing.
        x = Input("x")
        op = Inc()
        n1 = op(x)
        n2 = op(x)
        assert isinstance(n1, Node)
        assert isinstance(n2, Node)
        assert n1 is not n2
