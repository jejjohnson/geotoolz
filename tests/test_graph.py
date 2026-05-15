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


class TestPositionalApply:
    """Graph._apply must accept positional args bound to declared inputs.

    Without this, a Graph cannot compose inside a Sequential (which calls
    ``op(value)`` positionally) or nest as a node operator inside another
    Graph (which evaluates nodes positionally from their parents).
    """

    def test_positional_single_input(self) -> None:
        x = Input("x")
        g = Graph(inputs={"x": x}, outputs={"y": Inc()(x)})
        # Positional, like a Sequential step would call it
        assert g(10) == {"y": 11}

    def test_positional_multi_input_in_declaration_order(self) -> None:
        a, b = Input("a"), Input("b")
        g = Graph(inputs={"a": a, "b": b}, outputs={"s": Sum2()(a, b)})
        assert g(3, 4) == {"s": 7}

    def test_positional_arity_mismatch_raises(self) -> None:
        x = Input("x")
        g = Graph(inputs={"x": x}, outputs={"y": Inc()(x)})
        with pytest.raises(TypeError, match="expected 1 positional"):
            g(1, 2)

    def test_positional_and_keyword_mutually_exclusive(self) -> None:
        x = Input("x")
        g = Graph(inputs={"x": x}, outputs={"y": Inc()(x)})
        with pytest.raises(TypeError, match="not both"):
            g(1, x=2)

    def test_graph_in_sequential(self) -> None:
        """A Sequential calling op(value) positionally must work for Graph ops."""
        from geotoolz.core import Sequential

        x = Input("x")
        inner = Graph(inputs={"x": x}, outputs={"y": Inc()(x)})
        # Sequential threads positional values through; a single-output Graph
        # returns a dict, which the next step in the chain must accept. Here
        # we wrap the dict-extract with a Lambda.
        from geotoolz.core import Lambda

        pipe = Sequential(
            [
                inner,
                Lambda(lambda d: d["y"], name="extract"),
                Inc(),
            ]
        )
        assert pipe(0) == 2  # 0 → {"y": 1} → 1 → 2

    def test_nested_graph(self) -> None:
        """A Graph node whose operator is itself a Graph must evaluate cleanly."""
        # Inner: single-input, single-output Graph
        xi = Input("xi")
        inner = Graph(inputs={"xi": xi}, outputs={"yi": Inc()(xi)})

        # Outer: wraps `inner` as a node, then adds one more step
        xo = Input("xo")
        # The wrapper extracts inner's dict output, since downstream Inc()
        # expects a scalar.
        from geotoolz.core import Lambda

        unwrap = Lambda(lambda d: d["yi"], name="unwrap")
        wrapped = unwrap(inner(xo))  # call chain: xo → inner → unwrap → ...
        outer_y = Inc()(wrapped)
        outer = Graph(inputs={"xo": xo}, outputs={"y": outer_y})

        assert outer(xo=10) == {"y": 12}  # 10 → inner → {yi:11} → 11 → 12


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
