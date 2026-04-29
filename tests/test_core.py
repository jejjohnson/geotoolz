"""Operator contract + Sequential + Graph execution tests."""

from __future__ import annotations

import pytest

from xr_toolz.core import Graph, Input, Node, Operator, Sequential


class AddConst(Operator):
    """Single-input toy operator: returns ``x + const``."""

    def __init__(self, const: float):
        self.const = const

    def _apply(self, x):
        return x + self.const

    def get_config(self):
        return {"const": self.const}


class Mul(Operator):
    """Two-input toy operator: returns ``a * b``."""

    def _apply(self, a, b):
        return a * b


# --- Operator contract -------------------------------------------------------


def test_operator_eager_call():
    op = AddConst(const=1.0)
    assert op(2.0) == 3.0


def test_operator_get_config_and_repr():
    op = AddConst(const=2.5)
    assert op.get_config() == {"const": 2.5}
    assert repr(op) == "AddConst(const=2.5)"


def test_base_apply_raises():
    class Bare(Operator):
        pass

    with pytest.raises(NotImplementedError):
        Bare()(1)


def test_pipe_builds_sequential():
    chained = AddConst(1) | AddConst(2)
    assert isinstance(chained, Sequential)
    assert chained(0) == 3


def test_pipe_flattens_into_existing_sequential():
    tail = Sequential([AddConst(2), AddConst(3)])
    chained = AddConst(1) | tail
    assert isinstance(chained, Sequential)
    assert len(chained.operators) == 3
    assert chained(0) == 6


# --- Sequential --------------------------------------------------------------


def test_sequential_applies_in_order():
    pipeline = Sequential([AddConst(1), AddConst(10), AddConst(100)])
    assert pipeline(0) == 111


def test_sequential_nests():
    inner = Sequential([AddConst(1), AddConst(2)])
    outer = Sequential([inner, AddConst(10)])
    assert outer(0) == 13


def test_sequential_get_config_roundtrip():
    pipeline = Sequential([AddConst(1), AddConst(2)])
    config = pipeline.get_config()
    assert config == {
        "operators": [
            {"class": "AddConst", "config": {"const": 1}},
            {"class": "AddConst", "config": {"const": 2}},
        ]
    }


def test_sequential_describe_renders_tree():
    pipeline = Sequential([AddConst(1), AddConst(2)])
    text = pipeline.describe()
    assert text == "Sequential (2 ops)\n├── AddConst(const=1)\n└── AddConst(const=2)"


def test_sequential_describe_handles_nested_pipelines():
    inner = Sequential([AddConst(1), AddConst(2)])
    outer = Sequential([inner, AddConst(10)])
    text = outer.describe()
    assert (
        text == "Sequential (2 ops)\n"
        "├── Sequential (2 ops)\n"
        "│   ├── AddConst(const=1)\n"
        "│   └── AddConst(const=2)\n"
        "└── AddConst(const=10)"
    )


def test_sequential_describe_wraps_long_configs():
    class WideOp(AddConst):
        def get_config(self):
            return {f"k{i}": "x" * 8 for i in range(6)}

    pipeline = Sequential([WideOp(0)])
    lines = pipeline.describe(max_width=40).splitlines()
    assert lines[0] == "Sequential (1 ops)"
    assert lines[1].startswith("└── WideOp(")
    assert all(ln.startswith("    ") for ln in lines[2:])


def test_sequential_describe_closes_single_param_wrap():
    """A wrapped one-field config must still emit the closing paren."""

    class OneFieldOp(AddConst):
        def get_config(self):
            return {"only_field": "x" * 80}

    pipeline = Sequential([OneFieldOp(0)])
    text = pipeline.describe(max_width=40)
    last = text.splitlines()[-1]
    assert last.endswith(")"), f"missing close paren: {last!r}"


def test_sequential_describe_passes_reduced_width_into_nested():
    """Nested describe() must wrap against the remaining width, not the outer one."""

    class WideOp(AddConst):
        def get_config(self):
            return {"a": "x" * 30, "b": "y" * 30}

    inner = Sequential([WideOp(0)])
    outer = Sequential([Sequential([Sequential([inner])])])
    lines = outer.describe(max_width=80).splitlines()
    # No produced line may exceed max_width (this includes branch prefixes).
    assert all(len(ln) <= 80 for ln in lines), [
        (len(ln), ln) for ln in lines if len(ln) > 80
    ]


# --- Symbolic dispatch -------------------------------------------------------


def test_operator_returns_node_when_called_on_input():
    x = Input("x")
    node = AddConst(1)(x)
    assert isinstance(node, Node)
    assert node.operator is not None
    assert node.parents == (x,)


def test_operator_returns_node_when_called_on_node():
    x = Input("x")
    n1 = AddConst(1)(x)
    n2 = AddConst(2)(n1)
    assert isinstance(n2, Node)
    assert n2.parents == (n1,)


# --- Graph execution ---------------------------------------------------------


def test_graph_single_input_single_output_kwargs():
    x = Input("x")
    y = AddConst(10)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    out = graph(x=5)
    assert out == {"y": 15}


def test_graph_single_input_positional_shortcut():
    x = Input("x")
    y = AddConst(10)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    assert graph(5) == 15


def test_graph_multi_input_multi_output():
    a = Input("a")
    b = Input("b")
    prod = Mul()(a, b)
    sum_ab = AddConst(0)(a)  # reuse input "a"
    graph = Graph(
        inputs={"a": a, "b": b},
        outputs={"prod": prod, "a_passthrough": sum_ab},
    )
    out = graph(a=3, b=4)
    assert out == {"prod": 12, "a_passthrough": 3}


def test_graph_branching_from_shared_input():
    x = Input("x")
    plus1 = AddConst(1)(x)
    plus2 = AddConst(2)(x)
    graph = Graph(inputs={"x": x}, outputs={"a": plus1, "b": plus2})
    out = graph(x=0)
    assert out == {"a": 1, "b": 2}


def test_graph_rejects_missing_input_kwarg():
    a = Input("a")
    b = Input("b")
    prod = Mul()(a, b)
    graph = Graph(inputs={"a": a, "b": b}, outputs={"prod": prod})
    with pytest.raises(ValueError, match="missing"):
        graph(a=1)


def test_graph_rejects_unexpected_input_kwarg():
    a = Input("a")
    out = AddConst(1)(a)
    graph = Graph(inputs={"a": a}, outputs={"out": out})
    with pytest.raises(ValueError, match="unexpected"):
        graph(a=1, b=2)


def test_graph_rejects_positional_with_multiple_inputs():
    a = Input("a")
    b = Input("b")
    prod = Mul()(a, b)
    graph = Graph(inputs={"a": a, "b": b}, outputs={"prod": prod})
    with pytest.raises(ValueError, match="Positional"):
        graph(1)


def test_graph_rejects_unused_input():
    a = Input("a")
    b = Input("b")  # never used
    out = AddConst(1)(a)
    with pytest.raises(ValueError, match="not used"):
        Graph(inputs={"a": a, "b": b}, outputs={"out": out})


def test_graph_as_step_in_sequential():
    # A single-input/single-output Graph composes inside Sequential.
    x = Input("x")
    y = AddConst(5)(x)
    inner = Graph(inputs={"x": x}, outputs={"y": y})
    pipeline = Sequential([AddConst(1), inner, AddConst(100)])
    assert pipeline(0) == 106


def test_graph_describe_includes_inputs_and_outputs():
    x = Input("x")
    y = AddConst(10)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    text = graph.describe()
    assert "Inputs: ['x']" in text
    assert "Outputs: ['y']" in text


# --- Graph as Operator -------------------------------------------------------


def test_graph_is_an_operator_subclass():
    x = Input("x")
    y = AddConst(1)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    assert isinstance(graph, Operator)


def test_graph_nests_inside_another_graph():
    # Build the inner graph: single-in / single-out.
    ix = Input("ix")
    iy = AddConst(5)(ix)
    inner = Graph(inputs={"ix": ix}, outputs={"iy": iy})

    # Compose it symbolically into an outer graph.
    ox = Input("ox")
    via_inner = inner(ox)  # Node, not eager — because ox is an Input
    assert isinstance(via_inner, Node)
    final = AddConst(100)(via_inner)
    outer = Graph(inputs={"ox": ox}, outputs={"final": final})

    result = outer(ox=1)
    assert result == {"final": 106}


def test_graph_composes_with_pipe_operator():
    x = Input("x")
    y = AddConst(1)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    chained = graph | AddConst(10)
    assert isinstance(chained, Sequential)
    assert chained(0) == 11


def test_graph_get_config_is_json_serializable():
    import json

    x = Input("x")
    y = AddConst(5)(x)
    graph = Graph(inputs={"x": x}, outputs={"y": y})
    config = graph.get_config()
    # Must round-trip through JSON (rich PyTree state is excluded).
    json.dumps(config)
    assert config["inputs"] == {"x": 0}
    assert set(config["outputs"]) == {"y"}
