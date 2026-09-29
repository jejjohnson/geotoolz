"""Top-level package smoke tests."""

from __future__ import annotations

from typing import Any

import geotoolz


def test_import() -> None:
    assert geotoolz is not None


def test_version_attribute() -> None:
    assert isinstance(geotoolz.__version__, str)
    assert geotoolz.__version__.count(".") >= 2


def test_core_re_exports_at_top_level() -> None:
    """Public symbols should be reachable as ``gz.X`` (re-exported from pipekit)."""
    import pipekit

    pipekit_names = (
        "Operator",
        "Sequential",
        "Graph",
        "Input",
        "Node",
        "Tap",
        "Snapshot",
        "ShapeTrace",
        "Branch",
        "Switch",
        "Fanout",
        "Identity",
        "Const",
        "Lambda",
        "Sink",
    )
    for name in pipekit_names:
        assert getattr(geotoolz, name) is getattr(pipekit, name), name
    # ModelOp is geotoolz-specific (built on top of pipekit.Operator).
    from geotoolz.learn import ModelOp

    assert geotoolz.ModelOp is ModelOp


def test_from_state_refuses_forbid_in_yaml() -> None:
    """pipekit >= 0.0.2 refuses to rebuild a ``forbid_in_yaml`` operator.

    Guards the pipekit pin: with 0.0.1 this "succeeded" and returned an
    operator whose ``geometries == '<list n=0>'``.
    """
    import pytest
    from pipekit import Operator

    from geotoolz.geom import Rasterize

    op = Rasterize(geometries=[])
    assert type(op).forbid_in_yaml
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        Operator.from_state(op.state)


def test_no_operator_overrides_call() -> None:
    """Operators implement ``_apply``; ``Operator.__call__`` owns dispatch.

    Overriding ``__call__`` bypasses graph construction (``Node``) and the
    post-apply hook, so the operator cannot be used with ``Input``.
    """
    from _helpers import all_operator_classes

    offenders = [
        c.__qualname__ for c in all_operator_classes() if "__call__" in vars(c)
    ]
    assert offenders == []


def _multi_input_operators() -> list[Any]:
    from geotoolz.compositing import BlendMatched, StackMatched
    from geotoolz.geom import coregister as co

    return [
        StackMatched(),
        BlendMatched(),
        co.RasterToRasterLike(),
        co.SwathToGrid(target_crs="EPSG:4326", target_res=0.1),
        co.GridToSwath(),
        co.RasterToPoints(),
        co.PointsToRaster(),
        co.RasterToPointCloud(),
        co.PointCloudToRaster(),
        co.VectorToRasterAgg(agg="count"),
    ]


def test_formerly_call_overriding_operators_support_graph_mode() -> None:
    """Regression for #135: these ten used to override ``__call__``."""
    import inspect

    from pipekit import Input, Node

    for op in _multi_input_operators():
        params = inspect.signature(op._apply).parameters.values()
        n_inputs = sum(1 for p in params if p.default is inspect.Parameter.empty)
        node = op(*(Input(f"x{i}") for i in range(n_inputs)))
        assert isinstance(node, Node), type(op).__name__
        assert node.operator is op
