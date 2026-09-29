"""Top-level package smoke tests."""

from __future__ import annotations

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
