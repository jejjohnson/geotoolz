"""Top-level package smoke tests."""

from __future__ import annotations

import geotoolz


def test_import() -> None:
    assert geotoolz is not None


def test_version_attribute() -> None:
    assert isinstance(geotoolz.__version__, str)
    assert geotoolz.__version__.count(".") >= 2


def test_core_re_exports_at_top_level() -> None:
    """Public symbols should be reachable as ``gz.X`` and ``gz.core.X``."""
    for name in (
        "Operator",
        "Sequential",
        "Graph",
        "Input",
        "Node",
        "ModelOp",
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
    ):
        assert getattr(geotoolz, name) is getattr(geotoolz.core, name), name
