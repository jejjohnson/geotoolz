"""geotoolz — composable Operator library for remote sensing.

Top-level re-exports of the composition core. Domain modules
(`radiometry`, `indices`, `cloud`, ...) will sit alongside ``core``
once they land — each is imported from its own submodule.

    import geotoolz as gz
    pipe = gz.Sequential([gz.Tap(print), gz.Identity()])
"""

from __future__ import annotations

from geotoolz import core
from geotoolz.core import (
    Branch,
    Carrier,
    Const,
    Fanout,
    Graph,
    Identity,
    Input,
    Lambda,
    ModelOp,
    Node,
    Operator,
    Sequential,
    ShapeTrace,
    Sink,
    Snapshot,
    Switch,
    Tap,
)


__version__ = "0.0.2"

__all__ = [
    "Branch",
    "Carrier",
    "Const",
    "Fanout",
    "Graph",
    "Identity",
    "Input",
    "Lambda",
    "ModelOp",
    "Node",
    "Operator",
    "Sequential",
    "ShapeTrace",
    "Sink",
    "Snapshot",
    "Switch",
    "Tap",
    "__version__",
    "core",
]
