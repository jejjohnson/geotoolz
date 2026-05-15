"""Composition core — the operator algebra.

Public surface:
- `Operator` — base class for every operator
- `Sequential` — linear eager composition (also via the ``|`` operator)
- `Graph` / `Input` / `Node` — symbolic multi-input / multi-output graphs
- `Fanout` — sugar for one-input / many-outputs `Graph`s
- `ModelOp` — framework-agnostic inference wrapper
- `Tap` / `Snapshot` / `ShapeTrace` — identity operators with side effects
- `Branch` / `Switch` — control flow
- `Identity` / `Const` / `Lambda` / `Sink` — small building blocks

Underlying modules live in :mod:`geotoolz.core._src` and are private.
"""

from __future__ import annotations

from geotoolz.core._src.building_blocks import Const, Identity, Lambda, Sink
from geotoolz.core._src.composition import Fanout
from geotoolz.core._src.control import Branch, Switch
from geotoolz.core._src.graph import Graph, Input, Node
from geotoolz.core._src.model import ModelOp
from geotoolz.core._src.observers import ShapeTrace, Snapshot, Tap
from geotoolz.core._src.operator import Carrier, Operator
from geotoolz.core._src.sequential import Sequential


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
]
