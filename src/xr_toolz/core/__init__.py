"""Composition primitives — domain-agnostic.

Exports the :class:`Operator` base class, the :class:`Sequential` chain,
and the functional :class:`Graph` API (:class:`Input`, :class:`Node`,
:class:`Graph`).
"""

from xr_toolz.core.graph import Graph, Input, Node
from xr_toolz.core.operator import Operator
from xr_toolz.core.sequential import Sequential


__all__ = [
    "Graph",
    "Input",
    "Node",
    "Operator",
    "Sequential",
]
