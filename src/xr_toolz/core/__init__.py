"""Composition primitives — domain-agnostic.

Exports the :class:`Operator` base class, the :class:`Sequential` chain,
the functional :class:`Graph` API (:class:`Input`, :class:`Node`,
:class:`Graph`), and the small operator combinators
(:class:`Augment`, :class:`Tap`, :class:`ApplyToEach`).
"""

from xr_toolz.core.combinators import ApplyToEach, Augment, Tap
from xr_toolz.core.graph import Graph, Input, Node
from xr_toolz.core.operator import Operator
from xr_toolz.core.sequential import Sequential
from xr_toolz.core.signature import Signature


__all__ = [
    "ApplyToEach",
    "Augment",
    "Graph",
    "Input",
    "Node",
    "Operator",
    "Sequential",
    "Signature",
    "Tap",
]
