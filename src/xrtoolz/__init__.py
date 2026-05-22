"""xrtoolz — composable operators for Earth System Data Cubes."""

from pipekit import Graph, Input, Node, Operator, Sequential, Tap

from xrtoolz.combinators import ApplyToEach, Augment
from xrtoolz.signature import Signature


__version__ = "0.0.8"

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
    "__version__",
]
