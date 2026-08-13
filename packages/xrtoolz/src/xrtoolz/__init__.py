"""xrtoolz — composable operators for Earth System Data Cubes."""

from pipekit import Graph, Input, Node, Sequential, Tap

from xrcore import Operator, Signature
from xrtoolz.combinators import ApplyToEach, Augment


__version__ = "0.1.0"

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
