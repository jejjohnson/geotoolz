"""Linear chain of single-input operators."""

from __future__ import annotations

from typing import Any

from xr_toolz.core.operator import Operator


class Sequential(Operator):
    """A pipeline of single-input operators, applied left to right.

    ``Sequential`` is itself an :class:`Operator`, so pipelines nest::

        preprocess = Sequential([ValidateCoords(), Regrid(grid)])
        full = Sequential([preprocess, RemoveClimatology(clim)])
    """

    def __init__(self, operators: list[Operator]):
        self.operators = list(operators)

    def _apply(self, ds: Any) -> Any:
        for op in self.operators:
            ds = op(ds)
        return ds

    def get_config(self) -> dict[str, Any]:
        return {
            "operators": [
                {"class": op.__class__.__name__, "config": op.get_config()}
                for op in self.operators
            ]
        }

    def describe(self) -> str:
        """Pretty-print the pipeline steps."""
        lines = [f"Sequential ({len(self.operators)} steps):"]
        for i, op in enumerate(self.operators):
            lines.append(f"  [{i}] {op!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Sequential({self.operators!r})"
