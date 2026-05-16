"""Sequential — eager linear composition of operators.

`Sequential` is the workhorse of geotoolz pipelines: a list of operators
applied left-to-right, each consuming the previous one's output.

    pipe = Sequential([MaskClouds(...), NDVI(...)])
    result = pipe(gt)

Equivalent and idiomatic via the ``|`` operator (inherited from
`Operator`)::

    pipe = MaskClouds(...) | NDVI(...)

`Sequential` is itself an `Operator`, so it composes recursively — a
`Sequential` of `Sequential`s is fine, and `__or__` flattens adjacent
`Sequential`s to keep the structure shallow.

Two invariants the constructor enforces:

1. **No terminal operators in non-terminal position.** Subclasses marked
   ``_terminal = True`` (e.g. `WriteCOG`, viz operators that legitimately
   return `None`) can only appear as the last step. Anywhere else, the
   pipeline would break the ``GeoTensor → next op`` contract.
2. **All steps are `Operator` instances.** Catching this at construction
   gives a clearer error than a runtime `AttributeError` deep in the
   chain.

See `geotoolz.md` §6.3 and §7 sharp edge #5.
"""

from __future__ import annotations

from typing import Any

from geotoolz.core._src.operator import Carrier, Operator


_MISSING = object()


class Sequential(Operator):
    """Apply a list of operators in order, threading the output of each
    into the next.

    Args:
        operators: A list of `Operator` instances. Empty list is allowed;
            calling an empty `Sequential` is the identity.

    Raises:
        TypeError: if any element of ``operators`` is not an `Operator`,
            or if any element except the last is marked ``_terminal``.

    Examples:
        Basic::

            pipe = Sequential([Scale(0.5), Scale(2.0)])
            assert pipe(gt) == gt  # 0.5 * 2.0

        Via the pipe operator::

            pipe = Scale(0.5) | Scale(2.0)

        Terminal op at the end is fine::

            pipe = Sequential([NDVI(...), WriteCOG("/out.tif")])

        Terminal op anywhere else raises::

            Sequential([WriteCOG("/x"), NDVI(...)])  # TypeError
    """

    def __init__(self, operators: list[Operator]) -> None:
        for i, op in enumerate(operators):
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Sequential[{i}] is {type(op).__name__}, expected Operator."
                )
        for i, op in enumerate(operators[:-1]):
            if op._terminal:
                raise TypeError(
                    f"Sequential[{i}] is a terminal operator "
                    f"({type(op).__name__}) — terminal operators are only "
                    "valid as the last step of a Sequential."
                )
        self.operators = list(operators)

    def _apply(self, gt: Carrier = _MISSING) -> Any:
        # Return type stays ``Any`` rather than ``Carrier``: a Sequential
        # may legitimately reduce (e.g. end in a ``Mean`` that returns a
        # scalar) so we can't promise the carrier shape survives.
        if gt is _MISSING and not self.operators:
            raise TypeError("Sequential([]) requires an input value.")
        if gt is _MISSING:
            out = self.operators[0]()
            operators = self.operators[1:]
        else:
            out = gt
            operators = self.operators
        for op in operators:
            out = op(out)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "operators": [
                {"class": type(op).__name__, "config": op.get_config()}
                for op in self.operators
            ]
        }

    def __or__(self, other: Operator) -> Sequential:
        """Append on the right; flatten nested `Sequential`s.

        ``Sequential([a, b]) | c`` → ``Sequential([a, b, c])``
        ``Sequential([a, b]) | Sequential([c, d])`` → ``Sequential([a, b, c, d])``
        """
        if isinstance(other, Sequential):
            return Sequential([*self.operators, *other.operators])
        if not isinstance(other, Operator):
            return NotImplemented
        return Sequential([*self.operators, other])

    def __repr__(self) -> str:
        if not self.operators:
            return "Sequential([])"
        inner = ", ".join(repr(op) for op in self.operators)
        return f"Sequential([{inner}])"

    def __len__(self) -> int:
        return len(self.operators)

    def __getitem__(self, key: int | slice) -> Operator | Sequential:
        """Index or slice into the underlying operator list.

        ``pipe[0]`` returns the first operator. ``pipe[1:3]`` returns a
        new `Sequential` containing the slice.
        """
        if isinstance(key, slice):
            return Sequential(self.operators[key])
        return self.operators[key]
