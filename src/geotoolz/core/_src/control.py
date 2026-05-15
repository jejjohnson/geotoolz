"""Control-flow operators — Branch, Switch.

The `Operator` interface is general enough to express conditionals.
`Branch` is the binary case; `Switch` is multi-way.

The predicate (`Branch.predicate`) and the key (`Switch.key`) are plain
callables, not Operators — they make a boolean / hashable *decision*
about the input, they don't transform it. The arms / cases *are*
Operators because they perform the work.

Both flag ``forbid_in_yaml = True`` because the predicate / key callable
can't round-trip.

See `tips_n_tricks.md` §"Control flow".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from geotoolz.core._src.building_blocks import Identity
from geotoolz.core._src.operator import Carrier, Operator


class Branch(Operator):
    """Apply ``if_true`` when ``predicate(gt)`` is truthy, else
    ``if_false``.

    Args:
        predicate: Callable ``(gt) -> bool``.
        if_true: Operator applied when the predicate returns truthy.
        if_false: Operator applied otherwise. Default `Identity()`.

    Examples:
        Only correct atmospherically if the scene is reasonably clear::

            Sequential([
                Branch(
                    predicate=lambda gt: cloud_fraction(gt) < 0.3,
                    if_true=TOAToBOA(...),
                    if_false=Identity(),
                ),
                NDVI(...),
            ])
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        predicate: Callable[[Carrier], bool],
        if_true: Operator,
        if_false: Operator | None = None,
    ) -> None:
        if not isinstance(if_true, Operator):
            raise TypeError(
                f"if_true must be an Operator, got {type(if_true).__name__}."
            )
        if if_false is not None and not isinstance(if_false, Operator):
            raise TypeError(
                f"if_false must be an Operator or None, got {type(if_false).__name__}."
            )
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false if if_false is not None else Identity()

    def _apply(self, gt: Carrier) -> Any:
        # Output type is ``Any`` rather than ``Carrier``: each arm is an
        # arbitrary Operator and may legitimately transform the carrier.
        if self.predicate(gt):
            return self.if_true(gt)
        return self.if_false(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "predicate": getattr(self.predicate, "__name__", repr(self.predicate)),
            "if_true": {
                "class": type(self.if_true).__name__,
                "config": self.if_true.get_config(),
            },
            "if_false": {
                "class": type(self.if_false).__name__,
                "config": self.if_false.get_config(),
            },
        }


class Switch(Operator):
    """Multi-way dispatch on ``key(gt)``.

    Computes ``k = key(gt)``, then runs ``cases[k](gt)``. If ``k`` is not
    in ``cases``, runs ``default(gt)``.

    Args:
        key: Callable ``(gt) -> hashable`` whose return value selects a
            case. Common patterns: ``lambda gt: gt.metadata["sensor"]``,
            ``lambda gt: gt.dtype.name``.
        cases: Map of ``key-value → Operator``.
        default: Operator applied when no case matches. Default
            `Identity()` (silent passthrough). Pass a custom operator
            that raises if you want strict-mode behaviour.

    Examples:
        Cross-sensor pipeline::

            Switch(
                key=lambda gt: gt.metadata["sensor"],
                cases={
                    "S2":      S2_L2A_NDVI(),
                    "Landsat": L8_BOA_NDVI(),
                },
            )
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        key: Callable[[Carrier], Any],
        cases: dict[Any, Operator],
        default: Operator | None = None,
    ) -> None:
        for k, op in cases.items():
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Switch case {k!r} is {type(op).__name__}, expected Operator."
                )
        if default is not None and not isinstance(default, Operator):
            raise TypeError(
                f"default must be an Operator or None, got {type(default).__name__}."
            )
        self.key = key
        self.cases = dict(cases)
        self.default = default if default is not None else Identity()

    def _apply(self, gt: Carrier) -> Any:
        k = self.key(gt)
        op = self.cases.get(k, self.default)
        return op(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "key": getattr(self.key, "__name__", repr(self.key)),
            "cases": {
                str(k): {
                    "class": type(op).__name__,
                    "config": op.get_config(),
                }
                for k, op in self.cases.items()
            },
            "default": {
                "class": type(self.default).__name__,
                "config": self.default.get_config(),
            },
        }
