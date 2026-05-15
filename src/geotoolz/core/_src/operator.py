"""Operator base class — the composition primitive.

Every concrete operator (`NDVI`, `MaskClouds`, `ModelOp`, ...) subclasses
`Operator` and implements `_apply`. The base class handles the two
behaviours every operator inherits:

1. **Dual-mode `__call__`** — `op(gt)` runs eagerly; `op(node)` records a
   `Node` in a `Graph` construction. The dispatch lives in `__call__` so
   subclasses only ever implement `_apply`.
2. **Config round-trip** — `get_config()` returns a JSON-serialisable dict
   of the constructor args, used for `__repr__`, pickling sanity, and
   Hydra-zen `builds()` integration.

Operators carrying user closures (`Tap`, `Lambda`, `Branch`, etc.) set
`forbid_in_yaml = True` to signal that their `get_config()` is a debug
repr, not a faithful YAML round-trip. The flag is documented for future
YAML loader enforcement; no runtime check yet.

See ``geotoolz`` design report §6.2 and `tips_n_tricks.md` §"Round-trip
discipline" for the full rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from geotoolz.core._src.sequential import Sequential


# The "carrier" that flows through a pipeline. In production this is
# ``georeader.GeoTensor``; the core composition primitives are
# deliberately generic over it so the same algebra works for
# ``ndarray`` / scalars / arbitrary objects in tests and for utility
# ops (``Const``, ``Lambda``) that accept any input type. Domain
# operators (``NDVI``, ``MaskClouds``, ...) narrow this to ``GeoTensor``
# at their own signatures.
Carrier = Any


class Operator:
    """Base class for geotoolz operators.

    Subclasses implement ``_apply(self, *args, **kwargs)``. The base
    class dispatches ``__call__`` to either ``_apply`` (eager mode) or
    `Node` construction (graph-building mode) based on whether any
    argument is a `Node` / `Input`.

    Attributes:
        forbid_in_yaml: ``True`` on subclasses that hold non-serialisable
            user state (callables, closures, runtime references). Future
            YAML loaders should refuse to instantiate flagged operators
            and YAML dumpers should refuse to serialise graphs containing
            them. Default ``False`` — most operators round-trip.
        _terminal: ``True`` on subclasses that legitimately return ``None``
            (or otherwise break the ``GeoTensor → GeoTensor`` contract).
            ``Sequential`` rejects terminal operators in any position
            except the last. Default ``False``.

    Examples:
        Implement a tiny operator::

            class Scale(Operator):
                def __init__(self, factor: float) -> None:
                    self.factor = factor

                def _apply(self, gt):
                    return gt * self.factor

                def get_config(self) -> dict:
                    return {"factor": self.factor}

        Compose with ``|``::

            pipeline = Scale(0.5) | Scale(2.0)   # Sequential([Scale(0.5), Scale(2.0)])
            result = pipeline(gt)
    """

    forbid_in_yaml: ClassVar[bool] = False
    _terminal: ClassVar[bool] = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch on argument type.

        If any positional argument is a graph node, returns a new ``Node``
        (graph-building mode). Otherwise calls ``_apply`` (eager mode).
        Subclasses should override ``_apply``, not ``__call__``.
        """
        # Lazy import to avoid circular dependency with graph.py
        from geotoolz.core._src.graph import Input, Node

        if any(isinstance(a, (Node, Input)) for a in args):
            return Node(operator=self, parents=tuple(args))
        out = self._apply(*args, **kwargs)
        # Reserved post-apply hook dispatch (Spy/Hook family lands in v0.2).
        # Keeping the call site here means adding hooks later won't require
        # editing every Operator subclass.
        self._dispatch_post_apply_hooks(args, out)
        return out

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the operator's behaviour. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement _apply().")

    def _dispatch_post_apply_hooks(self, args: tuple[Any, ...], out: Any) -> None:
        """No-op in v0.1. Reserved for Spy / observer hook dispatch in v0.2."""
        return None

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of constructor args.

        Override in subclasses to enable ``__repr__``, pickling assertions,
        and Hydra-zen ``builds()`` integration. Operators that hold user
        closures should still return what config they can but additionally
        set ``forbid_in_yaml = True``.
        """
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"{type(self).__name__}({params})"

    def __or__(self, other: Operator) -> Sequential:
        """``op1 | op2`` returns ``Sequential([op1, op2])``.

        Flattens nested ``Sequential`` instances on the right-hand side so
        ``a | (b | c)`` and ``(a | b) | c`` both produce a single
        three-element ``Sequential``.
        """
        # Lazy import: Sequential subclasses Operator.
        from geotoolz.core._src.sequential import Sequential

        if isinstance(other, Sequential):
            return Sequential([self, *other.operators])
        return Sequential([self, other])
