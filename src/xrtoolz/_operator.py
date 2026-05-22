"""xarray-aware :class:`Operator` — adds ``DataTree`` dispatch on top of pipekit.

``pipekit.Operator`` is carrier-agnostic and only knows two call modes:
eager ``_apply`` and symbolic ``Node``-construction. Earth-science data
also flows through ``xarray.DataTree`` (multi-group / multi-resolution
hierarchies). Rather than asking every diagnostic operator to opt in to
tree handling by hand, we widen the base class with one additional
branch: if any positional argument is a ``DataTree``, the operator is
mapped over every leaf via ``xr.map_over_datasets``.

The dispatch order matches the design doc (see
``docs/design/xarray-native-primitives.md``):

1. **Symbolic graph mode** — any ``Node`` argument routes to
   ``pipekit.Operator.__call__`` (which builds a ``Node`` recording this
   operator and its parents).
2. **DataTree mode** — any ``DataTree`` argument: apply
   ``xr.map_over_datasets`` to thread ``_apply`` over each matching
   leaf. ``xarray`` enforces the multi-input structural-match
   requirement; mixing a ``DataTree`` with a plain ``Dataset`` raises.
3. **Eager mode** — fall through to ``pipekit.Operator.__call__``,
   which runs ``_apply`` directly on the carrier(s).

Consequence: every operator that inherits from this class — every
``xrtoolz`` diagnostic, every combinator, every ``Sequential`` /
``Graph`` built out of them — gains ``DataTree`` support for free.
``Sequential`` and ``Graph`` themselves do not need changes: they are
carrier-agnostic in pipekit and simply thread whatever each step
returns to the next, so a ``DataTree`` in / ``DataTree`` out chain
composes naturally.

This is the implementation of "PR α" from the design doc.
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from pipekit import Node, Operator as _PipekitOperator


class Operator(_PipekitOperator):
    """``pipekit.Operator`` plus xarray ``DataTree`` dispatch.

    Subclasses implement ``_apply(carrier, *extra)`` exactly as they
    would against ``pipekit.Operator`` — the override only affects
    ``__call__`` dispatch, not ``_apply``.

    Example:
        A shape-preserving diagnostic works against any of the three
        xarray containers without per-class branching::

            class GaussianSmooth(Operator):
                def __init__(self, variable, *, dim, sigma):
                    self.variable, self.dim, self.sigma = variable, dim, sigma

                def _apply(self, ds: xr.Dataset) -> xr.Dataset:
                    return ds.assign({self.variable: ...})

            op = GaussianSmooth("ssh", dim="time", sigma=3)
            op(ds)   # → Dataset
            op(dt)   # → DataTree, each leaf smoothed independently
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch eager / symbolic / DataTree modes on positional args.

        - Any ``Node`` arg → pipekit's symbolic graph construction.
        - Any ``DataTree`` arg → leaf-wise map via
          ``xr.map_over_datasets``; the operator is applied to every
          matching ``Dataset`` leaf and the resulting leaves are
          reassembled into a ``DataTree`` with the input's structure.
          A node is skipped (via ``None`` return) only when **every**
          input leaf at that path is empty — this handles the synthetic
          root of a tree assembled from a flat ``{path: Dataset}`` dict
          without silently swallowing a real / empty mismatch in
          multi-input mode. ``DataArray`` returns are wrapped into a
          single-variable ``Dataset`` so ``map_over_datasets`` can stitch
          them into a result tree; anything other than ``Dataset`` /
          ``DataArray`` (e.g. a ``Figure`` from a terminal viz op) is
          rejected with a clear ``TypeError`` — terminal/visualisation
          operators are not meaningful in DataTree mode.
        - Otherwise → eager ``_apply`` via the pipekit base class.
        """
        if any(isinstance(a, Node) for a in args):
            return super().__call__(*args, **kwargs)
        if any(isinstance(a, xr.DataTree) for a in args):
            apply = self._apply
            cls_name = type(self).__name__

            def _leaf(*leaves: xr.Dataset) -> xr.Dataset | None:
                # Only skip when *every* input leaf is empty (synthetic
                # root of a ``from_dict`` tree). Partial emptiness should
                # fall through to ``_apply`` so the user sees a real
                # error rather than a silent no-op.
                if all(len(leaf.data_vars) == 0 for leaf in leaves):
                    return None
                result = apply(*leaves, **kwargs)
                if isinstance(result, xr.Dataset) or result is None:
                    return result
                if isinstance(result, xr.DataArray):
                    name = result.name if result.name is not None else "value"
                    return result.to_dataset(name=name)
                raise TypeError(
                    f"{cls_name}: DataTree dispatch requires _apply to return "
                    f"a Dataset or DataArray, got {type(result).__name__}. "
                    "Terminal / visualisation operators are not supported in "
                    "DataTree mode — call them on a single Dataset leaf."
                )

            return xr.map_over_datasets(_leaf, *args)
        return super().__call__(*args, **kwargs)
