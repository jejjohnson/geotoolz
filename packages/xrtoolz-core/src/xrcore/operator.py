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
2. **DataTree mode** — any ``DataTree`` argument: call ``_apply_tree``,
   whose default applies ``xr.map_over_datasets`` to thread ``_apply``
   over each matching leaf. Operators that need the whole tree at once
   (e.g. to fit one model across nodes) override ``_apply_tree``.
   ``xarray`` enforces the multi-input structural-match requirement;
   mixing a ``DataTree`` with a plain ``Dataset`` raises.
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
        One ``_apply`` implementation serves ``Dataset`` and
        ``DataTree`` callers alike — the tree case maps over every leaf:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrcore import Operator
        >>> class Scale(Operator):
        ...     def __init__(self, factor: float) -> None:
        ...         self.factor = factor
        ...     def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        ...         return ds * self.factor
        >>> ds = xr.Dataset({"ssh": ("x", np.arange(3.0))})
        >>> Scale(10.0)(ds)["ssh"].values
        array([ 0., 10., 20.])
        >>> tree = xr.DataTree.from_dict({"coarse": ds, "fine": ds * 2})
        >>> out = Scale(10.0)(tree)
        >>> sorted(out.children), float(out["fine"]["ssh"][2])
        (['coarse', 'fine'], 40.0)

        ```
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch eager / symbolic / DataTree modes on positional args.

        - Any ``Node`` arg → pipekit's symbolic graph construction.
        - Any ``DataTree`` arg → :meth:`_apply_tree`. Subclasses that need
          the whole tree override it; the default is a leaf-wise map via
          ``xr.map_over_datasets``: the operator is applied to every
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
            return self._apply_tree(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _apply_tree(self, *args: Any, **kwargs: Any) -> Any:
        """Handle a call whose positional arguments include a ``DataTree``.

        The default maps :meth:`_apply` over every matching leaf (see
        :meth:`__call__`). Override it when an operator needs the *whole*
        tree at once — e.g. to fit one model across several nodes, or to
        route each node to its own fitted state — rather than one leaf at a
        time. Overrides receive exactly the arguments ``__call__`` got.

        Args:
            *args: Positional call arguments; at least one is a DataTree.
            **kwargs: Keyword call arguments, forwarded to :meth:`_apply`.

        Returns:
            The operator's result for the tree (a ``DataTree`` for the
            default leaf-wise map).

        Raises:
            TypeError: If ``_apply`` returns something other than a
                ``Dataset`` / ``DataArray`` for a leaf.

        Example:
            Normalise a variable by its maximum over *every* node, which a
            leaf-at-a-time ``_apply`` cannot see:

            ```pycon
            >>> import xarray as xr
            >>> from xrcore import Operator
            >>> class GlobalScale(Operator):
            ...     def _apply(self, ds):
            ...         return ds
            ...     def _apply_tree(self, tree):
            ...         peak = max(float(n["v"].max()) for n in tree.leaves)
            ...         return tree.map_over_datasets(
            ...             lambda ds: ds / peak if ds.data_vars else ds
            ...         )
            >>> tree = xr.DataTree.from_dict({
            ...     "a": xr.Dataset({"v": ("x", [1.0, 2.0])}),
            ...     "b": xr.Dataset({"v": ("x", [4.0, 8.0])}),
            ... })
            >>> GlobalScale()(tree)["a"]["v"].values
            array([0.125, 0.25 ])

            ```
        """
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
