"""xarray-aware :class:`Operator` — adds ``DataTree`` dispatch on top of pipekit.

``pipekit_ops.Operator`` is carrier-agnostic and only knows two call modes:
eager ``_apply`` and symbolic ``Node``-construction. Earth-science data
also flows through ``xarray.DataTree`` (multi-group / multi-resolution
hierarchies). Rather than asking every diagnostic operator to opt in to
tree handling by hand, we widen the base class with one additional
branch: if any positional argument is a ``DataTree``, the operator is
mapped over every leaf via ``xr.map_over_datasets``.

The dispatch order matches the design doc (see
``docs/design/xarray-native-primitives.md``):

1. **Symbolic graph mode** — any ``Node`` argument routes to
   ``pipekit_ops.Operator.__call__`` (which builds a ``Node`` recording this
   operator and its parents).
2. **DataTree mode** — any ``DataTree`` argument: call ``_apply_tree``,
   whose default applies ``xr.map_over_datasets`` to thread ``_apply``
   over each matching leaf. Operators that need the whole tree at once
   (e.g. to fit one model across nodes) override ``_apply_tree``.
   ``xarray`` enforces the multi-input structural-match requirement;
   mixing a ``DataTree`` with a plain ``Dataset`` raises.
3. **Eager mode** — fall through to ``pipekit_ops.Operator.__call__``,
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

import functools
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

import xarray as xr
from pipekit_ops import (
    BackendError,
    Node,
    Operator as _PipekitOperator,
    backend_of,
    report_move,
    to_backend,
)


def _parts(value: Any) -> Iterator[Any]:
    """What of ``value`` has a backend: its variables, leaves or items, or itself."""
    if isinstance(value, xr.DataTree):
        for node in value.subtree:
            yield from node.data_vars.values()
    elif isinstance(value, xr.Dataset):
        yield from value.data_vars.values()
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _parts(item)
    else:
        yield value


def _raw(part: Any) -> Any:
    """The array inside a DataArray; anything else unchanged."""
    return part.data if isinstance(part, xr.DataArray) else part


def _moved(value: Any, backend: Literal["numpy", "jax"]) -> Any:
    """`pipekit_ops.to_backend`, into Datasets and DataTrees too."""
    if isinstance(value, xr.DataTree):
        return value.map_over_datasets(lambda ds: _moved(ds, backend))
    if isinstance(value, xr.Dataset):
        arrays = {name: to_backend(da, backend) for name, da in value.data_vars.items()}
        return value.assign(arrays)
    if type(value) in (list, tuple):
        return type(value)(_moved(item, backend) for item in value)
    return to_backend(value, backend)


class Operator(_PipekitOperator):
    """``pipekit_ops.Operator`` plus xarray ``DataTree`` dispatch.

    Subclasses implement ``_apply(carrier, *extra)`` exactly as they
    would against ``pipekit_ops.Operator`` — the override only affects
    ``__call__`` dispatch, not ``_apply``.

    Example:
        One ``_apply`` implementation serves ``Dataset`` and
        ``DataTree`` callers alike — the tree case maps over every leaf:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrops import Operator
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

    #: The backends of its inputs' data (OP-O6): every xrops operator takes
    #: NumPy- and dask-backed carriers; one that runs elsewhere (or refuses
    #: dask) overrides this. A Dataset counts by its data variables. JAX-backed
    #: carriers go to the host and back (pipekit#253).
    backends: ClassVar[tuple[str, ...] | None] = ("numpy", "dask")

    def _place_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        backends: tuple[str, ...] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
        """The inputs as they are: :meth:`__call__` has placed them."""
        return args, kwargs, False

    def _placed(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
        """pipekit's backend rule on xarray, and whether JAX data left JAX.

        A NumPy operand beside a dask one (a static mask, an in-memory
        reference field) joins the dask graph without moving: xarray folds it
        into the lazy computation. JAX-backed inputs of an operator that does
        not list JAX go to the host, and :meth:`__call__` returns the result to
        JAX; `pipekit_ops.guard` reports it. A Dataset counts by its data
        variables, a DataTree by its leaves.

        Raises:
            BackendError: JAX beside dask, traced data in an operator without
                JAX, or a backend that cannot move to one it lists.
        """
        if self.backends is None:
            return args, kwargs, False
        name, runs = type(self).__name__, " and ".join(self.backends)
        parts = [p for value in (*args, *kwargs.values()) for p in _parts(value)]
        found = {backend_of(part) for part in parts} - {None}
        if found == {"numpy", "dask"}:
            found = {"dask"}
        if len(found) < 2 and found <= set(self.backends):
            return args, kwargs, False
        if found <= {"numpy", "jax"} and {"numpy", "jax"} & set(self.backends):
            target = "jax" if "jax" in self.backends else "numpy"
            if target == "jax":
                report_move(name, "NumPy data joined jax")
            elif any("Tracer" in type(_raw(part)).__name__ for part in parts):
                raise BackendError(
                    f"{name} runs on {runs} only, so it cannot run inside jax.jit, "
                    "grad or vmap: the data is traced and has no values."
                )
            else:
                report_move(name, "JAX data copied to the host")
            moved = tuple(_moved(value, target) for value in args)
            named = {key: _moved(value, target) for key, value in kwargs.items()}
            return moved, named, target == "numpy"
        if len(found) > 1:
            raise BackendError(
                f"{name}: inputs on {sorted(map(str, found))}; put them on one first."
            )
        raise BackendError(
            f"{name} runs on {runs} only, not {found.pop()}. Convert the input first."
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Make each ``compute_output_signature`` override pass an unknown on.

        An upstream step that does not track shape (a nested `pipekit_ops.Graph`,
        whose output is a dict) hands ``None`` to the next step; the operator
        then has nothing to transform and reports ``None`` too, rather than
        failing on ``None.dims``.
        """
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("compute_output_signature")
        if own is None or getattr(own, "_passes_unknown", False):
            return

        @functools.wraps(own)
        def compute_output_signature(self: Any, input_signature: Any) -> Any:
            if input_signature is None:
                return None
            return own(self, input_signature)

        compute_output_signature.__dict__["_passes_unknown"] = True
        cls.compute_output_signature = compute_output_signature

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
        # the declared backends hold for every leaf of a tree too
        args, kwargs, on_host = self._placed(args, kwargs)
        if any(isinstance(a, xr.DataTree) for a in args):
            out = self._apply_tree(*args, **kwargs)
        else:
            out = super().__call__(*args, **kwargs)
        return _moved(out, "jax") if on_host else out

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
            >>> from xrops import Operator
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
