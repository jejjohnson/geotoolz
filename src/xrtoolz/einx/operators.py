"""Layer-1 ``Operator`` wrappers around :mod:`xrtoolz.einx` functions.

One operator per Layer-0 function (D7). Each subclasses
:class:`xrtoolz.Operator`, so it inherits DataTree leaf-wise dispatch and
composes inside ``Sequential`` / ``Graph`` / ``Augment`` unchanged.
``compute_output_signature`` threads dim shapes through pipeline
summaries without executing data paths.

``pack_dataset`` / ``unpack_dataset`` are intentionally **not** wrapped
as operators (D6) — they are plain functions in
:mod:`xrtoolz.einx.dataset`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.einx._src._pattern import infer_output_signature
from xrtoolz.einx._src.core import einsum, rearrange, reduce, repeat
from xrtoolz.einx._src.linalg import batch_matmul, matmul, outer
from xrtoolz.signature import Signature


def _as_sigs(
    input_signature: Signature | tuple[Signature, ...],
) -> tuple[Signature, ...]:
    if isinstance(input_signature, Signature):
        return (input_signature,)
    return tuple(input_signature)


def _coords_config(coords: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """JSON-safe, round-trippable form of a ``coords`` mapping.

    Array-like coord values are converted to plain lists so the config
    survives ``json.dumps`` and ``cls(**get_config())`` reconstructs an
    equivalent operator (xarray accepts list coords).
    """
    if coords is None:
        return None
    import numpy as np

    out: dict[str, Any] = {}
    for name, value in coords.items():
        if isinstance(value, np.ndarray):
            out[name] = value.tolist()
        elif hasattr(value, "values"):
            out[name] = np.asarray(value.values).tolist()
        else:
            out[name] = value
    return out


class Einsum(Operator):
    """Named-tensor Einstein summation over one or more DataArrays.

    Applies :func:`xrtoolz.einx.einsum` using DataArray dim names as the
    pattern's axis tokens, e.g. ``"b i j, b j k -> b i k"``.

    Args:
        pattern: einx einsum pattern; axis tokens are DataArray dim names.
        coords: Optional coordinates to attach to newly created output dims.
        align: When ``True``, broadcast-align inputs on shared dims before
            contracting.
        **shape_kwargs: Sizes for composite/unknown axes in ``pattern``.

    Returns:
        The contracted DataArray with the dims named on the pattern's
        right-hand side.
    """

    def __init__(
        self,
        pattern: str,
        *,
        coords: Mapping[str, Any] | None = None,
        align: bool = False,
        **shape_kwargs: int,
    ) -> None:
        self.pattern = pattern
        self.coords = coords
        self.align = align
        self.shape_kwargs = dict(shape_kwargs)

    def _apply(self, *das: xr.DataArray) -> xr.DataArray:
        return einsum(
            self.pattern,
            *das,
            coords=self.coords,
            align=self.align,
            **self.shape_kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"pattern": self.pattern, "align": self.align}
        cfg.update(self.shape_kwargs)
        if self.coords is not None:
            cfg["coords"] = _coords_config(self.coords)
        return cfg

    def __repr__(self) -> str:
        return f"Einsum({self.pattern!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        return infer_output_signature(
            self.pattern, _as_sigs(input_signature), self.shape_kwargs
        )


class Rearrange(Operator):
    """Reshape / transpose a DataArray by a named-tensor pattern.

    Applies :func:`xrtoolz.einx.rearrange`, e.g. ``"(h w) c -> h w c"`` to
    split a flattened axis into named dims.

    Args:
        pattern: einx rearrange pattern over DataArray dim names.
        coords: Optional coordinates for newly created dims.
        **shape_kwargs: Sizes for composite axes that cannot be inferred.

    Returns:
        The rearranged DataArray.
    """

    def __init__(
        self,
        pattern: str,
        *,
        coords: Mapping[str, Any] | None = None,
        **shape_kwargs: int,
    ) -> None:
        self.pattern = pattern
        self.coords = coords
        self.shape_kwargs = dict(shape_kwargs)

    def _apply(self, da: xr.DataArray) -> xr.DataArray:
        return rearrange(self.pattern, da, coords=self.coords, **self.shape_kwargs)

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"pattern": self.pattern}
        cfg.update(self.shape_kwargs)
        if self.coords is not None:
            cfg["coords"] = _coords_config(self.coords)
        return cfg

    def __repr__(self) -> str:
        return f"Rearrange({self.pattern!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        return infer_output_signature(
            self.pattern, _as_sigs(input_signature), self.shape_kwargs
        )


class Reduce(Operator):
    """Reduce a DataArray over named axes with a chosen reduction.

    Applies :func:`xrtoolz.einx.reduce`, e.g. ``"b [i] j -> b j"`` to reduce
    the bracketed axis ``i``.

    Note:
        Distinct from :class:`xrtoolz.geo.Reduce`, which aggregates a whole
        Dataset over xarray dims; this reduces a single DataArray by an einx
        pattern.

    Args:
        pattern: einx reduce pattern; bracketed axes are reduced.
        op: Reduction — a name (``"sum"``, ``"mean"``, ``"max"``, …) or a
            callable accepting an ``axis`` argument.
        **shape_kwargs: Sizes for composite axes that cannot be inferred.

    Returns:
        The reduced DataArray.
    """

    def __init__(
        self,
        pattern: str,
        *,
        op: Callable[..., Any] | str,
        **shape_kwargs: int,
    ) -> None:
        self.pattern = pattern
        self.op = op
        self.shape_kwargs = dict(shape_kwargs)

    def _apply(self, da: xr.DataArray) -> xr.DataArray:
        return reduce(self.pattern, da, op=self.op, **self.shape_kwargs)

    def get_config(self) -> dict[str, Any]:
        op = (
            self.op
            if isinstance(self.op, str)
            else getattr(self.op, "__name__", repr(self.op))
        )
        return {"pattern": self.pattern, "op": op, **self.shape_kwargs}

    def __repr__(self) -> str:
        return f"Reduce({self.pattern!r}, op={self.op!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        return infer_output_signature(
            self.pattern, _as_sigs(input_signature), self.shape_kwargs
        )


class Repeat(Operator):
    """Broadcast / tile a DataArray along new named axes.

    Applies :func:`xrtoolz.einx.repeat`, e.g. ``"h w -> h w c"`` with
    ``c=3`` to add and fill a new dimension.

    Args:
        pattern: einx repeat pattern over DataArray dim names.
        coords: Optional coordinates for the repeated dims.
        **shape_kwargs: Sizes of the new axes introduced by ``pattern``.

    Returns:
        The repeated DataArray.
    """

    def __init__(
        self,
        pattern: str,
        *,
        coords: Mapping[str, Any] | None = None,
        **shape_kwargs: int,
    ) -> None:
        self.pattern = pattern
        self.coords = coords
        self.shape_kwargs = dict(shape_kwargs)

    def _apply(self, da: xr.DataArray) -> xr.DataArray:
        return repeat(self.pattern, da, coords=self.coords, **self.shape_kwargs)

    def get_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"pattern": self.pattern}
        cfg.update(self.shape_kwargs)
        if self.coords is not None:
            cfg["coords"] = _coords_config(self.coords)
        return cfg

    def __repr__(self) -> str:
        return f"Repeat({self.pattern!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        return infer_output_signature(
            self.pattern, _as_sigs(input_signature), self.shape_kwargs
        )


class _BinaryLinalgOp(Operator):
    """Shared scaffolding for the two-input linalg convenience operators."""

    def _signature_from_dims(
        self, dims: Sequence[str], sigs: tuple[Signature, ...]
    ) -> Signature:
        known: dict[str, int | None] = {}
        for sig in sigs:
            for name, size in sig.dims.items():
                known.setdefault(name, size)
        out = {name: known.get(name) for name in dims}
        dtypes = [s.dtype for s in sigs if s.dtype is not None]
        dtype = None
        if dtypes:
            import numpy as np

            dtype = dtypes[0]
            for dt in dtypes[1:]:
                dtype = np.promote_types(dtype, dt)
        return Signature(out, dtype=dtype)


class Matmul(_BinaryLinalgOp):
    """Contract two DataArrays over a shared dimension (matrix multiply).

    Applies :func:`xrtoolz.einx.matmul`: sums the product over ``dim``,
    keeping all other (broadcast) dims.

    Args:
        dim: Name of the shared dimension contracted over.

    Returns:
        The matrix product, with ``dim`` removed.
    """

    def __init__(self, *, dim: str) -> None:
        self.dim = dim

    def _apply(self, a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        return matmul(a, b, dim=self.dim)

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def __repr__(self) -> str:
        return f"Matmul(dim={self.dim!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        a_sig, b_sig = _as_sigs(input_signature)
        dims = [d for d in a_sig.dims if d != self.dim] + [
            d for d in b_sig.dims if d != self.dim
        ]
        return self._signature_from_dims(dims, (a_sig, b_sig))


class Outer(_BinaryLinalgOp):
    """Outer product of two DataArrays over their disjoint dims.

    Applies :func:`xrtoolz.einx.outer`: the result carries every dim of the
    first input followed by every dim of the second.

    Returns:
        The outer-product DataArray.
    """

    def _apply(self, a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        return outer(a, b)

    def get_config(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return "Outer()"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        a_sig, b_sig = _as_sigs(input_signature)
        return self._signature_from_dims([*a_sig.dims, *b_sig.dims], (a_sig, b_sig))


class BatchMatmul(_BinaryLinalgOp):
    """Batched matrix multiply: contract ``dim`` within shared batch dims.

    Applies :func:`xrtoolz.einx.batch_matmul`, contracting ``dim`` while
    broadcasting over ``batch_dims`` (which are kept on the output).

    Args:
        dim: Name of the shared dimension contracted over.
        batch_dims: Dimensions broadcast over (not contracted), kept on the
            output.

    Returns:
        The batched matrix product, with ``dim`` removed.
    """

    def __init__(self, *, dim: str, batch_dims: Sequence[str] = ()) -> None:
        self.dim = dim
        self.batch_dims = tuple(batch_dims)

    def _apply(self, a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
        return batch_matmul(a, b, dim=self.dim, batch_dims=self.batch_dims)

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim, "batch_dims": list(self.batch_dims)}

    def __repr__(self) -> str:
        return f"BatchMatmul(dim={self.dim!r}, batch_dims={list(self.batch_dims)!r})"

    def compute_output_signature(
        self, input_signature: Signature | tuple[Signature, ...]
    ) -> Signature:
        a_sig, b_sig = _as_sigs(input_signature)
        batch = list(self.batch_dims)
        a_rest = [d for d in a_sig.dims if d != self.dim and d not in batch]
        b_rest = [d for d in b_sig.dims if d != self.dim and d not in batch]
        return self._signature_from_dims([*batch, *a_rest, *b_rest], (a_sig, b_sig))


__all__ = [
    "BatchMatmul",
    "Einsum",
    "Matmul",
    "Outer",
    "Rearrange",
    "Reduce",
    "Repeat",
]
