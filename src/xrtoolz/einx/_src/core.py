"""Layer-0 labeled einx primitives — DataArray in, DataArray out.

Each function parses an einx pattern whose axis tokens are DataArray
*dim names*, reconciles the inputs (name-matched transpose + coord
policy), dispatches to einx on the underlying arrays, and rewraps the
result as a labeled DataArray with coords forwarded from the inputs.

Semantics summary:

- ``einsum`` / ``reduce`` / ``repeat`` are **name-matched**: each input
  slot lists the input's dim names (in any order); the bridge transposes
  to the slot order before dispatch, so patterns are independent of how
  upstream code ordered the dims.
- ``rearrange`` is **positional** (like einx itself): merge / split
  groups have no single dim name to match against, so the input slot
  describes the array's existing axes *in order*. Its labeled value is
  naming the output dims.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from xrtoolz.einx._src._pattern import EinxPattern, parse_pattern
from xrtoolz.einx._src.coords import forward_coords
from xrtoolz.einx._src.errors import CoordMismatch, PatternError


_NATIVE_REDUCERS = frozenset(
    {"sum", "mean", "min", "max", "prod", "any", "all", "std", "var"}
)


def einsum(
    pattern: str,
    *arrays: xr.DataArray,
    coords: Mapping[str, Any] | None = None,
    align: bool = False,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Labeled einx einsum (contraction) over named dims.

    Args:
        pattern: einx pattern; each axis token is a DataArray dim name.
        *arrays: one input DataArray per pattern input slot. Each input's
            slot must reference exactly that array's dims; the bridge
            transposes to match before dispatch.
        coords: explicit coords for output dims absent from every input.
        align: if ``False`` (default), shared dims with mismatched coords
            raise :class:`CoordMismatch`. If ``True``, inputs are inner-
            joined with ``xr.align`` before dispatch.
        **shape_kwargs: sizes for pattern axes whose size cannot be
            inferred from the inputs.

    Returns:
        DataArray with dims given by the pattern's output slot, coords
        forwarded from the first input carrying each surviving dim.

    Example:
        >>> total = einsum("time lat lon, lat lon -> time", field, mask)
    """
    import einx

    parsed = parse_pattern(pattern)
    if len(arrays) != len(parsed.inputs):
        raise PatternError(
            f"Pattern declares {len(parsed.inputs)} input(s) but {len(arrays)} "
            "array(s) were passed."
        )
    prepared = _prepare_name_matched(arrays, parsed, align=align)
    values = [da.values for da in prepared]
    result = einx.dot(pattern, *values, **shape_kwargs)
    return _wrap(result, parsed, prepared, coords)


def rearrange(
    pattern: str,
    da: xr.DataArray,
    coords: Mapping[str, Any] | None = None,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Labeled einx rearrange (reshape / transpose / merge / split).

    Unlike :func:`einsum`, the input slot describes ``da``'s axes **in
    their current order** (merge / split groups have no single dim name
    to match). Output dims are named from the output slot; a merged group
    ``(a b)`` becomes a single dim named ``a_b``. Surviving named dims
    keep their coord; new / merged dims are unindexed unless ``coords``
    supplies them.

    Example:
        >>> patches = rearrange(
        ...     "time (lat_blk lat_in) (lon_blk lon_in) "
        ...     "-> time (lat_blk lon_blk) lat_in lon_in",
        ...     field, lat_in=4, lon_in=4,
        ... )
    """
    import einx

    parsed = parse_pattern(pattern)
    if len(parsed.inputs) != 1:
        raise PatternError("rearrange takes exactly one input slot.")
    result = einx.id(pattern, da.values, **shape_kwargs)
    return _wrap(result, parsed, [da], coords)


def reduce(
    pattern: str,
    da: xr.DataArray,
    *,
    op: Callable[..., Any] | str,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Reduce over named axes (axes on the input slot, absent from output).

    Args:
        pattern: einx pattern; reduced dims appear only on the input slot.
        da: input DataArray.
        op: reduction op. Strings ``'sum' | 'mean' | 'min' | 'max' |
            'prod' | 'std' | 'var' | 'any' | 'all'`` dispatch to einx's
            native reducers; ``'median'`` and any callable route through
            einx's numpy-like reduce adapter for the input's backend.

    Example:
        >>> climatology = reduce(
        ...     "time lat lon -> lat lon", sst, op="mean",
        ... )
    """
    parsed = parse_pattern(pattern)
    if len(parsed.inputs) != 1:
        raise PatternError("reduce takes exactly one input slot.")
    prepared = _prepare_name_matched((da,), parsed, align=False)
    reducer = _resolve_reducer(op, prepared[0].values)
    result = reducer(pattern, prepared[0].values, **shape_kwargs)
    return _wrap(result, parsed, prepared, None)


def repeat(
    pattern: str,
    da: xr.DataArray,
    coords: Mapping[str, Any] | None = None,
    **shape_kwargs: int,
) -> xr.DataArray:
    """Broadcast / tile along new named axes.

    Name-matched like :func:`einsum`: the input slot lists ``da``'s dims;
    the output slot adds new dims whose sizes come from ``shape_kwargs``.
    New dims are unindexed unless ``coords`` supplies them.

    Example:
        >>> seasonal = repeat(
        ...     "lat lon -> month lat lon", mean_field, month=12,
        ... )
    """
    import einx

    parsed = parse_pattern(pattern)
    if len(parsed.inputs) != 1:
        raise PatternError("repeat takes exactly one input slot.")
    prepared = _prepare_name_matched((da,), parsed, align=False)
    result = einx.id(pattern, prepared[0].values, **shape_kwargs)
    return _wrap(result, parsed, prepared, coords)


# ---------- internals ------------------------------------------------------


def _prepare_name_matched(
    arrays: tuple[xr.DataArray, ...],
    parsed: EinxPattern,
    *,
    align: bool,
) -> list[xr.DataArray]:
    """Validate dim names, reconcile coords, and transpose to slot order."""
    for index, (arr, slot) in enumerate(zip(arrays, parsed.inputs, strict=True)):
        if any(not isinstance(el, str) for el in slot):
            raise PatternError(
                "einsum / reduce / repeat input slots must be flat dim names; "
                "use rearrange for '(group)' merge/split restructuring."
            )
        names = parsed.flat_input_names(index)
        if set(names) != set(arr.dims):
            raise PatternError(
                f"Input {index} slot {names} does not match the array's dims "
                f"{tuple(arr.dims)}."
            )

    arrays = _reconcile_coords(arrays, align=align)
    return [
        arr.transpose(*parsed.flat_input_names(index))
        for index, arr in enumerate(arrays)
    ]


def _reconcile_coords(
    arrays: tuple[xr.DataArray, ...], *, align: bool
) -> tuple[xr.DataArray, ...]:
    """Inner-join (``align=True``) or strict-check (``align=False``) coords."""
    if len(arrays) < 2:
        return arrays
    if align:
        return tuple(xr.align(*arrays, join="inner"))
    _assert_coords_consistent(arrays)
    return arrays


def _assert_coords_consistent(arrays: tuple[xr.DataArray, ...]) -> None:
    seen: dict[Any, xr.DataArray] = {}
    for arr in arrays:
        for dim in arr.dims:
            if dim not in arr.coords or arr.coords[dim].ndim != 1:
                continue
            coord = arr.coords[dim]
            if dim in seen:
                if not seen[dim].equals(coord):
                    raise CoordMismatch(
                        f"Shared dim {dim!r} has mismatched coords across "
                        "inputs. Pass align=True to inner-join, or align the "
                        "inputs yourself."
                    )
            else:
                seen[dim] = coord


def _resolve_reducer(op: Callable[..., Any] | str, sample: Any) -> Callable[..., Any]:
    import einx

    if isinstance(op, str):
        if op in _NATIVE_REDUCERS:
            return getattr(einx, op)
        if op == "median":
            fn: Callable[..., Any] = np.median
        else:
            allowed = [*sorted(_NATIVE_REDUCERS), "median"]
            raise PatternError(
                f"Unknown reduce op {op!r}. Use one of {allowed} or pass a callable."
            )
    else:
        fn = op
    return _backend_module(sample).adapt_numpylike_reduce(fn)


def _backend_module(array: Any) -> Any:
    """Pick the einx backend submodule matching the array's library."""
    module = type(array).__module__
    if module.startswith("jax"):
        import einx.jax as backend
    elif module.startswith("torch"):
        import einx.torch as backend
    else:
        import einx.numpy as backend
    return backend


def _wrap(
    result: Any,
    parsed: EinxPattern,
    inputs: list[xr.DataArray],
    coords: Mapping[str, Any] | None,
) -> xr.DataArray:
    out_dims = parsed.output_dims
    out_coords = forward_coords(inputs, out_dims, extra=coords)
    name = inputs[0].name if len(inputs) == 1 else None
    return xr.DataArray(result, dims=out_dims, coords=out_coords, name=name)


__all__ = ["einsum", "rearrange", "reduce", "repeat"]
