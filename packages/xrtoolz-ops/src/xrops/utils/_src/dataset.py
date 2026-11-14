"""``pack_dataset`` / ``unpack_dataset`` — multi-variable convenience.

Both are pure xarray. They turn a Dataset into the one ``DataArray`` that a
carrier-agnostic operator (`pipekit_array.einx`, a `pipekit_array.KernelOp`)
takes, with a ``variable`` dim, and back.

Per design decision D6, these are plain functions — not ``Operator``
subclasses — so pipelines are designed around a single packing point
rather than sprinkling pack/unpack into a chain.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


def pack_dataset(
    ds: xr.Dataset,
    variables: Sequence[str] | None = None,
    *,
    new_dim: str = "variable",
) -> xr.DataArray:
    """Stack named variables of a Dataset along a new dim.

    The variables must share dims and coords; the result is a single
    DataArray with one extra ``new_dim`` axis whose coordinate is the
    variable names.

    Args:
        ds: source Dataset.
        variables: variable names to stack, in order. Defaults to all
            data variables (sorted by insertion order).
        new_dim: name of the new stacking dim.

    Raises:
        ValueError: if no variables are selected or ``new_dim`` already
            exists as a dim on the selected variables.

    Each variable's attrs are kept on the ``new_dim`` coordinate (as
    ``variable_attrs``), so :func:`unpack_dataset` restores them; the packed
    array carries only the attrs every variable shares.

    Inverse of :func:`unpack_dataset`.
    """
    names = list(variables) if variables is not None else list(ds.data_vars)
    if not names:
        raise ValueError("pack_dataset: no variables to stack.")
    arrays = [ds[name] for name in names]
    first_dims = tuple(arrays[0].dims)
    for name, arr in zip(names, arrays, strict=True):
        if new_dim in arr.dims:
            raise ValueError(
                f"pack_dataset: new_dim {new_dim!r} already a dim of {name!r}."
            )
        if tuple(arr.dims) != first_dims:
            raise ValueError(
                f"pack_dataset: variable {name!r} has dims {tuple(arr.dims)} but "
                f"{names[0]!r} has {first_dims}; all variables must share dims."
            )
    # join="exact" rejects misaligned coords rather than silently NaN-filling
    # an outer-joined union grid, keeping pack/unpack lossless.
    stacked = xr.concat(arrays, dim=new_dim, join="exact")
    stacked = stacked.assign_coords({new_dim: np.array(names)})
    # concat keeps the first variable's attrs; keep each one's, for unpacking
    stacked[new_dim].attrs["variable_attrs"] = {
        name: dict(arr.attrs) for name, arr in zip(names, arrays, strict=True)
    }
    common = {
        k: v
        for k, v in arrays[0].attrs.items()
        if all(k in a.attrs and _same(a.attrs[k], v) for a in arrays)
    }
    stacked.attrs = common
    return stacked


def unpack_dataset(
    da: xr.DataArray,
    *,
    dim: str = "variable",
) -> xr.Dataset:
    """Split a 'variable'-style dim back into a Dataset.

    Args:
        da: DataArray with a ``dim`` axis whose coordinate is a sequence
            of variable-name strings.
        dim: the stacking dim to split on.

    Raises:
        ValueError: if ``dim`` is absent or carries no coordinate labels.

    Inverse of :func:`pack_dataset`.
    """
    if dim not in da.dims:
        raise ValueError(f"unpack_dataset: dim {dim!r} not on input {tuple(da.dims)}.")
    if dim not in da.coords:
        raise ValueError(
            f"unpack_dataset: dim {dim!r} has no coordinate of variable names."
        )
    names = [str(name) for name in da.coords[dim].values]
    attrs = da.coords[dim].attrs.get("variable_attrs", {})
    return xr.Dataset(
        {
            name: da.isel({dim: i})
            .drop_vars(dim)
            .assign_attrs(attrs.get(name, da.attrs))
            for i, name in enumerate(names)
        }
    )


def _same(a: object, b: object) -> bool:
    try:
        return bool(np.all(a == b))
    except (TypeError, ValueError):
        return False


__all__ = ["pack_dataset", "unpack_dataset"]
