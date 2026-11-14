"""Packing xarray objects into one sample table and back (geostack OP-O3).

A ``DataArray`` or every data variable of a ``Dataset`` is flattened with
`pipekit_array.to_rows` over the same sample dims, and the blocks are put
side by side as the columns of one table: a `pipekit_array.DimArray` with
dims ``(SAMPLE, FEATURE)``. `pipekit_array.Estimator` fits and applies on
that table; the layout recorded here rebuilds xarray objects from what it
returns:

- one value per sample: a ``DataArray`` over the sample dims;
- the fitted features (a scaler, ``inverse_transform``): the input's
  variables on their own grids, as a ``DataArray`` or a ``Dataset``;
- a new output dim (PCA scores, class probabilities): one ``DataArray``
  with that dim where the first variable's feature dims were.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence
from typing import Any, cast

import numpy as np
import xarray as xr
from pipekit_array import DimArray, to_rows


#: Dim names of the packed table; private, so no user dim can collide.
SAMPLE = "__sample__"
FEATURE = "__feature__"


@dataclasses.dataclass(frozen=True, eq=False)
class Variable:
    """One variable's place in the table: its feature grid and metadata."""

    name: Hashable
    dims: tuple[Hashable, ...]
    feature_dims: tuple[Hashable, ...]
    feature_sizes: tuple[int, ...]
    coords: dict[Hashable, xr.DataArray]
    attrs: dict[Hashable, Any]

    @property
    def n_features(self) -> int:
        """The number of table columns this variable takes."""
        return int(np.prod(self.feature_sizes, dtype=np.int64))


@dataclasses.dataclass(frozen=True, eq=False)
class Layout:
    """The fit-time layout: the variables, and whether the input was a DataArray."""

    variables: tuple[Variable, ...]
    is_array: bool
    attrs: dict[Hashable, Any] = dataclasses.field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """The table's width."""
        return sum(v.n_features for v in self.variables)


@dataclasses.dataclass(frozen=True, eq=False)
class Samples:
    """The sample dims of one call's input, in table-row order.

    ``coords`` holds that input's coordinates on the sample dims and its
    scalar coordinates (a selected ``member`` or level), so outputs carry the
    labels of the data passed in, not the fit-time ones.
    """

    dims: tuple[Hashable, ...]
    sizes: tuple[int, ...]
    coords: dict[Hashable, xr.DataArray]
    #: the input Dataset's own attrs (``None`` for a DataArray input)
    attrs: dict[Hashable, Any] | None = None


def _items(
    obj: Any, fitted: Layout | None
) -> tuple[list[tuple[Hashable, xr.DataArray]], bool]:
    if isinstance(obj, xr.DataArray):
        if fitted is not None and not fitted.is_array:
            raise TypeError("XarrayEstimator was fitted on a Dataset; pass a Dataset.")
        return [(obj.name, obj)], True
    if isinstance(obj, xr.Dataset):
        if fitted is not None and fitted.is_array:
            raise TypeError(
                "XarrayEstimator was fitted on a DataArray; pass a DataArray."
            )
        if fitted is None:
            if not obj.data_vars:
                raise ValueError("XarrayEstimator: the Dataset has no data variables.")
            return list(obj.data_vars.items()), False
        names = [v.name for v in fitted.variables]
        if set(names) != set(obj.data_vars):
            raise ValueError(
                f"XarrayEstimator was fitted on variables {names}, "
                f"got {list(obj.data_vars)}."
            )
        return [(name, obj[name]) for name in names], False
    raise TypeError(
        "XarrayEstimator takes a DataArray, Dataset or DataTree; "
        f"got {type(obj).__name__}."
    )


def _features(
    name: Hashable,
    da: xr.DataArray,
    sample_dims: Sequence[Hashable],
    ref: Variable | None,
) -> tuple[Hashable, ...]:
    """The variable's feature dims, checked against the fit-time grid."""
    absent = [d for d in sample_dims if d not in da.dims]
    if absent:
        raise ValueError(
            f"XarrayEstimator: variable {name!r} has dims {tuple(da.dims)}, "
            f"without the sample dim(s) {absent}."
        )
    if ref is None:
        return tuple(d for d in da.dims if d not in sample_dims)
    if (
        set(da.dims) != set(sample_dims) | set(ref.feature_dims)
        or tuple(da.sizes[d] for d in ref.feature_dims) != ref.feature_sizes
    ):
        want = dict(zip(ref.feature_dims, ref.feature_sizes, strict=True))
        got = {d: n for d, n in da.sizes.items() if d not in sample_dims}
        raise ValueError(
            f"XarrayEstimator was fitted on features {want} for {name!r}, got {got}."
        )
    return ref.feature_dims


def _align(da: xr.DataArray, dim: Hashable, want: Any, what: str) -> xr.DataArray:
    """``da`` with ``dim`` in the order of the labels ``want``, matched by label.

    Raises:
        ValueError: ``da``'s labels on ``dim`` are not a permutation of ``want``.
    """
    have = da.indexes.get(dim)
    if have is None or have.equals(want):
        return da
    pos = have.get_indexer(want)
    if have.has_duplicates or len(have) != len(want) or (pos < 0).any():
        raise ValueError(
            f"XarrayEstimator: {what} has other {dim!r} labels than the fit-time "
            "ones; pass the same labels, in any order."
        )
    return da.isel({dim: pos})


def _check_auxiliary(da: xr.DataArray, ref: Variable, name: Hashable) -> None:
    """Reject a grid whose auxiliary feature coordinates differ from the fit's.

    Two curvilinear grids can share their ``y`` / ``x`` indexes and sizes but
    not ``nav_lat(y, x)``; the columns would then mean other places.
    """
    for k, want in ref.coords.items():
        if k in want.dims or k not in da.coords:
            continue
        have = da.coords[k]
        if set(have.dims) != set(want.dims) or not np.array_equal(
            have.transpose(*want.dims).values, want.values, equal_nan=_has_nan(want)
        ):
            raise ValueError(
                f"XarrayEstimator: variable {name!r} has another {k!r} coordinate "
                "than the fit-time grid."
            )


def _has_nan(c: xr.DataArray) -> bool:
    return np.issubdtype(c.dtype, np.floating) or np.issubdtype(
        c.dtype, np.complexfloating
    )


def _names(dims: Sequence[Hashable]) -> Sequence[str]:
    """xarray dim names for `pipekit_array.to_rows`, which types them ``str``."""
    return cast("Sequence[str]", tuple(dims))


def _concat(blocks: list[Any], axis: int) -> Any:
    if len(blocks) == 1:
        return blocks[0]
    if any(hasattr(b, "dask") for b in blocks):
        import dask.array as dsa

        return dsa.concatenate(blocks, axis=axis)
    return np.concatenate([np.asarray(b) for b in blocks], axis=axis)


def pack(
    obj: xr.DataArray | xr.Dataset,
    sample_dims: Sequence[Hashable],
    fitted: Layout | None = None,
) -> tuple[DimArray, Layout, Samples]:
    """``obj`` as one ``(SAMPLE, FEATURE)`` table, with its layout and samples.

    Args:
        obj: A DataArray, or a Dataset whose every variable has the sample dims.
        sample_dims: The dims whose entries are samples.
        fitted: The fit-time layout; given, ``obj`` must have its variables
            and feature grids (feature dims and coordinate labels in any order).

    Returns:
        The table, the layout (``fitted`` when given) and the samples, whose
        dims follow the first variable's order.

    Raises:
        TypeError: ``obj`` is not a DataArray or Dataset, or not the kind fitted.
        ValueError: A variable lacks a sample dim, or does not match ``fitted``.
    """
    items, is_array = _items(obj, fitted)
    first = items[0][1]
    for name, da in items:
        _features(name, da, sample_dims, None)
    order = tuple(d for d in first.dims if d in sample_dims)
    blocks, variables = [], []
    for i, (name, da) in enumerate(items):
        ref = None if fitted is None else fitted.variables[i]
        features = _features(name, da, sample_dims, ref)
        if ref is not None:
            # match feature columns by label, not position (permuted coordinates)
            for d in ref.feature_dims:
                if d in ref.coords and d in da.indexes:
                    da = _align(da, d, ref.coords[d].to_index(), f"variable {name!r}")
            _check_auxiliary(da, ref, name)
        rows, _ = to_rows(da.transpose(*order, *features), _names(order))
        blocks.append(rows)
        variables.append(
            Variable(
                name=name,
                dims=tuple(da.dims),
                feature_dims=features,
                feature_sizes=tuple(da.sizes[d] for d in features),
                coords={
                    k: c
                    for k, c in da.coords.items()
                    if c.dims and set(c.dims) <= set(features)
                },
                attrs=dict(da.attrs),
            )
        )
    attrs = None if is_array else dict(obj.attrs)
    layout = fitted or Layout(tuple(variables), is_array, attrs or {})
    samples = Samples(
        dims=order,
        sizes=tuple(first.sizes[d] for d in order),
        coords={k: c for k, c in first.coords.items() if set(c.dims) <= set(order)},
        attrs=attrs,
    )
    return DimArray(_concat(blocks, axis=1), (SAMPLE, FEATURE)), layout, samples


def pack_target(y: Any, samples: Samples) -> DimArray:
    """A target ``DataArray`` as one value (or one row) per table row.

    Raises:
        TypeError: ``y`` is not a DataArray.
        ValueError: ``y`` lacks a sample dim, has more than one other dim,
            or its sample sizes or labels differ from the input's. Labels
            in another order are matched to the input's.
    """
    if not isinstance(y, xr.DataArray):
        raise TypeError(
            f"XarrayEstimator: y must be a DataArray; got {type(y).__name__}."
        )
    absent = [d for d in samples.dims if d not in y.dims]
    rest = [d for d in y.dims if d not in samples.dims]
    if absent or len(rest) > 1:
        raise ValueError(
            f"XarrayEstimator: y has dims {tuple(y.dims)}; it needs the sample dims "
            f"{samples.dims} and at most one target dim."
        )
    if tuple(y.sizes[d] for d in samples.dims) != samples.sizes:
        raise ValueError("XarrayEstimator: y's sample sizes differ from the input's.")
    for d in samples.dims:
        # match targets to samples by label, not position
        if d in samples.coords and d in y.indexes:
            y = _align(y, d, samples.coords[d].to_index(), "y")
    rows, _ = to_rows(y.transpose(*samples.dims, *rest), _names(samples.dims))
    return (
        DimArray(rows, (SAMPLE, "__target__"))
        if rest
        else DimArray(rows[:, 0], (SAMPLE,))
    )


def concat_tables(tables: list[DimArray]) -> DimArray:
    """Tables (or targets) stacked along their rows, for a pooled fit."""
    return DimArray(_concat([t.data for t in tables], axis=0), tables[0].dims)


def _variable(block: Any, var: Variable, samples: Samples) -> xr.DataArray:
    data = np.reshape(np.asarray(block), (*samples.sizes, *var.feature_sizes))
    da = xr.DataArray(
        data,
        dims=(*samples.dims, *var.feature_dims),
        coords={**samples.coords, **var.coords},
        name=var.name,
        attrs=var.attrs,
    )
    return da.transpose(*var.dims)


def unpack(
    out: DimArray, layout: Layout, samples: Samples
) -> xr.DataArray | xr.Dataset:
    """Rebuild xarray objects from `pipekit_array.Estimator`'s output table.

    Args:
        out: ``(SAMPLE,)``, ``(SAMPLE, FEATURE)`` or ``(SAMPLE, new_dim)``.
        layout: The fit-time layout.
        samples: The samples of the input the table came from.

    Returns:
        A DataArray over the sample dims; the input's variables (a DataArray
        or a Dataset); or one DataArray with ``new_dim``.
    """
    data = np.asarray(out.data)
    if out.dims == (SAMPLE,):
        return xr.DataArray(
            data.reshape(samples.sizes), dims=samples.dims, coords=samples.coords
        )
    if out.dims == (SAMPLE, FEATURE):
        cuts = np.cumsum([v.n_features for v in layout.variables])[:-1]
        arrays = [
            _variable(block, var, samples)
            for block, var in zip(
                np.split(data, cuts, axis=1), layout.variables, strict=True
            )
        ]
        if layout.is_array:
            return arrays[0]
        attrs = layout.attrs if samples.attrs is None else samples.attrs
        return xr.Dataset({da.name: da for da in arrays}, attrs=attrs)
    new = out.dims[1]
    da = xr.DataArray(
        data.reshape(*samples.sizes, data.shape[1]),
        dims=(*samples.dims, new),
        coords=samples.coords,
    )
    first = layout.variables[0]
    order: list[Hashable] = []
    for d in first.dims:
        if d not in first.feature_dims:
            order.append(d)
        elif new not in order:
            order.append(new)
    if new not in order:
        order.append(new)
    return da.transpose(*order)


__all__ = [
    "FEATURE",
    "SAMPLE",
    "Layout",
    "Samples",
    "Variable",
    "concat_tables",
    "pack",
    "pack_target",
    "unpack",
]
