"""Coordinate-axis remapping on xarray DataArrays.

Per the PR γ primitive-flip (``docs/design/xarray-native-primitives.md``),
the Layer-0 primitives in this module are DataArray-in / DataArray-out:
one variable goes in, one variable comes out. The Dataset selection /
per-variable loop lives in the Layer-1 ``Operator`` wrappers at
:mod:`xrtoolz.interpolate.operators` (``RemapAxis``, ``ToPhase``, and
the vertical presets).

The generic primitive is :func:`remap_axis`: given a source dimension
and a 1D target coordinate vector, the DataArray is interpolated onto
the target axis along that dim. With ``source_coords`` the source axis
values may vary per column (terrain-following / hybrid grids), and
``extrapolate`` decides what happens beyond each column's range.
:func:`to_phase` folds a time axis onto a phase axis by binning +
averaging.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr

from xrtoolz.transforms._src import _coord_remap_kernels as _kernels
from xrtoolz.transforms._src._coord_remap_kernels import Extrapolate
from xrtoolz.utils._src.finite import _finite_mask


_TARGET_TMP_DIM = "__remap_axis_target__"


def remap_axis(
    da: xr.DataArray,
    *,
    source_dim: str,
    target_coords: xr.DataArray | np.ndarray,
    target_name: str | None = None,
    method: str = "linear",
    source_coords: str | xr.DataArray | None = None,
    extrapolate: Extrapolate = "nan",
) -> xr.DataArray:
    """Remap ``da`` along ``source_dim`` onto ``target_coords``.

    If ``target_coords`` is a :class:`xr.DataArray` and ``target_name``
    is None, the new dim name is taken from ``target_coords.name``;
    otherwise ``target_name`` (or ``source_dim``) is used.

    Every coordinate that depends on ``source_dim`` is dropped from the
    output; the new target coordinate is attached under the new name.
    The Layer-1 ``RemapAxis`` operator handles Dataset selection /
    per-variable looping.

    Args:
        da: Numeric DataArray carrying ``source_dim``.
        source_dim: Name of the dimension being replaced.
        target_coords: 1-D target axis values.
        target_name: Explicit name for the new dimension.
        method: ``"linear"`` or ``"nearest"``.
        source_coords: Source axis *values*. ``None`` (default) uses the
            1-D dimension coordinate ``da[source_dim]``. A coordinate name
            on ``da`` or a DataArray broadcastable against ``da`` supplies
            per-column levels — it must carry ``source_dim`` and may vary
            along any of the other dims (e.g. ``z_agl(time, level, y, x)``
            on a terrain-following grid). Every column must be strictly
            monotone along ``source_dim``.
        extrapolate: ``"nan"`` (default) — targets outside a column's
            range become ``NaN``; ``"nearest"`` — they hold that column's
            end value.

    Returns:
        ``da`` with ``source_dim`` replaced by the target axis. When
        ``source_coords`` is given, chunked (dask) inputs stay lazy as
        long as ``source_dim`` sits in a single chunk.

    Raises:
        ValueError: If ``source_dim`` is missing, ``source_coords`` does
            not carry it, a column is not monotone, or ``extrapolate`` /
            ``method`` is unknown.
        TypeError: If ``da`` is not numeric.

    Example:
        Remap a model-level field with a 4-D height-above-ground
        coordinate onto fixed heights, holding the lowest / highest model
        level beyond each column's range:

        ```pycon
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.transforms import remap_axis
        >>> level = np.arange(4)
        >>> z_agl = 20.0 * (level[None, :, None, None] + 1) + np.zeros((2, 4, 3, 3))
        >>> z_agl[..., 0, 0] += 15.0  # one column starts higher up
        >>> t = xr.DataArray(
        ...     300.0 - 0.01 * z_agl,
        ...     dims=("time", "level", "y", "x"),
        ...     coords={"z_agl": (("time", "level", "y", "x"), z_agl)},
        ... )
        >>> out = remap_axis(
        ...     t,
        ...     source_dim="level",
        ...     target_coords=np.array([10.0, 50.0, 90.0]),
        ...     target_name="height",
        ...     source_coords="z_agl",
        ...     extrapolate="nearest",
        ... )
        >>> out.dims, out.sizes["height"]
        (('time', 'height', 'y', 'x'), 3)
        >>> float(out.isel(time=0, height=1, y=1, x=1))  # in range: linear
        299.5

        ```
    """
    if source_dim not in da.dims:
        raise ValueError(
            f"source_dim {source_dim!r} not in DataArray dims {tuple(da.dims)}"
        )
    if extrapolate not in _kernels._EXTRAPOLATE_MODES:
        raise ValueError(
            f"unknown extrapolate {extrapolate!r}; expected one of "
            f"{_kernels._EXTRAPOLATE_MODES!r}"
        )
    if source_coords is None and source_dim not in da.coords:
        raise ValueError(
            f"DataArray must carry a coordinate named {source_dim!r} "
            "for the source axis values"
        )
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(
            f"remap_axis requires numeric data; got dtype {da.dtype}. "
            "Drop or convert the variable before calling remap_axis."
        )

    if isinstance(target_coords, xr.DataArray):
        new_name = target_name or target_coords.name or source_dim
        tgt = np.asarray(target_coords.values, dtype=float)
    else:
        new_name = target_name or source_dim
        tgt = np.asarray(target_coords, dtype=float)

    if source_coords is not None or extrapolate != "nan":
        return _remap_axis_columns(
            da,
            source_dim=source_dim,
            tgt=tgt,
            new_name=str(new_name),
            method=method,
            source_coords=source_coords,
            extrapolate=extrapolate,
        )

    src = np.asarray(da[source_dim].values, dtype=float)

    axis = da.get_axis_num(source_dim)
    new_values = _kernels.remap_axis(
        da.values,
        axis=axis,
        source_coords=src,
        target_coords=tgt,
        method=method,
    )
    new_dims = tuple(new_name if d == source_dim else d for d in da.dims)
    new_coords = {
        cname: c
        for cname, c in da.coords.items()
        if source_dim not in c.dims and cname != source_dim
    }
    new_coords[new_name] = xr.DataArray(tgt, dims=(new_name,), name=new_name)
    return xr.DataArray(
        new_values,
        dims=new_dims,
        coords=new_coords,
        attrs=dict(da.attrs),
        name=da.name,
    )


def _remap_axis_columns(
    da: xr.DataArray,
    *,
    source_dim: str,
    tgt: np.ndarray,
    new_name: str,
    method: str,
    source_coords: str | xr.DataArray | None,
    extrapolate: Extrapolate,
) -> xr.DataArray:
    """Per-column path of :func:`remap_axis` via ``apply_ufunc``."""
    if source_coords is None:
        src_da = da[source_dim]
    elif isinstance(source_coords, str):
        if source_coords not in da.coords:
            raise ValueError(
                f"source_coords {source_coords!r} is not a coordinate on the "
                f"DataArray; got coords {tuple(da.coords)}"
            )
        src_da = da.coords[source_coords]
    else:
        src_da = source_coords
    if source_dim not in src_da.dims:
        raise ValueError(
            f"source_coords must carry source_dim {source_dim!r}; "
            f"got dims {tuple(src_da.dims)}"
        )
    extra = [d for d in src_da.dims if d not in da.dims]
    if extra:
        raise ValueError(
            f"source_coords has dims {extra!r} that are not on the DataArray "
            f"(dims {tuple(da.dims)})"
        )

    # Coordinates riding on ``source_dim`` describe the axis being replaced.
    da_in = da.drop_vars([c for c in da.coords if source_dim in da[c].dims])
    # ``src_da`` may itself be one of its own coords (a dimension coordinate
    # or a named coordinate pulled off ``da``); never drop the array itself.
    src_in = src_da.drop_vars(
        [c for c in src_da.coords if c != src_da.name and source_dim in src_da[c].dims]
    )
    out_dtype = np.complex128 if np.iscomplexobj(da.data) else np.float64

    def _kernel(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
        values, coords = np.broadcast_arrays(values, coords)
        return _kernels.remap_axis_columns(
            values,
            axis=-1,
            source_coords=coords,
            target_coords=tgt,
            method=method,
            extrapolate=extrapolate,
        )

    out = xr.apply_ufunc(
        _kernel,
        da_in,
        src_in,
        input_core_dims=[[source_dim], [source_dim]],
        output_core_dims=[[_TARGET_TMP_DIM]],
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {_TARGET_TMP_DIM: tgt.size},
            "allow_rechunk": False,
        },
    )
    out = out.rename({_TARGET_TMP_DIM: new_name})
    out = out.assign_coords({new_name: (new_name, tgt)})
    order = tuple(new_name if d == source_dim else d for d in da.dims)
    out = out.transpose(*order)
    out.name = da.name
    out.attrs = dict(da.attrs)
    return out


def to_phase(
    da: xr.DataArray,
    *,
    time_dim: str,
    period: float,
    n_bins: int,
    epoch: float = 0.0,
) -> xr.DataArray:
    """Fold ``time_dim`` onto a phase axis by binning + averaging.

    The time coordinate must be numeric in the same units as ``period``.
    Phase is computed as ``((t - epoch) / period) mod 1`` and binned
    into ``n_bins`` evenly-spaced bins on ``[0, 1)``. Output carries
    a ``"phase"`` dim with coordinate values at bin centers.

    The Layer-1 ``ToPhase`` operator handles Dataset selection /
    per-variable looping.
    """
    if time_dim not in da.dims:
        raise ValueError(
            f"time_dim {time_dim!r} not in DataArray dims {tuple(da.dims)}"
        )
    if time_dim not in da.coords:
        raise ValueError(
            f"DataArray must carry a coordinate named {time_dim!r} to compute phase"
        )
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(
            f"to_phase requires numeric data; got dtype {da.dtype}. "
            "Drop or convert the variable before calling to_phase."
        )

    t = np.asarray(da[time_dim].values, dtype=float)
    finite_t = _finite_mask(t)
    phase = np.where(finite_t, ((t - epoch) / period) % 1.0, 0.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, n_bins - 1)

    axis = da.get_axis_num(time_dim)
    moved = np.moveaxis(da.values, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    # Use a complex accumulator if the data is complex so the imaginary
    # part isn't dropped (P1 review).
    is_complex = np.iscomplexobj(flat)
    acc_dtype = np.complex128 if is_complex else float
    # cast: with dtype resolved at runtime (complex128 or float), the
    # specialised ndarray type numpy's stubs infer would reject the row
    # assignments below.
    sums = cast(np.ndarray, np.zeros((n_bins, flat.shape[1]), dtype=acc_dtype))
    counts = np.zeros((n_bins, flat.shape[1]), dtype=float)
    # A sample is valid only if its time coord is finite AND every data
    # value is finite. Excluding NaN-time rows keeps stale rows out of
    # the phase means (P2 review). ``np.isfinite`` handles integer
    # dtypes (treated as all-finite) where ``np.isnan`` would raise.
    valid_value = (
        np.isfinite(flat)
        if not is_complex
        else (np.isfinite(flat.real) & np.isfinite(flat.imag))
    )
    valid = finite_t[:, None] & valid_value
    for b in range(n_bins):
        m = (bin_idx == b) & finite_t
        if not m.any():
            continue
        sub = flat[m]
        sub_valid = valid[m]
        zero = acc_dtype(0)
        sums[b] = np.where(sub_valid, sub, zero).sum(axis=0)
        counts[b] = sub_valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        nan_fill = (np.nan + 0j) if is_complex else np.nan
        mean = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), nan_fill)
    out_shape = (n_bins, *moved.shape[1:])
    new_values = mean.reshape(out_shape)
    new_values = np.moveaxis(new_values, 0, axis)
    new_dims = tuple("phase" if d == time_dim else d for d in da.dims)
    new_coords = {
        cname: c
        for cname, c in da.coords.items()
        if time_dim not in c.dims and cname != time_dim
    }
    new_coords["phase"] = xr.DataArray(centers, dims=("phase",), name="phase")
    return xr.DataArray(
        new_values,
        dims=new_dims,
        coords=new_coords,
        attrs=dict(da.attrs),
        name=da.name,
    )


__all__ = ["Extrapolate", "remap_axis", "to_phase"]
