"""Coordinate-axis remapping on xarray Datasets.

The generic primitive is :func:`remap_axis`: given a source dimension and
a 1D target coordinate vector, every numeric variable that carries the
source dim is interpolated onto the target axis along that dim.

Vertical / temporal presets (``ToSigma``, ``FromSigma``,
``ToIsopycnal``, ``ToPressureLevels``, ``ToHeight``, ``ToPhase``) live
as :class:`Operator` subclasses in :mod:`xrtoolz.interpolate.operators`;
they parametrise the generic operator with conventional dim/coord names.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xrtoolz.transforms._src import array_coord_remap as _array
from xrtoolz.utils._src.finite import _finite_mask


def remap_axis(
    ds: xr.Dataset,
    *,
    source_dim: str,
    target_coords: xr.DataArray | np.ndarray,
    target_name: str | None = None,
    method: str = "linear",
) -> xr.Dataset:
    """Remap every variable carrying ``source_dim`` onto ``target_coords``.

    If ``target_coords`` is a :class:`xr.DataArray` and ``target_name``
    is None, the new dim name is taken from ``target_coords.name``;
    otherwise ``target_name`` (or ``source_dim``) is used.

    Targets outside the source range produce NaN (no extrapolation).
    """
    if source_dim not in ds.dims:
        raise ValueError(
            f"source_dim {source_dim!r} not in Dataset dims {tuple(ds.dims)}"
        )
    if source_dim not in ds.coords:
        raise ValueError(
            f"Dataset must carry a coordinate named {source_dim!r} "
            "for the source axis values"
        )

    if isinstance(target_coords, xr.DataArray):
        new_name = target_name or target_coords.name or source_dim
        tgt = np.asarray(target_coords.values, dtype=float)
    else:
        new_name = target_name or source_dim
        tgt = np.asarray(target_coords, dtype=float)

    src = np.asarray(ds[source_dim].values, dtype=float)

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if source_dim not in da.dims:
            # Variable doesn't depend on the source axis — pass through.
            out_vars[str(name)] = da
            continue
        if not np.issubdtype(da.dtype, np.number):
            # A non-numeric variable that depends on source_dim cannot be
            # interpolated; passing it through would leave the output with a
            # phantom source_dim axis (and possibly incompatible sizes).
            raise TypeError(
                f"variable {name!r} carries source_dim {source_dim!r} but has "
                f"non-numeric dtype {da.dtype}; drop or convert it before "
                "calling remap_axis"
            )
        axis = da.get_axis_num(source_dim)
        new_values = _array.remap_axis(
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
        out_vars[str(name)] = xr.DataArray(
            new_values,
            dims=new_dims,
            coords=new_coords,
            attrs=dict(da.attrs),
            name=da.name,
        )

    base_coords = {
        cname: c
        for cname, c in ds.coords.items()
        if source_dim not in c.dims and cname != source_dim
    }
    base_coords[new_name] = xr.DataArray(tgt, dims=(new_name,), name=new_name)
    return xr.Dataset(out_vars, coords=base_coords, attrs=dict(ds.attrs))


def to_phase(
    ds: xr.Dataset,
    *,
    time_dim: str,
    period: float,
    n_bins: int,
    epoch: float = 0.0,
) -> xr.Dataset:
    """Fold ``time_dim`` onto a phase axis by binning + averaging.

    The time coordinate must be numeric in the same units as ``period``.
    Phase is computed as ``((t - epoch) / period) mod 1`` and binned
    into ``n_bins`` evenly-spaced bins on ``[0, 1)``. Output carries
    a ``"phase"`` dim with coordinate values at bin centers.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"time_dim {time_dim!r} not in Dataset dims {tuple(ds.dims)}")
    if time_dim not in ds.coords:
        raise ValueError(
            f"Dataset must carry a coordinate named {time_dim!r} to compute phase"
        )
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    t = np.asarray(ds[time_dim].values, dtype=float)
    finite_t = _finite_mask(t)
    phase = np.where(finite_t, ((t - epoch) / period) % 1.0, 0.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, n_bins - 1)

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if time_dim not in da.dims:
            out_vars[str(name)] = da
            continue
        if not np.issubdtype(da.dtype, np.number):
            raise TypeError(
                f"variable {name!r} carries time_dim {time_dim!r} but has "
                f"non-numeric dtype {da.dtype}; drop or convert it before "
                "calling to_phase"
            )
        axis = da.get_axis_num(time_dim)
        moved = np.moveaxis(da.values, axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        # Use a complex accumulator if the data is complex so the
        # imaginary part isn't dropped (P1 review).
        is_complex = np.iscomplexobj(flat)
        acc_dtype = np.complex128 if is_complex else float
        sums = np.zeros((n_bins, flat.shape[1]), dtype=acc_dtype)
        counts = np.zeros((n_bins, flat.shape[1]), dtype=float)
        # A sample is valid only if its time coord is finite AND every
        # data value is finite. Excluding NaN-time rows keeps stale rows
        # out of the phase means (P2 review).
        valid_value = (
            ~np.isnan(flat)
            if not is_complex
            else (~np.isnan(flat.real) & ~np.isnan(flat.imag))
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
            mean = np.where(
                counts > 0, sums / np.where(counts > 0, counts, 1.0), nan_fill
            )
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
        out_vars[str(name)] = xr.DataArray(
            new_values,
            dims=new_dims,
            coords=new_coords,
            attrs=dict(da.attrs),
            name=da.name,
        )

    base_coords = {
        cname: c
        for cname, c in ds.coords.items()
        if time_dim not in c.dims and cname != time_dim
    }
    base_coords["phase"] = xr.DataArray(centers, dims=("phase",), name="phase")
    return xr.Dataset(out_vars, coords=base_coords, attrs=dict(ds.attrs))


__all__ = ["remap_axis", "to_phase"]
