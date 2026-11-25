"""Empirical orthogonal functions of a field, weighted and with land."""

from __future__ import annotations

import numpy as np
import xarray as xr
from diagnostix import eofs as _eofs


def eof(
    da: xr.DataArray,
    sample_dim: str = "time",
    *,
    n_modes: int,
    weights: xr.DataArray | None = None,
) -> xr.Dataset:
    """Leading empirical orthogonal functions (EOFs) of a field.

    The thin singular value decomposition of the field without its mean along
    ``sample_dim``, with each cell times the root of its weight
    (``diagnostix.eofs.eofs``). Without weights it is the principal component
    analysis of the cells, as :func:`xrops.transforms.pca`, up to the signs; the
    loading of each mode that is largest in size is positive.

    The field is loaded: a decomposition reads all of it.

    Args:
        da: Field with ``sample_dim`` and at least one other dim, the grid.
        sample_dim: Dim of the samples.
        n_modes: Number of modes, at most the smaller of the samples and the
            cells with data.
        weights: Weight of each cell, 0 or more, on dims of the grid; the areas,
            ``cos(lat)`` on a lon–lat grid. 1 if not given.

    Returns:
        Dataset with ``modes`` (``mode``, grid), orthonormal under the weights;
        ``principal_components`` (``sample_dim``, ``mode``);
        ``explained_variance`` (``mode``), the share of each mode; the ``mean``
        that was removed; and the ``weights``. A cell without any data, such as
        land, is NaN in the modes and the mean.

    Raises:
        ValueError: If ``sample_dim`` is missing or the only dim, ``n_modes`` is
            out of range, the weights are not on the grid, or a cell has data at
            some samples only.

    Example:
        Area-weighted modes of the anomalies of a sea-surface temperature::

            modes = eof(sst, n_modes=3, weights=np.cos(np.deg2rad(sst.lat)))
            sst_3 = eof_reconstruct(modes)
    """
    grid = _grid_dims(da, sample_dim)
    field = da.transpose(sample_dim, *grid).load()
    missing = np.isnan(field.values)
    valid = ~missing.all(axis=0)
    gaps = int((missing.any(axis=0) & valid).sum())
    if gaps:
        raise ValueError(
            f"{gaps} cell(s) have data at some samples only; fill the gaps, or mask "
            "the cells, before the decomposition."
        )
    mean = field.mean(sample_dim, skipna=False)
    weight = _on_grid(weights, mean)
    out = _eofs.eofs(
        (field - mean).data, n_modes=n_modes, weights=weight.data, valid=valid
    )
    mode = {"mode": np.arange(n_modes)}
    over_grid = {name: c for name, c in mean.coords.items() if name in grid}
    samples = {sample_dim: field[sample_dim]} if sample_dim in field.coords else {}
    return xr.Dataset(
        {
            "modes": (("mode", *grid), out.modes),
            "principal_components": ((sample_dim, "mode"), out.principal_components),
            "explained_variance": (("mode",), out.explained_variance),
            "mean": mean.rename(None),
            "weights": weight,
        },
        coords={**mode, **over_grid, **samples},
    )


def eof_project(da: xr.DataArray, eofs: xr.Dataset) -> xr.DataArray:
    """The amplitude of each mode in a field: ``Σ w (x − mean) e`` over the cells.

    For the field the modes came from, these are its principal components
    (``diagnostix.eofs.project``).

    Args:
        da: Field on the grid of the modes, with any other dims. A NaN in a cell
            that the modes have gives NaN.
        eofs: Output of :func:`eof`.

    Returns:
        The amplitudes, with ``mode`` in place of the grid.
    """
    grid = list(eofs["mean"].dims)
    anomaly = da - eofs["mean"]
    record = _record(eofs)
    valid = ~np.isnan(eofs["mean"].values)
    weights = eofs["weights"].values
    out = xr.apply_ufunc(
        lambda x: _eofs.project(x, record, weights=weights, valid=valid),
        anomaly,
        input_core_dims=[grid],
        output_core_dims=[["mode"]],
        dask="parallelized",
        keep_attrs=False,
        output_dtypes=[anomaly.dtype],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {"mode": eofs.sizes["mode"]},
        },
    )
    return out.assign_coords(mode=eofs["mode"])


def eof_reconstruct(
    eofs: xr.Dataset, principal_components: xr.DataArray | None = None
) -> xr.DataArray:
    """The field that some amplitudes of the modes add up to: ``mean + Σ a e``.

    Args:
        eofs: Output of :func:`eof`.
        principal_components: Amplitudes on ``mode`` and any other dims; those
            of ``eofs`` if not given. A mode set to 0 is left out.

    Returns:
        The field, with the grid in place of ``mode``, NaN where the modes are
        (``diagnostix.eofs.reconstruct``).
    """
    if principal_components is None:
        principal_components = eofs["principal_components"]
    grid = list(eofs["mean"].dims)
    record = _record(eofs)
    out = xr.apply_ufunc(
        lambda a: _eofs.reconstruct(record, a),
        principal_components,
        input_core_dims=[["mode"]],
        output_core_dims=[grid],
        dask="parallelized",
        keep_attrs=False,
        output_dtypes=[principal_components.dtype],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {d: eofs.sizes[d] for d in grid},
        },
    )
    return out + eofs["mean"]


def _grid_dims(da: xr.DataArray, sample_dim: str) -> list[str]:
    if sample_dim not in da.dims or da.ndim < 2:
        raise ValueError(
            f"eof needs {sample_dim!r} and at least one other dim; got {da.dims!r}."
        )
    return [str(d) for d in da.dims if d != sample_dim]


def _on_grid(weights: xr.DataArray | None, mean: xr.DataArray) -> xr.DataArray:
    """The weights on the grid of the mean, 1 if there are none."""
    ones = xr.ones_like(mean, dtype=mean.dtype).rename(None).drop_attrs(deep=False)
    if weights is None:
        return ones
    extra = [d for d in weights.dims if d not in mean.dims]
    if extra:
        raise ValueError(f"the weights have dims {extra!r} that the grid has not.")
    return (ones * weights).transpose(*mean.dims).astype(mean.dtype)


def _record(eofs: xr.Dataset) -> _eofs.EOFs:
    grid = eofs["mean"].dims
    return _eofs.EOFs(
        modes=eofs["modes"].transpose("mode", *grid).values,
        principal_components=eofs["principal_components"].values,
        explained_variance=eofs["explained_variance"].values,
    )


__all__ = ["eof", "eof_project", "eof_reconstruct"]
