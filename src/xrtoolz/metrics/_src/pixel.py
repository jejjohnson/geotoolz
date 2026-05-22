"""Pixel-level (pointwise) evaluation metrics.

Layer-0 primitives take ``DataArray`` inputs positionally (``pred``,
``ref``) and reduce over the requested ``dim``. They return a
``DataArray`` with the remaining dimensions.

Convention: positive ``bias`` means the prediction is larger than the
reference. Correlation is Pearson's ``r``.

Per the design refresh in ``docs/design/xarray-native-primitives.md``
(PR β), primitives are DataArray-positional. The Layer-1 operator
wrappers below carry the ``variable=`` selector so that pipelines
threading Datasets still see a uniform Dataset-in interface; the
selection happens in the operator ``_apply``, not in the primitive.

The pointwise math lives in the Tier A array kernels at
:mod:`xrtoolz.metrics._src.array_pixel`; the Tier B wrappers below
delegate to those kernels via :func:`xr.apply_ufunc`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.metrics._src import array_pixel
from xrtoolz.signature import Signature


Dim = str | Sequence[str]


def _normalize_dim(dim: Dim) -> list[str]:
    return [dim] if isinstance(dim, str) else list(dim)


def _apply_pixel_kernel(
    fn: Any,
    pred: xr.DataArray,
    ref: xr.DataArray,
    dim: Dim,
) -> xr.DataArray:
    """Run a Tier A pixel kernel on two DataArrays, reducing ``dim``.

    Dispatches to the array kernel via :func:`xr.apply_ufunc` with
    ``dim`` as the input core dimensions. The kernel sees a flattened
    trailing block of axes and reduces with
    ``axis=tuple(range(-len(core), 0))``.

    For dask-backed inputs, ``allow_rechunk=True`` is set so a core
    dimension that is split across multiple chunks is rechunked into a
    single chunk before the kernel runs. Without this, ``apply_ufunc``
    raises on multi-chunk core dims, which is a common case for the
    reduce dimension (e.g. ``time`` chunked monthly).

    Output dtype is promoted to at least ``float64`` so that integer
    inputs don't truncate floating-point reductions.
    """
    core = _normalize_dim(dim)
    out_dtype = np.result_type(pred.dtype, np.float64)
    out: xr.DataArray = xr.apply_ufunc(
        lambda p, r: fn(p, r, axis=tuple(range(-len(core), 0))),
        pred,
        ref,
        input_core_dims=[core, core],
        dask="parallelized",
        output_dtypes=[out_dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return out


# ---------- Layer-0 (xarray DataArray-positional) -------------------------


def mse(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Mean squared error reduced over ``dim``."""
    return _apply_pixel_kernel(array_pixel.mse, pred, ref, dim)


def rmse(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Root mean squared error reduced over ``dim``."""
    return _apply_pixel_kernel(array_pixel.rmse, pred, ref, dim)


def nrmse(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)``.

    Returns a score in ``(-inf, 1]`` where 1 means a perfect match and 0
    means the prediction is as wrong as a zero prediction.
    """
    return _apply_pixel_kernel(array_pixel.nrmse, pred, ref, dim)


def mae(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Mean absolute error reduced over ``dim``."""
    return _apply_pixel_kernel(array_pixel.mae, pred, ref, dim)


def bias(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dim``."""
    return _apply_pixel_kernel(array_pixel.bias, pred, ref, dim)


def correlation(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dim``."""
    return _apply_pixel_kernel(array_pixel.correlation, pred, ref, dim)


def r2_score(pred: xr.DataArray, ref: xr.DataArray, *, dim: Dim) -> xr.DataArray:
    """Coefficient of determination: ``1 - SS_res / SS_tot``."""
    return _apply_pixel_kernel(array_pixel.r2_score, pred, ref, dim)


# ---------- Layer-1 (Operator wrappers) -----------------------------------


class _PixelMetricOp(Operator):
    """Base class for two-input pixel metrics.

    The constructor carries the Dataset-level selector (``variable``)
    and the reduce-axis spec (``dims``). ``_apply`` selects the named
    variable from each input Dataset and forwards the resulting
    DataArrays to the Layer-0 primitive.
    """

    _fn: Any = None

    def __init__(self, variable: str, dims: Dim):
        self.variable = variable
        self.dims = dims if isinstance(dims, str) else list(dims)

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return self.__class__._fn(
            ds_pred[self.variable],
            ds_ref[self.variable],
            dim=self.dims,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": self.dims if isinstance(self.dims, str) else list(self.dims),
        }

    def compute_output_signature(
        self,
        input_signature: Signature | tuple[Signature, ...],
    ) -> Signature:
        if isinstance(input_signature, tuple):
            signature = input_signature[0]
            for other in input_signature[1:]:
                _validate_compatible_signatures(signature, other)
        else:
            signature = input_signature
        dims = (self.dims,) if isinstance(self.dims, str) else tuple(self.dims)
        # Pixel metric kernels promote int inputs to float64 (see
        # _apply_pixel_kernel above), so the inferred output dtype follows.
        return Signature(signature.drop_dims(dims).dims, dtype=np.float64)


class MSE(_PixelMetricOp):
    _fn = staticmethod(mse)


class RMSE(_PixelMetricOp):
    _fn = staticmethod(rmse)


class NRMSE(_PixelMetricOp):
    _fn = staticmethod(nrmse)


class MAE(_PixelMetricOp):
    _fn = staticmethod(mae)


class Bias(_PixelMetricOp):
    _fn = staticmethod(bias)


class Correlation(_PixelMetricOp):
    _fn = staticmethod(correlation)


class R2Score(_PixelMetricOp):
    _fn = staticmethod(r2_score)


def _validate_compatible_signatures(left: Signature, right: Signature) -> None:
    if set(left.dims) != set(right.dims):
        raise ValueError(
            "Metric inputs must have matching dimension names; "
            f"got {tuple(left.dims)} and {tuple(right.dims)}."
        )
    mismatched = {
        name: (left.dims[name], right.dims[name])
        for name in left.dims
        if left.dims[name] is not None
        and right.dims[name] is not None
        and left.dims[name] != right.dims[name]
    }
    if mismatched:
        raise ValueError(f"Metric input signature sizes do not match: {mismatched}.")


__all__ = [
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "Bias",
    "Correlation",
    "R2Score",
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "r2_score",
    "rmse",
]
