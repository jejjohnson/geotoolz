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

The scores are ``diagnostix.skill``'s. A cell where either field is missing is
left out of every moment of a score, so the moments are of the same cells.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr
from diagnostix import skill
from pipekit_ops import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator


Dim = str | Sequence[str]


def _normalize_dim(dim: Dim) -> list[str]:
    return [dim] if isinstance(dim, str) else list(dim)


def _score(
    function: Callable[..., Any],
    pred: xr.DataArray,
    ref: xr.DataArray,
    dim: Dim,
    weights: xr.DataArray | None,
    **options: Any,
) -> xr.DataArray:
    """A score of two DataArrays over the cells of ``dim`` that both have.

    On dask a reduced dim becomes one chunk (``allow_rechunk``), and the result
    is lazy. Integers become float64.
    """
    core = _normalize_dim(dim)
    fields = [pred, ref]
    if weights is not None:
        # on the dims that are pooled, whichever of them the weights have
        missing = [d for d in core if d not in weights.dims]
        shape = {d: pred.sizes[d] for d in missing}
        fields.append(weights.expand_dims(shape) if missing else weights)
    dtype = np.result_type(pred.dtype, ref.dtype, np.float32)

    def kernel(p: Any, r: Any, w: Any = None) -> Any:
        both = (p == p) & (r == r)
        axis = tuple(range(-len(core), 0))
        out = function(p, r, weights=w, valid=both, axis=axis, **options)
        return out.astype(dtype)

    return xr.apply_ufunc(
        kernel,
        *fields,
        input_core_dims=[core] * len(fields),
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


# ---------- Layer-0 (xarray DataArray-positional) -------------------------
#
# Each takes ``weights``, 0 or more, on any of the dims that are pooled: cell
# areas, or ``cos(lat)`` on a lon–lat grid. 1 if not given.


def mse(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Mean squared error reduced over ``dim``."""
    return rmse(pred, ref, dim=dim, weights=weights) ** 2


def rmse(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Root mean squared error reduced over ``dim``."""
    return _score(skill.rmse, pred, ref, dim, weights)


def nrmse(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Normalized RMSE: ``1 - RMSE / sqrt(<ref^2>)``.

    Returns a score in ``(-inf, 1]`` where 1 means a perfect match and 0
    means the prediction is as wrong as a zero prediction.
    """
    return _score(skill.nrmse_score, pred, ref, dim, weights, normalise="rms")


def nrmse_score(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Mercator/OceanBench-flavour NRMSE skill: ``1 - RMSE / std(ref)``.

    Differs from :func:`nrmse` in the denominator: this uses the
    anomaly magnitude ``std(ref)`` while :func:`nrmse` uses the raw
    signal magnitude ``sqrt(<ref^2>)``. For zero-mean references the
    two agree. Against a constant reference a perfect prediction scores 1 and
    any other ``-inf``.
    """
    return _score(skill.nrmse_score, pred, ref, dim, weights, normalise="std")


def mae(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Mean absolute error reduced over ``dim``."""
    return _score(skill.mae, pred, ref, dim, weights)


def bias(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Mean bias ``<pred - ref>`` reduced over ``dim``."""
    return _score(skill.bias, pred, ref, dim, weights)


def correlation(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Pearson correlation between prediction and reference over ``dim``.

    NaN where either field is constant.
    """
    return _score(skill.correlation, pred, ref, dim, weights)


def r2_score(
    pred: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Dim,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Coefficient of determination: ``1 - SS_res / SS_tot``.

    Against a constant reference a perfect prediction scores 1 and any other
    ``-inf``.
    """
    return _score(skill.r2_score, pred, ref, dim, weights)


# ---------- Layer-1 (Operator wrappers) -----------------------------------


class _PixelMetricOp(Operator):
    """Base class for two-input pixel metrics.

    The constructor carries the Dataset-level selector (``variable``),
    the reduce-axis spec (``dims``) and, optionally, the name of a variable of
    the reference that holds the ``weights``. ``_apply`` selects the named
    variable from each input Dataset and forwards the resulting
    DataArrays to the Layer-0 primitive.
    """

    _fn: Any = None

    def __init__(self, variable: str, dims: Dim, weights: str | None = None):
        self.variable = variable
        self.dims = dims if isinstance(dims, str) else list(dims)
        self.weights = weights

    def _apply(self, ds_pred: xr.Dataset, ds_ref: xr.Dataset) -> xr.DataArray:
        return self.__class__._fn(
            ds_pred[self.variable],
            ds_ref[self.variable],
            dim=self.dims,
            weights=None if self.weights is None else ds_ref[self.weights],
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dims": self.dims if isinstance(self.dims, str) else list(self.dims),
            "weights": self.weights,
        }

    def compute_output_signature(
        self,
        input_signature: Signature | tuple[Signature, ...],
    ) -> Signature:
        """Infer the output signature: input dims minus the reduced ``dims``.

        Tuple inputs (one signature per metric operand) are validated
        for matching dim names and sizes before the first is used.
        """
        if isinstance(input_signature, tuple):
            signature = input_signature[0]
            for other in input_signature[1:]:
                _validate_compatible_signatures(signature, other)
        else:
            signature = input_signature
        dims = (self.dims,) if isinstance(self.dims, str) else tuple(self.dims)
        # Integers become float64 (see _score above), and the inferred output
        # dtype is that of a float64 input.
        return _sig.make(_sig.sizes(_sig.drop_dims(signature, dims)), dtype=np.float64)


class MSE(_PixelMetricOp):
    """Mean-squared-error operator."""

    _fn = staticmethod(mse)


class RMSE(_PixelMetricOp):
    """Root-mean-squared-error operator."""

    _fn = staticmethod(rmse)


class NRMSE(_PixelMetricOp):
    """Normalized-RMSE operator (``1 - RMSE / sqrt(<ref^2>)``)."""

    _fn = staticmethod(nrmse)


class NRMSEScore(_PixelMetricOp):
    """Mercator/OceanBench NRMSE skill operator (``1 - RMSE / std(ref)``)."""

    _fn = staticmethod(nrmse_score)


class MAE(_PixelMetricOp):
    """Mean-absolute-error operator."""

    _fn = staticmethod(mae)


class Bias(_PixelMetricOp):
    """Mean-bias operator (``<pred - ref>``)."""

    _fn = staticmethod(bias)


class Correlation(_PixelMetricOp):
    """Pearson-correlation operator."""

    _fn = staticmethod(correlation)


class R2Score(_PixelMetricOp):
    """Coefficient-of-determination operator (``1 - SS_res / SS_tot``)."""

    _fn = staticmethod(r2_score)


def _validate_compatible_signatures(left: Signature, right: Signature) -> None:
    lhs, rhs = _sig.sizes(left), _sig.sizes(right)
    if set(lhs) != set(rhs):
        raise ValueError(
            "Metric inputs must have matching dimension names; "
            f"got {tuple(lhs)} and {tuple(rhs)}."
        )
    mismatched = {
        name: (lhs[name], rhs[name])
        for name in lhs
        if lhs[name] is not None and rhs[name] is not None and lhs[name] != rhs[name]
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
    "NRMSEScore",
    "R2Score",
    "bias",
    "correlation",
    "mae",
    "mse",
    "nrmse",
    "nrmse_score",
    "r2_score",
    "rmse",
]
