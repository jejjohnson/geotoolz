"""Layer-1 ``Operator`` wrappers for generic dataset transforms."""

from __future__ import annotations

from typing import Any

from pipekit import Signature

from xrops._src import signature as _sig
from xrops._src.operator import Operator


_REDUCE_OPS = ("mean", "sum", "median", "min", "max", "std", "var")


class ReduceDims(Operator):
    """Reduce a Dataset over one or more dims with a named aggregation.

    Wraps the corresponding ``xr.Dataset`` reducer (``mean`` / ``sum`` /
    …). Promotes the inline ``MeanOverDim`` from the V1.5 PSD demo so
    notebooks no longer redefine a one-method subclass per reduction.

    Args:
        op: Aggregation name. One of ``"mean"``, ``"sum"``, ``"median"``,
            ``"min"``, ``"max"``, ``"std"``, ``"var"``.
        dim: Dim or tuple of dims to reduce over.
        keepdims: Whether to keep reduced dims as length-1 (forwarded to
            the underlying xarray reducer's ``keepdims`` kwarg).
    """

    def __init__(
        self,
        op: str = "mean",
        dim: str | tuple[str, ...] = ("time",),
        *,
        keepdims: bool = False,
    ) -> None:
        if op not in _REDUCE_OPS:
            raise ValueError(
                f"Unknown reduce op {op!r}; expected one of {_REDUCE_OPS}."
            )
        self.op = op
        self.dim = (dim,) if isinstance(dim, str) else tuple(dim)
        self.keepdims = bool(keepdims)

    def _apply(self, ds):
        reducer = getattr(ds, self.op)
        return reducer(dim=self.dim, keepdims=self.keepdims)

    def get_config(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "dim": list(self.dim),
            "keepdims": self.keepdims,
        }

    def compute_output_signature(self, input_signature: Signature) -> Signature:
        if self.keepdims:
            return _sig.replace_dims(input_signature, {dim: 1 for dim in self.dim})
        return _sig.drop_dims(input_signature, self.dim)


__all__ = ["ReduceDims"]
