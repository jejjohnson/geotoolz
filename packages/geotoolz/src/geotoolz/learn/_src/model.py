"""ModelOp — framework-agnostic inference operator.

Wraps any callable (a torch model, a sklearn estimator, a JAX function,
a plain function) as a `pipekit.Operator`. Duck-types the call — either
invokes ``model(arr)`` directly or ``getattr(model, method)(arr)`` for
sklearn-style ``predict``.

The wrapper never imports a framework — it only calls what the user
hands it. This keeps `geotoolz` framework-optional and lets users wire
their own training loops / optimizers / device placement upstream of the
operator graph.

See `geotoolz.md` §6.4.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pipekit import Carrier, Operator


class ModelOp(Operator):
    """Wrap any callable as an Operator.

    Materialises the GeoTensor to a plain ``np.ndarray`` (via
    ``np.asarray``) before handing it to the model — frameworks that
    strip the subclass (torch, JAX, sklearn) don't care, and frameworks
    that preserve it (numpy proper) still see something sensible.

    Args:
        model: Any object that can be called as ``model(arr)`` or whose
            ``method`` attribute can be called as
            ``model.predict(arr)``. No isinstance / framework imports.
        method: Method name to invoke on ``model``. Default
            ``"__call__"`` — equivalent to ``model(arr)``. Set to
            ``"predict"`` for sklearn estimators.
        batch_size: If set, split the input along axis 0 into chunks of
            this size, call the model once per chunk, concatenate the
            results along axis 0. Useful when the model can't fit the
            whole input in GPU memory.

    Note:
        ``forbid_in_yaml = True`` — the model is a runtime object and
        won't round-trip to YAML. Users typically pin a model artifact
        (state-dict + class config) themselves.

    Examples:
        Inference with a sklearn classifier::

            op = ModelOp(rf_clf, method="predict")
            preds = op(features_gt)

        Batched inference with a torch model::

            op = ModelOp(unet_model, batch_size=8)
            preds = op(chips_gt)  # iterates 8 chips at a time
    """

    forbid_in_yaml: ClassVar[bool] = True
    # ConfigMixin would auto-derive `model` from `__init__` as a non-JSON
    # opaque object; override with a curated debug repr below.
    __config_mixin_auto__: ClassVar[bool] = False

    def __init__(
        self,
        model: Any,
        *,
        method: str = "__call__",
        batch_size: int | None = None,
    ) -> None:
        self.model = model
        self.method = method
        self.batch_size = batch_size

    def _resolve_callable(self) -> Any:
        if self.method == "__call__":
            return self.model
        return getattr(self.model, self.method)

    def _apply(self, gt: Carrier) -> Any:
        arr = np.asarray(gt)
        fn = self._resolve_callable()
        if self.batch_size is None:
            return fn(arr)
        return self._batched(fn, arr)

    def _batched(self, fn: Any, arr: np.ndarray) -> np.ndarray:
        """Split ``arr`` along axis 0, call ``fn`` per chunk, concatenate.

        Plain ``np.concatenate`` along axis 0 — works when the model's
        output preserves the batch dimension (the common case).

        Empty inputs (``arr.shape[0] == 0``) are passed straight to the
        model in one call: ``np.concatenate`` cannot accept an empty list
        of chunks, and the model is free to return a meaningful
        zero-length result.
        """
        n = arr.shape[0]
        if n == 0:
            return fn(arr)
        chunks: list[Any] = []
        bs = int(self.batch_size or n)
        for start in range(0, n, bs):
            chunks.append(fn(arr[start : start + bs]))
        return np.concatenate(chunks, axis=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "model_type": type(self.model).__name__,
            "method": self.method,
            "batch_size": self.batch_size,
        }


__all__ = ["ModelOp"]
