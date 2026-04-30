"""``ModelOp`` and friends — Layer-1 wrappers for trained ML models.

``ModelOp`` is a framework-agnostic adapter that turns any callable
model into an :class:`~xr_toolz.core.Operator`. It marshals data
between :mod:`xarray` and raw arrays, dispatches to a configurable
method (``predict`` / ``predict_proba`` / ``transform`` / ...), and
wraps the result back into an :class:`xarray.DataArray` that carries
the input's non-feature coordinates.

Per design decision **D4**, this module never imports ``sklearn``,
``jax``, ``torch``, or ``equinox``. The two backend-flavored
subclasses set ergonomic defaults but defer any backend import to
first use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from xr_toolz.core import Operator


_XR_TYPES = (xr.DataArray, xr.Dataset)


class ModelOp(Operator):
    """Wrap a trained model as an :class:`~xr_toolz.core.Operator`.

    The wrapped model is expected to expose a method named ``method``
    (default ``"predict"``) that maps a 2-D ``(n_samples, n_features)``
    array to either a 1-D ``(n_samples,)`` or 2-D
    ``(n_samples, n_outputs)`` array. ``ModelOp`` reshapes leading
    non-feature dimensions of the input automatically.

    Args:
        model: Any object exposing the configured ``method``. Not
            stored in :meth:`get_config`; the reference is replaced by
            the literal string ``"<model>"`` so configs are JSON-safe.
        method: Name of the model method to call. Defaults to
            ``"predict"``. Common alternatives are ``"predict_proba"``
            and ``"transform"``.
        feature_dim: Name of the feature dimension on the input
            :class:`xarray.DataArray`. All other dims are treated as
            sample dims and preserved on the output.
        output_name: Name attached to the returned
            :class:`xarray.DataArray`. Defaults to ``"prediction"``.
        output_dim: Name of the dimension along the model's output
            channels (only used when the model returns a 2-D array).
            Defaults to ``"output"``.

    Notes:
        This class follows D4: no backend (sklearn / jax / torch) is
        imported here. The model is invoked via duck typing.
    """

    def __init__(
        self,
        model: Any,
        *,
        method: str = "predict",
        feature_dim: str = "feature",
        output_name: str = "prediction",
        output_dim: str = "output",
    ) -> None:
        self.model = model
        self.method = method
        self.feature_dim = feature_dim
        self.output_name = output_name
        self.output_dim = output_dim

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _resolve_fn(self) -> Callable[[np.ndarray], Any]:
        """Return the callable to invoke on the flattened ``(N, F)`` array.

        Subclasses override this to customise dispatch (e.g. JIT-compile
        a JAX function once and cache it).
        """
        try:
            return getattr(self.model, self.method)
        except AttributeError as exc:
            raise AttributeError(
                f"Model {type(self.model).__name__!r} has no method {self.method!r}."
            ) from exc

    # ------------------------------------------------------------------
    # xarray <-> array marshalling
    # ------------------------------------------------------------------

    def _apply(self, data: xr.DataArray | xr.Dataset | np.ndarray) -> xr.DataArray:
        """Run the model on ``data`` and wrap the output as a DataArray.

        Args:
            data: Either an :class:`xarray.DataArray` with a
                ``feature_dim`` axis, an :class:`xarray.Dataset` whose
                data variables are stacked along ``feature_dim``, or a
                raw 2-D ``numpy`` array. Raw arrays are returned as
                :class:`xarray.DataArray` with default ``sample`` /
                ``output_dim`` axes.
        """
        da = self._as_dataarray(data)

        if self.feature_dim not in da.dims:
            raise ValueError(
                f"Input is missing feature dim {self.feature_dim!r}; "
                f"got dims={da.dims}."
            )

        sample_dims = tuple(d for d in da.dims if d != self.feature_dim)
        flat = da.transpose(*sample_dims, self.feature_dim).values
        sample_shape = flat.shape[:-1]
        n_features = flat.shape[-1]
        x = flat.reshape(-1, n_features) if sample_shape else flat[np.newaxis, :]

        fn = self._resolve_fn()
        y = np.asarray(fn(x))

        return self._wrap_output(y, da, sample_dims, sample_shape)

    def _as_dataarray(
        self, data: xr.DataArray | xr.Dataset | np.ndarray
    ) -> xr.DataArray:
        if isinstance(data, xr.DataArray):
            return data
        if isinstance(data, xr.Dataset):
            return data.to_array(dim=self.feature_dim).transpose(..., self.feature_dim)
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(
                "Raw array inputs must be 2-D ``(n_samples, n_features)``; "
                f"got shape={arr.shape}. For a single sample, pass "
                "``arr[np.newaxis, :]`` (or wrap higher-rank data as a "
                "DataArray with a named feature dim)."
            )
        return xr.DataArray(arr, dims=("sample", self.feature_dim))

    def _wrap_output(
        self,
        y: np.ndarray,
        da: xr.DataArray,
        sample_dims: tuple[str, ...],
        sample_shape: tuple[int, ...],
    ) -> xr.DataArray:
        if y.ndim not in (1, 2):
            raise ValueError(
                "Model output must be 1-D ``(n_samples,)`` or 2-D "
                f"``(n_samples, n_outputs)``; got shape={y.shape}. "
                "Higher-rank outputs cannot be unambiguously reshaped "
                "back onto the input's sample dims."
            )

        sample_dims_set = set(sample_dims)
        # Carry through any coord (dim or auxiliary, e.g. lat(sample),
        # station_id(time, sample)) whose dims are entirely sample dims.
        # The feature dim and anything that depends on it are dropped.
        coords = {
            name: coord
            for name, coord in da.coords.items()
            if set(coord.dims).issubset(sample_dims_set)
        }

        if y.ndim == 1:
            target_shape = sample_shape if sample_shape else (1,)
            arr = y.reshape(target_shape)
            dims = sample_dims if sample_dims else ("sample",)
            return xr.DataArray(
                arr,
                dims=dims,
                coords=coords,
                name=self.output_name,
                attrs=dict(da.attrs),
            )

        if self.output_dim in sample_dims:
            raise ValueError(
                f"output_dim {self.output_dim!r} collides with an existing "
                f"sample dim {sample_dims}; pass a distinct ``output_dim=`` "
                "to ModelOp to disambiguate."
            )

        n_out = y.shape[-1]
        target_shape = (*sample_shape, n_out) if sample_shape else (1, n_out)
        arr = y.reshape(target_shape)
        dims = (
            (*sample_dims, self.output_dim)
            if sample_dims
            else (
                "sample",
                self.output_dim,
            )
        )
        return xr.DataArray(
            arr,
            dims=dims,
            coords=coords,
            name=self.output_name,
            attrs=dict(da.attrs),
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the wrapper config.

        The wrapped ``model`` is referenced as the literal string
        ``"<model>"`` because most trained models are not JSON-safe.
        """
        return {
            "model": "<model>",
            "method": self.method,
            "feature_dim": self.feature_dim,
            "output_name": self.output_name,
            "output_dim": self.output_dim,
        }


class SklearnModelOp(ModelOp):
    """``ModelOp`` flavored for scikit-learn estimators.

    Functionally identical to :class:`ModelOp`, but documents the
    expected ``predict`` / ``predict_proba`` interface and validates,
    via duck typing, that the wrapped model exposes the requested
    method. ``sklearn`` itself is never imported.

    Args:
        model: A fitted sklearn-style estimator.
        method: Either ``"predict"`` (default) or ``"predict_proba"``;
            any other method exposed by the estimator is also accepted.
        **kwargs: Forwarded to :class:`ModelOp`.
    """

    def __init__(
        self,
        model: Any,
        *,
        method: str = "predict",
        **kwargs: Any,
    ) -> None:
        if not hasattr(model, method):
            raise AttributeError(
                f"sklearn-style model {type(model).__name__!r} has no "
                f"method {method!r}; pass method= explicitly if needed."
            )
        super().__init__(model, method=method, **kwargs)


class JaxModelOp(ModelOp):
    """``ModelOp`` flavored for JAX / pytree-callable models.

    The model can be any pytree-callable: an ``equinox.Module``, a
    plain Python function closing over JAX arrays, a ``flax`` module
    paired with bound parameters, etc. By default the model is invoked
    as ``model(x)``; pass ``method=`` to dispatch to a named method.

    When ``jit=True`` (default), the dispatch function is compiled with
    :func:`jax.jit` on first use. ``jax`` is imported lazily inside
    :meth:`_resolve_fn`.

    Args:
        model: A callable / pytree-callable accepting a 2-D array.
        method: Optional method name. ``None`` (default) means call the
            model itself as ``model(x)``.
        jit: If True, jit-compile the dispatch function. Defaults to
            True.
        **kwargs: Forwarded to :class:`ModelOp` (``feature_dim``,
            ``output_name``, ``output_dim``).
    """

    def __init__(
        self,
        model: Any,
        *,
        method: str | None = None,
        jit: bool = True,
        **kwargs: Any,
    ) -> None:
        # Use a sentinel string for the parent's bookkeeping; we
        # override _resolve_fn so the actual method name is irrelevant
        # to the base class.
        super().__init__(model, method=method or "__call__", **kwargs)
        self._method_override = method
        self.jit = jit
        self._compiled: Callable[[np.ndarray], Any] | None = None

    def _resolve_fn(self) -> Callable[[np.ndarray], Any]:
        if self._compiled is not None:
            return self._compiled

        if self._method_override is None:
            base_fn: Callable[[np.ndarray], Any] = self.model
        else:
            try:
                base_fn = getattr(self.model, self._method_override)
            except AttributeError as exc:
                raise AttributeError(
                    f"JAX model {type(self.model).__name__!r} has no method "
                    f"{self._method_override!r}."
                ) from exc

        if self.jit:
            import jax  # lazy by design (D4)

            base_fn = jax.jit(base_fn)

        self._compiled = base_fn
        return base_fn

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["method"] = self._method_override
        cfg["jit"] = self.jit
        return cfg
