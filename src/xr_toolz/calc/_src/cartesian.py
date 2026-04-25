"""Finite-difference operators on uniform Cartesian grids.

All operators delegate the numerical work to :mod:`finitediffx` and only
handle the xarray ↔ raw-array plumbing plus the uniform-spacing check.
Coordinates are read directly from the :class:`xr.DataArray`; the only
requirement is that each differentiation dimension carries a 1-D
coordinate with constant spacing (within ``uniform_rtol``).
"""

from __future__ import annotations

import threading
from typing import Any, cast

import finitediffx as fdx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr


# ``fdx.difference`` is a JIT-compiled ``PjitFunction`` whose kwargs the
# type checker can't introspect. Re-type it as ``Any`` so we don't have
# to plaster ``# type: ignore`` on every call site.
_fdx_difference: Any = cast(Any, fdx.difference)


# JAX's ``jax_enable_x64`` flag is process-global. Concurrent calls to
# :class:`_x64_scope` from multiple threads would otherwise race and
# leave the flag in a wrong state for the duration of a peer's stencil
# call. Serialising the toggle (and the kernel call within it) under a
# module-level lock keeps each call's precision deterministic. The cost
# is negligible — the lock is held only for the duration of one
# ``fdx.difference`` invocation.
_X64_LOCK = threading.Lock()


class _x64_scope:
    """Toggle JAX's ``jax_enable_x64`` flag for the duration of a block.

    JAX defaults to float32; without x64 enabled the stencil arithmetic
    silently downcasts and the user's float64 input gets a float32-quality
    answer. We flip the flag on entry only when we need it and restore it
    on exit so we don't pollute the surrounding JAX session.

    Thread safety: enters under ``_X64_LOCK`` so concurrent callers can't
    interleave their flag flips. Reentrant calls from the same thread are
    not supported (use a single block per call).
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._prev: bool | None = None
        self._held = False

    def __enter__(self) -> None:
        if not self._enable:
            return
        _X64_LOCK.acquire()
        self._held = True
        self._prev = bool(jax.config.read("jax_enable_x64"))
        if not self._prev:
            jax.config.update("jax_enable_x64", True)

    def __exit__(self, *exc: object) -> None:
        try:
            if self._prev is False:
                jax.config.update("jax_enable_x64", False)
        finally:
            if self._held:
                _X64_LOCK.release()
                self._held = False


def _difference(
    values: np.ndarray,
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    method: str,
) -> np.ndarray:
    """Run :func:`finitediffx.difference` while preserving input dtype."""
    in_dtype = np.asarray(values).dtype
    with _x64_scope(in_dtype == np.float64):
        raw = _fdx_difference(
            jnp.asarray(values),
            axis=axis,
            step_size=step_size,
            accuracy=accuracy,
            method=method,
        )
        return np.asarray(raw)


def _uniform_step(coord: xr.DataArray, *, rtol: float = 1e-6) -> float:
    """Return the uniform step of a 1-D coordinate.

    Raises:
        ValueError: if the coordinate has fewer than two samples or is
            not uniformly spaced within ``rtol``.
    """
    values = np.asarray(coord.values)
    if values.size < 2:
        raise ValueError(
            f"Coordinate {coord.name!r} has {values.size} sample(s); "
            "need at least 2 to compute a finite-difference step."
        )
    diffs = np.diff(values)
    step = float(diffs[0])
    if step == 0.0:
        raise ValueError(
            f"Coordinate {coord.name!r} has zero spacing — every sample is "
            f"identical ({float(values[0])}). A finite-difference step is "
            "undefined."
        )
    if not np.allclose(diffs, step, rtol=rtol, atol=0.0):
        raise ValueError(
            f"Coordinate {coord.name!r} is not uniformly spaced "
            f"(min={float(diffs.min())}, max={float(diffs.max())}); "
            "use geometry='rectilinear' for non-uniform 1-D coordinates."
        )
    return step


def _output_name(da: xr.DataArray, dim: str) -> str | None:
    """Build a sensible name for ``∂da/∂<dim>``."""
    if da.name is None:
        return None
    return f"d{da.name}_d{dim}"


def cartesian_partial(
    da: xr.DataArray,
    dim: str,
    *,
    accuracy: int = 1,
    method: str = "central",
    uniform_rtol: float = 1e-6,
) -> xr.DataArray:
    """Partial derivative ``∂da/∂<dim>`` on a uniform Cartesian grid.

    Args:
        da: Input field. Must carry a 1-D coordinate for ``dim`` with
            constant spacing.
        dim: Dimension along which to differentiate.
        accuracy: ``finitediffx`` accuracy order. ``1`` matches
            :func:`numpy.gradient`'s 2nd-order centred default.
        method: ``"central"``, ``"forward"``, or ``"backward"``.
        uniform_rtol: Relative tolerance for the uniform-spacing check.

    Returns:
        DataArray with the same dims/coords as ``da`` and a default
        name of ``f"d{da.name}_d{dim}"``.
    """
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )
    axis = da.get_axis_num(dim)
    step = _uniform_step(da[dim], rtol=uniform_rtol)
    raw = _difference(
        da.values,
        axis=axis,
        step_size=step,
        accuracy=accuracy,
        method=method,
    )
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        name=_output_name(da, dim),
        attrs=dict(da.attrs),
    )


def cartesian_gradient(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    uniform_rtol: float = 1e-6,
) -> xr.Dataset:
    """Gradient ``∇da`` on a uniform Cartesian grid.

    Args:
        da: Input scalar field.
        dims: Dimensions to differentiate against. Defaults to ``da.dims``.
        accuracy: Scalar (applied to every dim) or per-dim tuple matching
            the length of ``dims``.
        method: Forwarded to :func:`finitediffx.difference`.
        uniform_rtol: Forwarded to :func:`cartesian_partial`.

    Returns:
        Dataset with one DataArray per requested dim, keyed
        ``f"d{da.name or 'f'}_d{dim}"``.
    """
    target_dims = tuple(da.dims) if dims is None else tuple(dims)
    if isinstance(accuracy, int):
        per_dim = (accuracy,) * len(target_dims)
    else:
        per_dim = tuple(accuracy)
        if len(per_dim) != len(target_dims):
            raise ValueError(
                f"accuracy tuple length ({len(per_dim)}) does not match "
                f"number of dims ({len(target_dims)})."
            )
    base = da.name or "f"
    out: dict[str, xr.DataArray] = {}
    for dim, acc in zip(target_dims, per_dim, strict=True):
        component = cartesian_partial(
            da,
            dim,
            accuracy=acc,
            method=method,
            uniform_rtol=uniform_rtol,
        )
        out[f"d{base}_d{dim}"] = component
    return xr.Dataset(out)
