"""Finite-difference operators on uniform Cartesian grids.

All operators delegate the numerical work to :mod:`finitediffx` and only
handle the xarray ↔ raw-array plumbing plus the uniform-spacing check.
Coordinates are read directly from the :class:`xr.DataArray`; the only
requirement is that each differentiation dimension carries a 1-D
coordinate with constant spacing (within ``uniform_rtol``).
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Hashable
from typing import Any, cast

import finitediffx as fdx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Float


# ``fdx.difference`` is a JIT-compiled ``PjitFunction`` whose kwargs the
# type checker can't introspect. Re-type it as ``Any`` so we don't have
# to plaster ignore comments on every call site.
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
    values: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    method: str,
    derivative: int = 1,
) -> Float[np.ndarray, "*shape"]:
    """Run :func:`finitediffx.difference` while preserving input dtype."""
    in_dtype = np.asarray(values).dtype
    with _x64_scope(in_dtype == np.float64):
        raw = _fdx_difference(
            jnp.asarray(values),
            axis=axis,
            step_size=step_size,
            accuracy=accuracy,
            method=method,
            derivative=derivative,
        )
        return np.asarray(raw)


def _stencil_halo(*, derivative: int, accuracy: int) -> int:
    """Upper bound on the half-width of the interior (central) stencil.

    ``finitediffx`` builds its interior stencil from
    ``_generate_central_offsets(derivative, accuracy + 1)``, whose widest
    offset is ``ceil((derivative + accuracy) / 2)``. We pad by the looser
    ``derivative + accuracy`` so we never under-pad if that internal rule
    changes; over-padding only costs a few extra columns, whereas
    under-padding would silently leave one-sided values at the seam.
    ``test_periodic.py`` pins the behaviour across accuracies so an
    insufficient halo fails loudly.
    """
    return derivative + accuracy


def _warn_duplicate_endpoint(
    values: Float[np.ndarray, "*shape"], *, axis: int, dim: Hashable
) -> None:
    """Warn when an axis looks endpoint-inclusive under a periodic wrap.

    The wrap treats the axis as endpoint-exclusive: the period is the
    full extent, ``n·Δ``, so the last sample's right neighbour is the
    first. A grid built with ``np.linspace(0, L, n)`` — endpoint
    *inclusive*, and numpy's default — stores the same physical location
    twice, so wrapping makes a point its own neighbour and roughly halves
    the seam derivative.

    Endpoint inclusion cannot be derived from the coordinate: both
    conventions are uniform with the same step, and the wrap is
    self-consistent either way. It also cannot be *concluded* from the
    field, because a legitimate endpoint-exclusive signal may take the
    same value at both ends (period 3 over ``[0, 1, 2]`` with values
    ``[0, 1, 0]``). So this only warns: the computation proceeds under
    the documented endpoint-exclusive contract, and a caller who knows
    their grid is exclusive can silence it. Where the period *is*
    knowable the check is a hard error instead — see the spherical
    backend, which requires a full 360° span.
    """
    first = np.take(values, 0, axis=axis)
    last = np.take(values, -1, axis=axis)
    if not np.allclose(first, last, rtol=1e-12, atol=1e-12, equal_nan=True):
        return
    if np.allclose(values, np.take(values, [0], axis=axis), rtol=1e-12, atol=1e-12):
        return  # constant along the axis: the derivative is 0 either way
    warnings.warn(
        f"Coordinate {dim!r} takes the same value at its first and last "
        "sample. If this grid stores both ends of the period (as "
        "np.linspace(0, L, n) does by default), periodic=True will make "
        "that point its own neighbour and halve the derivative at the "
        f"seam; drop the repeat with da.isel({dim}=slice(0, -1)). If the "
        "grid is genuinely endpoint-exclusive, this warning is spurious.",
        UserWarning,
        stacklevel=3,
    )


def _difference_periodic(
    values: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    method: str,
    derivative: int = 1,
) -> Float[np.ndarray, "*shape"]:
    """Finite difference with a periodic wrap along ``axis``.

    Pads the axis by a full stencil halo taken from the opposite edge,
    differentiates, then trims the padding away. Every original point
    therefore sees a centred stencil — including the first and last,
    which would otherwise fall back to one-sided differences and leave a
    visible seam (e.g. at the dateline on a global longitude axis).
    """
    size = values.shape[axis]
    halo = _stencil_halo(derivative=derivative, accuracy=accuracy)
    idx = np.arange(-halo, size + halo) % size
    padded = np.take(values, idx, axis=axis)
    raw = _difference(
        padded,
        axis=axis,
        step_size=step_size,
        accuracy=accuracy,
        method=method,
        derivative=derivative,
    )
    return np.take(raw, np.arange(halo, halo + size), axis=axis)


def _difference_maybe_periodic(
    values: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    method: str,
    derivative: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Dispatch to the periodic or the plain stencil."""
    kernel = _difference_periodic if periodic else _difference
    return kernel(
        values,
        axis=axis,
        step_size=step_size,
        accuracy=accuracy,
        method=method,
        derivative=derivative,
    )


def _validate_order(order: int) -> None:
    """Guard the derivative order shared by all three geometry backends.

    Raises:
        ValueError: if ``order`` is not a positive integer.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError(f"order must be a positive integer; got {order!r}.")


def _per_dim_accuracy(
    accuracy: int | tuple[int, ...], dims: tuple[Hashable, ...]
) -> tuple[int, ...]:
    """Broadcast a scalar accuracy over ``dims``, or validate a per-dim tuple.

    Raises:
        ValueError: if a tuple is given whose length does not match ``dims``.
    """
    if isinstance(accuracy, int):
        return (accuracy,) * len(dims)
    per_dim = tuple(accuracy)
    if len(per_dim) != len(dims):
        raise ValueError(
            f"accuracy tuple length ({len(per_dim)}) does not match "
            f"number of dims ({len(dims)})."
        )
    return per_dim


def _require_1d_coord(coord: xr.DataArray) -> None:
    """Guard against multi-dimensional (curvilinear) coordinates.

    Raises:
        ValueError: if ``coord`` is not one-dimensional.
    """
    if coord.ndim != 1:
        raise ValueError(
            f"Coordinate {coord.name!r} is {coord.ndim}-D; xrgrad supports "
            "1-D coordinates only. Curvilinear grids with 2-D lon/lat "
            "coordinates are not supported."
        )


def _require_coord(da: xr.DataArray, dim: str) -> None:
    """Guard against differentiating a dimension that carries no coordinate.

    Raises:
        ValueError: if ``dim`` has no coordinate on ``da``.
    """
    if dim not in da.coords:
        raise ValueError(
            f"Dimension {dim!r} has no coordinate on this DataArray; finite "
            "differences need coordinate values to compute step sizes. "
            f"Assign one, e.g. da.assign_coords({dim}=np.arange(da.sizes[{dim!r}]))."
        )


def _coord_to_float(
    values: np.ndarray, *, name: Hashable = None
) -> Float[np.ndarray, "n"]:
    """Return coordinate values as float64, measured from the first sample.

    ``datetime64`` and ``timedelta64`` become float **seconds**;
    integer dtypes are anchored before widening; floats pass through.
    Only differences of the result are ever used (step sizes and the
    rectilinear chain rule), so the shift is immaterial.

    The result is accumulated from **adjacent gaps** rather than from a
    full-span offset, because the full span is what overflows: nanosecond
    ticks are a signed 64-bit count saturating at roughly ±292 years, so
    for a ``datetime64[ns]`` axis covering 1800–2200 — comfortably inside
    what that dtype can *store* — even ``values - values[0]`` wraps to a
    negative offset. The same holds for a wide integer axis whose range
    exceeds int64 while its steps do not. Each gap is converted at the
    finest resolution that cannot overflow (see
    :func:`_temporal_gaps_seconds` and :func:`_integer_gaps`), and the
    running sum stays in float64.

    Working from gaps also keeps precision that a straight cast loses: an
    integer axis with a large origin (``10**18 + k``) collapses to a
    constant if cast to float64 directly, since adjacent values fall
    inside one float64 ulp there.

    Raises:
        ValueError: for object-dtype coordinates such as ``cftime``
            calendars, which have no lossless conversion here.
    """
    values = np.asarray(values)
    if values.dtype == object:
        raise ValueError(
            f"Coordinate {name!r} has object dtype (e.g. a cftime calendar); "
            "xrgrad cannot derive a numeric step from it. Convert to "
            "datetime64 first, e.g. ds.convert_calendar('standard', "
            "use_cftime=False)."
        )
    is_temporal = np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(
        values.dtype, np.timedelta64
    )
    if is_temporal:
        return _accumulate(_temporal_gaps_seconds(values), size=values.size)
    if np.issubdtype(values.dtype, np.integer):
        return _accumulate(_integer_gaps(values), size=values.size)
    return np.asarray(values, dtype=np.float64)


def _accumulate(gaps: Float[np.ndarray, "n-1"], *, size: int) -> Float[np.ndarray, "n"]:
    """Running sum of adjacent gaps, starting from zero."""
    out = np.zeros(size, dtype=np.float64)
    if size > 1:
        np.cumsum(gaps, out=out[1:])
    return out


def _temporal_gaps_seconds(
    values: np.ndarray,
) -> Float[np.ndarray, "n-1"]:
    """Adjacent gaps of a temporal axis, in float seconds.

    Subtracting in the source unit is exact but overflows when a single
    gap exceeds that unit's signed tick range — roughly 292 years for
    nanoseconds. Casting to a coarser unit only ever divides, so
    second-resolution gaps are always computable; they are used to decide
    whether the exact source-unit subtraction is safe. A gap too wide for
    the source unit could not have carried sub-second meaning anyway, so
    dropping to seconds there loses nothing real.
    """
    if values.size < 2:
        return np.zeros(0, dtype=np.float64)
    unit = np.datetime_data(values.dtype)[0]
    ticks_per_second = np.timedelta64(1, "s") / np.timedelta64(1, unit)
    if ticks_per_second <= 1.0:
        # Second-or-coarser ticks: a gap would have to span ~3e11 years
        # to overflow, which no representable axis does.
        return (np.diff(values) / np.timedelta64(1, "s")).astype(np.float64)
    coarse = (
        "datetime64[s]"
        if np.issubdtype(values.dtype, np.datetime64)
        else "timedelta64[s]"
    )
    gaps_s = np.diff(values.astype(coarse).astype(np.int64)).astype(np.float64)
    safe_span_s = 0.9 * np.iinfo(np.int64).max / ticks_per_second
    if np.abs(gaps_s).max() >= safe_span_s:
        return gaps_s
    return (np.diff(values) / np.timedelta64(1, "s")).astype(np.float64)


def _integer_gaps(values: np.ndarray) -> Float[np.ndarray, "n-1"]:
    """Adjacent gaps of an integer axis, as float64.

    Differencing adjacent values keeps a wide axis exact where anchoring
    on the first sample would not: ``[-9e18, …, 6e18]`` steps uniformly
    by ``5e18``, but its total range overflows int64. Where even an
    adjacent gap is too wide for the integer type, the float estimate is
    the best available and is used instead.
    """
    if values.size < 2:
        return np.zeros(0, dtype=np.float64)
    approx = np.diff(values.astype(np.float64))
    if np.abs(approx).max() >= 0.9 * np.iinfo(np.int64).max:
        return approx
    return np.diff(values.astype(np.int64)).astype(np.float64)


def _uniform_step(coord: xr.DataArray, *, rtol: float = 1e-6) -> float:
    """Return the uniform step of a 1-D coordinate.

    Temporal coordinates are converted to seconds first, so the step of a
    ``datetime64`` axis comes back as float seconds.

    Raises:
        ValueError: if the coordinate is not 1-D, has fewer than two
            samples, or is not uniformly spaced within ``rtol``.
    """
    _require_1d_coord(coord)
    values = _coord_to_float(coord.values, name=coord.name)
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


def _output_name(da: xr.DataArray, dim: str, *, order: int = 1) -> str | None:
    """Build a sensible name for ``∂ᵒʳᵈᵉʳda/∂<dim>ᵒʳᵈᵉʳ``."""
    if da.name is None:
        return None
    if order == 1:
        return f"d{da.name}_d{dim}"
    return f"d{order}{da.name}_d{dim}{order}"


def cartesian_partial(
    da: xr.DataArray,
    dim: str,
    *,
    order: int = 1,
    accuracy: int = 1,
    method: str = "central",
    periodic: bool = False,
    uniform_rtol: float = 1e-6,
) -> xr.DataArray:
    """Partial derivative ``∂ᵒʳᵈᵉʳda/∂<dim>ᵒʳᵈᵉʳ`` on a uniform Cartesian grid.

    Args:
        da: Input field. Must carry a 1-D coordinate for ``dim`` with
            constant spacing.
        dim: Dimension along which to differentiate.
        order: Derivative order. ``1`` (default) is the first derivative;
            ``2`` gives ``∂²da/∂dim²`` from a single second-derivative
            stencil rather than two composed first-derivative passes.
        accuracy: ``finitediffx`` accuracy order. ``1`` matches
            :func:`numpy.gradient`'s 2nd-order centred default.
        method: ``"central"``, ``"forward"``, or ``"backward"``.
        periodic: Treat ``dim`` as wrapping, so the first and last points
            are differentiated against each other instead of falling back
            to one-sided stencils. The axis is taken to be **endpoint
            exclusive**: the period is ``n * step``, so the last sample is
            one step before the first repeats. That convention cannot be
            verified from the coordinate, so a grid that stores both ends
            of the period (``np.linspace(0, L, n)``, endpoint inclusive)
            only draws a ``UserWarning`` when the field's ends coincide.
        uniform_rtol: Relative tolerance for the uniform-spacing check.

    Returns:
        DataArray with the same dims/coords as ``da`` and a default
        name of ``f"d{da.name}_d{dim}"`` (``f"d2{da.name}_d{dim}2"`` at
        ``order=2``).

    Raises:
        ValueError: if ``dim`` is absent from ``da.dims``, carries no
            coordinate, or ``order`` is not a positive integer.
    """
    _validate_order(order)
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )
    _require_coord(da, dim)
    axis = da.get_axis_num(dim)
    step = _uniform_step(da[dim], rtol=uniform_rtol)
    if periodic:
        # The spherical backend has a hard check of its own (the grid must
        # span a full 360°), so this advisory only covers the cartesian and
        # uniform-rectilinear paths, where the period is implicit.
        _warn_duplicate_endpoint(da.values, axis=axis, dim=dim)
    raw = _difference_maybe_periodic(
        da.values,
        axis=axis,
        step_size=step,
        accuracy=accuracy,
        method=method,
        derivative=order,
        periodic=periodic,
    )
    return xr.DataArray(
        raw,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        name=_output_name(da, dim, order=order),
        attrs=dict(da.attrs),
    )


def cartesian_gradient(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    periodic: frozenset[Hashable] = frozenset(),
    uniform_rtol: float = 1e-6,
) -> xr.Dataset:
    """Gradient ``∇da`` on a uniform Cartesian grid.

    Args:
        da: Input scalar field.
        dims: Dimensions to differentiate against. Defaults to ``da.dims``.
        accuracy: Scalar (applied to every dim) or per-dim tuple matching
            the length of ``dims``.
        method: Forwarded to :func:`finitediffx.difference`.
        periodic: Set of dimension names to treat as wrapping.
        uniform_rtol: Forwarded to :func:`cartesian_partial`.

    Returns:
        Dataset with one DataArray per requested dim, keyed
        ``f"d{da.name or 'f'}_d{dim}"``.
    """
    target_dims = tuple(da.dims) if dims is None else tuple(dims)
    per_dim = _per_dim_accuracy(accuracy, target_dims)
    base = da.name or "f"
    out: dict[str, xr.DataArray] = {}
    for dim, acc in zip(target_dims, per_dim, strict=True):
        component = cartesian_partial(
            da,
            dim,
            accuracy=acc,
            method=method,
            periodic=dim in periodic,
            uniform_rtol=uniform_rtol,
        )
        out[f"d{base}_d{dim}"] = component
    return xr.Dataset(out)
