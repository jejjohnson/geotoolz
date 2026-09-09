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
from collections.abc import Callable, Hashable
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


# Order matters: the widest stencil first, so a point keeps the centred
# estimate whenever its support is entirely valid.
_ADAPTIVE_METHODS = ("central", "forward", "backward")


def _validate_nan_policy(nan_policy: str, method: str) -> None:
    """Guard the NaN policy and its interaction with ``method``.

    Raises:
        ValueError: for an unknown policy, or for a non-central ``method``
            combined with ``"adaptive"`` — the fallback chain *is* method
            selection, so the two would contradict each other.
    """
    if nan_policy not in ("propagate", "adaptive"):
        raise ValueError(
            f"Unknown nan_policy {nan_policy!r}; expected 'propagate' or 'adaptive'."
        )
    if nan_policy == "adaptive" and method != "central":
        raise ValueError(
            f"nan_policy='adaptive' selects the stencil per point, so it "
            f"cannot be combined with method={method!r}. Leave method at "
            "'central' (adaptive falls back to forward/backward itself), or "
            "use nan_policy='propagate' to force one method everywhere."
        )


def _difference_adaptive(
    values: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    derivative: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Differentiate with the widest stencil that fits on finite data.

    Near a land mask a centred stencil reaches into NaN and the result is
    lost, taking a strip of valid water with it. This runs the same
    kernel once per candidate stencil and keeps, per point, the first
    result that came back finite.

    No mask arithmetic is needed because NaN is its own support detector:
    a stencil that reads a NaN yields NaN, including through a zero
    centre coefficient, since ``0 * nan`` is ``nan``. That also means the
    fallback never has to know how wide ``finitediffx`` built its
    stencil — an internal rule this package deliberately does not
    duplicate.

    The input mask is re-imposed at the end. Land is NaN because there is
    no data there, which is a property of the answer rather than of the
    arithmetic that happened to produce it.
    """

    def candidate(method: str, acc: int) -> Float[np.ndarray, "*shape"]:
        return _difference_maybe_periodic(
            values,
            axis=axis,
            step_size=step_size,
            accuracy=acc,
            method=method,
            derivative=derivative,
            periodic=periodic,
        )

    return _select_by_finiteness(candidate, values, accuracy=accuracy)


def _select_by_finiteness(
    candidate: Callable[[str, int], Float[np.ndarray, "*shape"]],
    values: Float[np.ndarray, "*shape"],
    *,
    accuracy: int,
) -> Float[np.ndarray, "*shape"]:
    """Keep, per point, the first candidate stencil that came back finite.

    Candidates are tried widest-first: every method in
    :data:`_ADAPTIVE_METHODS` at the requested ``accuracy``, then the same
    sweep at each lower accuracy down to 1. Dropping the accuracy matters
    for valid regions narrower than the requested one-sided stencil — a
    two-cell channel between two masked cells supports a first-order
    difference but nothing wider, and without the descent it would vanish
    precisely when the caller asked for *more* accuracy.

    ``candidate`` is called lazily, so a field the centred stencil already
    covers still costs exactly one kernel evaluation, and the extra
    accuracies are only reached while finite input cells remain
    unresolved.

    A gap that sits on the input mask can never be filled — no stencil
    reaches data there — so only gaps that coincide with finite input
    keep the search going.

    A stencil wider than the axis itself is skipped rather than fatal, so
    a short axis descends the same way an embedded narrow strip does:
    without that, two samples with ``accuracy=2`` would raise while the
    same two samples embedded in a longer masked axis resolved fine. If
    no accuracy fits at all the rejection is re-raised, so a genuinely
    undifferentiable axis still reports itself.

    Args:
        candidate: Builds the differenced array for a method and accuracy.
        values: The input field, supplying the mask to re-impose.
        accuracy: Highest stencil accuracy to try.

    Returns:
        The combined result, NaN wherever ``values`` is not finite.

    Raises:
        ValueError: If no accuracy down to 1 fits the axis. ``finitediffx``
            also surfaces this as a :exc:`TypeError` from JAX's slicing at
            some sizes, which is re-raised unchanged.
    """
    finite_input = np.isfinite(values)
    out: Float[np.ndarray, "*shape"] | None = None
    too_short: Exception | None = None
    for acc in range(accuracy, 0, -1):
        for method in _ADAPTIVE_METHODS:
            try:
                result = candidate(method, acc)
            except (ValueError, TypeError) as exc:
                # This stencil is wider than the axis; a narrower one may fit.
                too_short = exc
                continue
            out = result if out is None else np.where(~np.isfinite(out), result, out)
            if not (~np.isfinite(out) & finite_input).any():
                return np.where(finite_input, out, np.nan)
    if out is None:
        assert too_short is not None  # the only way every candidate was skipped
        raise too_short
    return np.where(finite_input, out, np.nan)


def _difference_adaptive_chain(
    values: Float[np.ndarray, "*shape"],
    coord_values: Float[np.ndarray, " n"],
    *,
    axis: int,
    accuracy: int,
) -> Float[np.ndarray, "*shape"]:
    """Adaptive first derivative against a non-uniform coordinate.

    The rectilinear backend differentiates on the index grid and divides
    by ``dx/di``. Numerator and denominator have to come from the *same*
    stencil at every point: a one-sided ``df/di`` over a centred ``dx/di``
    mixes two different local approximations, and on a stretched grid
    that is silently wrong — for ``x = [0, 1, 3, ...]`` and ``f = x`` with
    ``f[0]`` masked it returns ``(3-1)/((3-0)/2) = 4/3`` instead of 1.

    So the quotient — not just the numerator — is the candidate that
    :func:`_select_by_finiteness` picks between. The coordinate is finite
    and monotonic by the time we get here, so only the numerator's NaNs
    drive the selection.

    Args:
        values: Field to differentiate.
        coord_values: The 1-D coordinate along ``axis``.
        axis: Axis of ``values`` to differentiate.
        accuracy: Stencil accuracy order.

    Returns:
        ``df/dx``, NaN wherever ``values`` is not finite.
    """
    shape = [1] * values.ndim
    shape[axis] = coord_values.size

    def candidate(method: str, acc: int) -> Float[np.ndarray, "*shape"]:
        df_di = _difference(
            values, axis=axis, step_size=1.0, accuracy=acc, method=method
        )
        dx_di = _difference(
            coord_values, axis=0, step_size=1.0, accuracy=acc, method=method
        )
        return df_di / dx_di.reshape(shape)

    return _select_by_finiteness(candidate, values, accuracy=accuracy)


def _difference_dispatch(
    values: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    step_size: float,
    accuracy: int,
    method: str,
    derivative: int = 1,
    periodic: bool = False,
    nan_policy: str = "propagate",
) -> Float[np.ndarray, "*shape"]:
    """Route to the plain, periodic, or NaN-adaptive stencil."""
    if nan_policy == "adaptive":
        return _difference_adaptive(
            values,
            axis=axis,
            step_size=step_size,
            accuracy=accuracy,
            derivative=derivative,
            periodic=periodic,
        )
    return _difference_maybe_periodic(
        values,
        axis=axis,
        step_size=step_size,
        accuracy=accuracy,
        method=method,
        derivative=derivative,
        periodic=periodic,
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
    nan_policy: str = "propagate",
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
        nan_policy: ``"propagate"`` (default) lets NaN spread through any
            stencil that touches it, so a masked cell costs a halo of
            roughly ``accuracy`` valid cells. ``"adaptive"`` instead
            falls back per point to the widest stencil lying wholly on
            finite data, recovering those cells at one order lower
            accuracy; masked cells stay NaN.
        uniform_rtol: Relative tolerance for the uniform-spacing check.

    Returns:
        DataArray with the same dims/coords as ``da`` and a default
        name of ``f"d{da.name}_d{dim}"`` (``f"d2{da.name}_d{dim}2"`` at
        ``order=2``).

    Raises:
        ValueError: if ``dim`` is absent from ``da.dims``, carries no
            coordinate, ``order`` is not a positive integer, or
            ``nan_policy`` is unknown or contradicts ``method``.
    """
    _validate_order(order)
    _validate_nan_policy(nan_policy, method)
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
    raw = _difference_dispatch(
        da.values,
        axis=axis,
        step_size=step,
        accuracy=accuracy,
        method=method,
        derivative=order,
        periodic=periodic,
        nan_policy=nan_policy,
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
    nan_policy: str = "propagate",
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
        nan_policy: ``"propagate"`` or ``"adaptive"``; see
            :func:`cartesian_partial`.
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
            nan_policy=nan_policy,
            uniform_rtol=uniform_rtol,
        )
        out[f"d{base}_d{dim}"] = component
    return xr.Dataset(out)
