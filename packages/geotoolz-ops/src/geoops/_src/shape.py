"""Shared shape-normalisation primitives.

A `geoarray.GeoArray` is 2-D to 4-D with dims
``("time", "band", "y", "x")[-ndim:]``: the spatial axes are always the
trailing two and the band axis, when there is one, is always ``-3``
(never ``0`` -- on a ``(T, C, H, W)`` stack axis ``0`` is time). The
helpers here are the one implementation of that convention:

* :func:`band_axis` / :data:`BAND_AXIS` -- where the bands live;
* :func:`band_dim` / :func:`dim_axis` -- an operator's ``dim=`` (and the
  deprecated ``axis=``), and the position of that dim in a carrier, read
  from its stored dims;
* :func:`require_ndim` -- an early, operator-named rank check for
  operators that do not support every rank;
* :func:`keep_band_axis` -- keeps the time axis of a band-collapsing
  result on a 4-D input (``(T, H, W)`` -> ``(T, 1, H, W)``) so it is not
  reinterpreted as ``(band, y, x)``;
* :func:`map_frames` / :func:`over_frames` -- apply a per-scene function
  (or an operator's ``_apply``) to each frame of a ``(T, C, H, W)`` stack
  and restack the results;
* :func:`single_band` -- the 2-D view of a single-band map;
* :func:`gather_inputs` -- the positional inputs of an N-ary reducer,
  ``op(a, b, ...)`` or ``op([a, b, ...])``.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Final

import numpy as np
from geoarray import GeoArray
from jaxtyping import Shaped


__all__ = [
    "BAND_AXIS",
    "DIMS",
    "band_axis",
    "band_dim",
    "dim_axis",
    "gather_inputs",
    "keep_band_axis",
    "map_frames",
    "over_frames",
    "require_ndim",
    "single_band",
]

#: Position of the band axis in a 3-D ``(C, H, W)`` or 4-D ``(T, C, H, W)``
#: carrier.
BAND_AXIS: Final = -3

_LAYOUTS: Final = {2: "(H, W)", 3: "(C, H, W)", 4: "(T, C, H, W)"}

#: The 0.2 layout a plain array is read in: ``DIMS[-ndim:]``.
DIMS: Final = ("time", "band", "y", "x")


def band_axis(arr: Any) -> int:
    """Axis holding the bands of a 3-D or 4-D carrier: always ``-3``.

    Args:
        arr: A ``(C, H, W)`` or ``(T, C, H, W)`` carrier (any array-like;
            only its rank is read).

    Returns:
        ``-3``.

    Raises:
        ValueError: If ``arr`` has fewer than three dimensions -- a 2-D
            ``(H, W)`` map has no band axis (it is a single band; callers
            that accept one handle it explicitly).

    Examples:
        >>> band_axis(np.zeros((2, 3, 4, 4)))
        -3
        >>> np.zeros((2, 3, 4, 4)).shape[band_axis(np.zeros((2, 3, 4, 4)))]
        3
    """
    ndim = np.ndim(arr)
    if ndim < 3:
        raise ValueError(
            f"a {ndim}-D array has no band axis; expected (C, H, W) or (T, C, H, W)"
        )
    return BAND_AXIS


def band_dim(
    dim: str | None,
    axis: int | None,
    name: str,
    *,
    legacy: tuple[int, ...] = (BAND_AXIS,),
) -> str | None:
    """The dim an operator works over, from ``dim=`` or the deprecated ``axis=``.

    ``axis=`` keeps working for one release: a band position in ``legacy``
    (``-3``, or ``0`` for the per-frame skimage wrappers) maps to
    ``"band"`` with a `DeprecationWarning`.

    Args:
        dim: The ``dim=`` argument.
        axis: The deprecated ``axis=`` argument (``None`` when not given).
        name: Operator name for the messages.
        legacy: The old band positions that map to ``"band"``.

    Returns:
        The dim name, or ``None`` when the operator allows "no band dim".

    Raises:
        ValueError: ``axis`` is not one of the ``legacy`` band positions, or
            both ``axis`` and a non-default ``dim`` are given.

    Examples:
        >>> band_dim("band", None, "NDVI")
        'band'
    """
    if axis is None:
        return dim
    warnings.warn(
        f"{name}(axis=...) is deprecated: pass dim='band' (the band dim's name). "
        "axis= is removed in the next minor release.",
        DeprecationWarning,
        stacklevel=3,
    )
    if dim != "band":
        raise ValueError(f"{name}: pass dim= or the deprecated axis=, not both")
    if axis not in legacy:
        raise ValueError(
            f"{name}: axis={axis} is not the band axis; pass dim= with the name "
            "of the dim to work over"
        )
    return "band"


def dim_axis(x: Any, dim: str, *, name: str) -> int:
    """The (negative) position of ``dim`` in ``x``.

    A `GeoArray` answers from its stored dims; a plain array is read in the
    0.2 layout, ``DIMS[-ndim:]``. A 2-D input without a band dim (a
    single-band map) answers ``-3``, the 0.2 band position, as before.

    Args:
        x: The carrier, or a plain array.
        dim: The dim name.
        name: Operator name for the message.

    Returns:
        The axis, counted from the end (``-3`` for the band dim of a
        ``("band", "y", "x")`` carrier), so it also holds for a frame of a
        stack.

    Raises:
        ValueError: No ``dim`` in ``x`` (other than the band dim of a 2-D
            map); the message lists its dims.

    Examples:
        >>> dim_axis(np.zeros((2, 3, 4, 4)), "band", name="NDVI")
        -3
    """
    ndim = int(np.ndim(x))
    dims = x.dims if isinstance(x, GeoArray) else DIMS[-ndim:]
    if dim in dims:
        return dims.index(dim) - len(dims)
    if dim == "band" and ndim < 3:
        # a single-band map: the 0.2 band position, which callers that
        # accept a 2-D input never index with
        return BAND_AXIS
    if isinstance(x, GeoArray):
        raise ValueError(f"{name}: the carrier has no {dim!r} dim; its dims are {dims}")
    raise ValueError(f"{name}: a {np.ndim(x)}-D array has no {dim!r} dim ({dims})")


def _describe_ranks(allowed: tuple[int, ...]) -> str:
    parts = [
        f"{n}-D {_LAYOUTS[n]}" if n in _LAYOUTS else f"{n}-D" for n in sorted(allowed)
    ]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]


def require_ndim(arr: Any, allowed: int | Iterable[int], name: str) -> int:
    """Raise a clear ``ValueError`` unless ``arr`` has an accepted rank.

    Args:
        arr: The operator input (any array-like; only its rank is read).
        allowed: Accepted number(s) of dimensions.
        name: Operator name, placed at the start of the message so the
            failure is attributable (e.g. ``"OpticalFlowTVL1"``).

    Returns:
        ``arr``'s number of dimensions.

    Raises:
        ValueError: If ``arr.ndim`` is not in ``allowed``. The message names
            the operator, the accepted ranks and layouts, and the shape.

    Examples:
        >>> require_ndim(np.zeros((3, 4, 4)), (2, 3), "PhaseAlign")
        3
        >>> require_ndim(np.zeros((2, 3, 4, 4)), (2, 3), "PhaseAlign")
        Traceback (most recent call last):
        ...
        ValueError: PhaseAlign accepts 2-D (H, W) or 3-D (C, H, W) input; ...
    """
    ranks = (allowed,) if isinstance(allowed, int) else tuple(allowed)
    ndim = int(np.ndim(arr))
    if ndim not in ranks:
        raise ValueError(
            f"{name} accepts {_describe_ranks(ranks)} input; got a {ndim}-D "
            f"array of shape {tuple(np.shape(arr))}"
        )
    return ndim


def keep_band_axis(out: Any, ref: Any) -> Any:
    """Keep the time axis of a band-collapsing result on a 4-D input.

    Operators that collapse the band axis (a spectral index, a score map)
    turn ``(C, H, W)`` into ``(H, W)``. The same reduction over a
    ``(T, C, H, W)`` stack gives ``(T, H, W)``, which every carrier would
    read as ``(band, y, x)``; re-inserting a singleton band axis keeps it a
    time series of single-band maps.

    Args:
        out: The reduced result.
        ref: The operator input.

    Returns:
        ``out`` with a singleton axis inserted at ``-3`` when ``ref`` is
        4-D and ``out`` is 3-D; otherwise ``out`` unchanged.

    Examples:
        >>> keep_band_axis(np.zeros((2, 4, 4)), np.zeros((2, 3, 4, 4))).shape
        (2, 1, 4, 4)
        >>> keep_band_axis(np.zeros((4, 4)), np.zeros((3, 4, 4))).shape
        (4, 4)
    """
    if np.ndim(ref) == 4 and np.ndim(out) == 3:
        return np.expand_dims(np.asarray(out), BAND_AXIS)
    return out


def _is_time_stack(x: Any) -> bool:
    """A time series: a `GeoArray` led by a ``time`` dim, or a 4-D plain array."""
    if isinstance(x, GeoArray):
        return x.ndim >= 3 and x.dims[0] == "time"
    return np.ndim(x) == 4


def _frame(x: Any, t: int) -> Any:
    if isinstance(x, GeoArray):
        return x.isel(time=t)
    return np.asarray(x)[t]


def map_frames(fn: Callable[[Any], Any], x: Any, *, name: str) -> Any:
    """Apply a per-scene function to every frame of a ``(T, C, H, W)`` stack.

    For operators whose semantics are per scene (a segmentation, a QA
    decode, a colour composite): the result is exactly the per-frame
    results restacked along a new leading time axis. A time series is a
    `GeoArray` whose first dim is named ``time`` (``("time", "band", "y",
    "x")`` or a single-band ``("time", "y", "x")``), or a 4-D plain array.
    On a stack with bands, a 2-D per-frame result gains a singleton band
    axis (``(T, 1, H, W)``, see :func:`keep_band_axis`); a single-band
    series stays ``("time", "y", "x")``.

    Args:
        fn: Function of one ``(C, H, W)`` frame returning an ``(H, W)`` or
            ``(C', H', W')`` carrier (GeoArray or ndarray).
        x: The input. Only a time series is split; anything else is passed
            to ``fn`` unchanged.
        name: Operator name for error messages.

    Returns:
        The ``(T, C', H', W')`` restacked result, a GeoArray (georeferenced
        like the first frame's result, with its attrs and fill value) when
        the per-frame results are GeoArrays, else an ndarray.

    Raises:
        ValueError: If a per-frame result is not a 2-D / 3-D array, or the
            frames' results differ in shape or grid.
    """
    if not _is_time_stack(x):
        return fn(x)
    outs = [fn(_frame(x, t)) for t in range(np.shape(x)[0])]
    arrays = []
    for out in outs:
        if not isinstance(out, np.ndarray | GeoArray) or out.ndim not in (2, 3):
            raise ValueError(
                f"{name} returns {type(out).__name__} per frame, which cannot be "
                "restacked into a (T, C, H, W) time series; apply it to each "
                "frame separately"
            )
        arr = np.asarray(out)
        arrays.append(arr[None] if arr.ndim == 2 else arr)
    first = outs[0]
    if len({a.shape for a in arrays}) != 1 or any(
        isinstance(o, GeoArray) != isinstance(first, GeoArray)
        or (
            isinstance(o, GeoArray)
            and not np.array_equal(np.asarray(o.transform), np.asarray(first.transform))
        )
        for o in outs[1:]
    ):
        raise ValueError(
            f"{name} produced per-frame results on different grids; they cannot "
            "be restacked into a (T, C, H, W) time series"
        )
    stacked = np.stack(arrays)
    if not isinstance(first, GeoArray):
        return stacked
    # a 2-D result keeps a singleton band axis on a stack that has bands
    # (#147); a single-band ("time", "y", "x") series stays one
    has_bands = not isinstance(x, GeoArray) or "band" in x.dims
    if first.ndim == 2 and not has_bands:
        stacked = stacked[:, 0]
    frame_dims = (
        first.dims if first.ndim == 3 or not has_bands else ("band", *first.dims)
    )
    # each frame lost the time coordinate; the frames are in order, so the
    # input's comes back with the restacked time dim
    time = tuple(c for c in getattr(x, "coords", ()) if c.dim == "time")
    return first.with_data(
        stacked,
        attrs=first.attrs,
        dims=("time", *frame_dims),
        coords=time + tuple(c for c in first.coords if c.dim in frame_dims),
    )


def over_frames(apply: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate an operator's ``_apply`` to run per frame on a time series.

    ``@over_frames`` on ``def _apply(self, x, ...)`` routes a time series
    (a carrier led by a ``time`` dim, or a 4-D plain array) through
    :func:`map_frames` (each frame handled by the undecorated ``_apply``,
    named after the operator class in errors); other inputs are
    unaffected. The wrapper keeps the
    wrapped signature, so pipekit's input introspection is unchanged.

    Args:
        apply: The ``_apply`` method; its first positional argument after
            ``self`` is the carrier.

    Returns:
        The wrapped method.
    """

    @functools.wraps(apply)
    def wrapper(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        return map_frames(
            lambda frame: apply(self, frame, *args, **kwargs),
            x,
            name=type(self).__name__,
        )

    return wrapper


def single_band(
    values: Shaped[np.ndarray, "h w"] | Shaped[np.ndarray, "1 h w"],
    *,
    name: str = "input",
) -> Shaped[np.ndarray, "h w"]:
    """Return the 2-D view of a single-band array.

    Args:
        values: A ``(H, W)`` array, or a ``(1, H, W)`` cube whose leading
            band axis is squeezed away. Any array-like is accepted.
        name: Label used in the error message so callers can attribute
            the failure to their own operator (e.g. ``"Otsu"``).

    Returns:
        The ``(H, W)`` array. No copy is made for ndarray input.

    Raises:
        ValueError: If ``values`` is neither ``(H, W)`` nor ``(1, H, W)``.
    """
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"{name} expects a single-band map with shape (H, W) or (1, H, W); "
        f"got shape {arr.shape}"
    )


def _is_pair(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def gather_inputs(
    inputs: tuple[Any, ...],
    name: str,
    *,
    pairs: bool = False,
    split_stack: bool = False,
) -> list[Any]:
    """The inputs of an N-ary reducer called as ``op(a, b, ...)`` or ``op([a, b])``.

    The multi-input convention (#141): N-ary reducers take their carriers
    as positional arguments, so they wire into a ``pipekit.Graph`` as
    ``op(Input("a"), Input("b"))``; a single list / tuple of them is
    accepted too. With ``pairs=True`` each input is a ``(carrier, extra)``
    pair, and a lone 2-tuple is one pair, not a sequence of two inputs
    (unless both of its items are pairs themselves). With
    ``split_stack=True`` a lone 4-D ``(T, C, H, W)`` carrier is a time
    stack whose frames are the inputs.

    Args:
        inputs: The ``*args`` tuple of the operator's ``_apply``.
        name: Operator name for error messages.
        pairs: Whether each input is a ``(carrier, extra)`` pair.
        split_stack: Whether a lone 4-D carrier is split into its frames.

    Returns:
        The inputs as a list.

    Raises:
        ValueError: If there are no inputs.
    """
    if len(inputs) == 1 and isinstance(inputs[0], list | tuple):
        only = inputs[0]
        if not pairs or not _is_pair(only) or all(_is_pair(item) for item in only):
            inputs = tuple(only)
    elif split_stack and len(inputs) == 1 and np.ndim(inputs[0]) == 4:
        inputs = tuple(_frame(inputs[0], t) for t in range(np.shape(inputs[0])[0]))
    if not inputs:
        raise ValueError(f"{name} needs at least one input")
    return list(inputs)
