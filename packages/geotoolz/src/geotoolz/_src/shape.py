"""Shared shape-normalisation primitives.

georeader's ``GeoTensor`` is 2-D to 4-D with dims
``("time", "band", "y", "x")[-ndim:]``: the spatial axes are always the
trailing two and the band axis, when there is one, is always ``-3``
(never ``0`` -- on a ``(T, C, H, W)`` stack axis ``0`` is time). The
helpers here are the one implementation of that convention:

* :func:`band_axis` / :data:`BAND_AXIS` -- where the bands live;
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
from collections.abc import Callable, Iterable
from typing import Any, Final

import numpy as np
from jaxtyping import Shaped


__all__ = [
    "BAND_AXIS",
    "band_axis",
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


def _frame(x: Any, t: int) -> Any:
    isel = getattr(x, "isel", None)
    if isel is not None and getattr(x, "dims", None) == ("time", "band", "y", "x"):
        return isel({"time": t})
    return np.asarray(x)[t]


def map_frames(fn: Callable[[Any], Any], x: Any, *, name: str) -> Any:
    """Apply a per-scene function to every frame of a ``(T, C, H, W)`` stack.

    For operators whose semantics are per scene (a segmentation, a QA
    decode, a colour composite): the result is exactly the per-frame
    results restacked along a new leading time axis. A 2-D per-frame
    result gains a singleton band axis (``(T, 1, H, W)``, see
    :func:`keep_band_axis`).

    Args:
        fn: Function of one ``(C, H, W)`` frame returning an ``(H, W)`` or
            ``(C', H', W')`` carrier (GeoTensor or ndarray).
        x: The input. Only a 4-D input is split; anything else is passed to
            ``fn`` unchanged.
        name: Operator name for error messages.

    Returns:
        The ``(T, C', H', W')`` restacked result, a GeoTensor (georeferenced
        like the first frame's result, with a copy of its attrs and its
        fill value) when the per-frame results are GeoTensors, else an
        ndarray.

    Raises:
        ValueError: If a per-frame result is not a 2-D / 3-D array, or the
            frames' results differ in shape or grid.
    """
    if np.ndim(x) != 4:
        return fn(x)
    outs = [fn(_frame(x, t)) for t in range(np.shape(x)[0])]
    arrays = []
    for out in outs:
        if not isinstance(out, np.ndarray) or out.ndim not in (2, 3):
            raise ValueError(
                f"{name} returns {type(out).__name__} per frame, which cannot be "
                "restacked into a (T, C, H, W) time series; apply it to each "
                "frame separately"
            )
        arr = np.asarray(out)
        arrays.append(arr[None] if arr.ndim == 2 else arr)
    first = outs[0]
    if len({a.shape for a in arrays}) != 1 or any(
        getattr(o, "transform", None) != getattr(first, "transform", None)
        for o in outs[1:]
    ):
        raise ValueError(
            f"{name} produced per-frame results on different grids; they cannot "
            "be restacked into a (T, C, H, W) time series"
        )
    stacked = np.stack(arrays)
    if not hasattr(first, "transform"):
        return stacked
    from georeader.geotensor import GeoTensor

    return GeoTensor(
        stacked,
        transform=first.transform,
        crs=first.crs,
        fill_value_default=first.fill_value_default,
        attrs=dict(first.attrs or {}),
    )


def over_frames(apply: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate an operator's ``_apply`` to run per frame on 4-D input.

    ``@over_frames`` on ``def _apply(self, x, ...)`` routes a
    ``(T, C, H, W)`` input through :func:`map_frames` (each frame handled
    by the undecorated ``_apply``, named after the operator class in
    errors); 2-D and 3-D inputs are unaffected. The wrapper keeps the
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
