"""Tier-B Operators — carrier-aware einx dispatch and pattern presets.

`Einx` is the generic escape hatch: any einx function, any pattern,
with the spatial-survival rule from ``array.py`` deciding whether a
``GeoTensor`` carrier's metadata rides through (`wrap_like`) or the
result drops to a plain ``np.ndarray``. The presets wrap the patterns
we reach for constantly in RS pipelines — channel-order flips, per-band
reductions, and spatial pooling (the one op that *updates* the
geotransform instead of preserving or dropping it).

Optional extra: ``pip install 'geotoolz[einx]'``. Importing this module
without einx installed raises a friendly ``ImportError`` naming the
extra, mirroring ``geotoolz.patch_ops``.

Design notes (geotoolz issue #69):

- Arrays are unwrapped with ``np.asarray`` *before* reaching einx, so
  `GeoTensor.__array_ufunc__` never interacts with einx internals (Q6).
- Pattern validation is permissive at construction (einx itself
  validates at dispatch); only the op name and bracket balance are
  checked eagerly (Q2).
- ``einx.vmap`` and other callable-taking forms are not supported by
  `Einx` — configs must stay YAML-serializable (Q5 / design rule 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pipekit import Operator

from geotoolz._src.config import jsonable
from geotoolz._src.wrap import wrap_like
from geotoolz.einx._src.array import spatial_survives


try:
    import einx as _einx
except ImportError as _e:  # pragma: no cover - exercised when [einx] is missing
    raise ImportError(
        "geotoolz.einx requires the `einx` package. "
        "Install with `pip install 'geotoolz[einx]'` (or `pip install einx`)."
    ) from _e


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

#: einx entry points `Einx` refuses because they take callables (not
#: YAML-serializable) or don't return an array.
_UNSUPPORTED_OPS = frozenset({"vmap", "vmap_with_axis", "custom", "trace", "jit"})


def _resolve_op(op: str) -> Any:
    """Return the einx callable named ``op``, validating eagerly.

    Args:
        op: Name of an einx entry point (``"id"``, ``"mean"``, ``"sum"``,
            ``"dot"``, ``"rearrange"``, ...).

    Returns:
        The einx function.

    Raises:
        ValueError: Unknown, private, or unsupported op name.
    """
    if op.startswith("_") or op in _UNSUPPORTED_OPS:
        raise ValueError(f"Einx: op {op!r} is not supported")
    fn = getattr(_einx, op, None)
    if not callable(fn):
        raise ValueError(
            f"Einx: {op!r} is not an einx operation "
            "(expected e.g. 'id', 'mean', 'sum', 'max', 'dot', 'add')"
        )
    return fn


class Einx(Operator):
    """Apply any einx operation to the carrier, geo-aware.

    The pattern is analyzed once at construction with
    `geotoolz.einx._src.array.spatial_survives`: when the trailing two
    output axes are the bare spatial axes (default ``("y", "x")``) and
    no spatial axis is composed anywhere, a ``GeoTensor`` input comes
    back as a ``GeoTensor`` (transform / CRS / fill propagated via
    ``wrap_like``); otherwise the result is a plain ``np.ndarray``.
    Plain-array input always returns a plain array.

    Multi-input patterns work by passing extra arrays at call time —
    the first argument is the carrier, the rest are unwrapped to plain
    arrays.

    Args:
        op: einx entry-point name (``"id"``, ``"mean"``, ``"sum"``,
            ``"max"``, ``"dot"``, ``"add"``, ...). Callable-taking
            entry points (``"vmap"``, ...) are rejected — configs must
            stay YAML-serializable.
        pattern: The einx pattern string.
        spatial_axes: Names of the (row, column) axes in the pattern.
            Default ``("y", "x")``.
        **op_kwargs: Extra keyword arguments forwarded to the einx call
            (axis sizes like ``py=2``, ...). Must be JSON-serializable
            for ``get_config`` round-trips.

    Examples:
        >>> import geotoolz as gz
        >>> # Per-pixel mean over bands; spatial survives -> GeoTensor out.
        >>> mean_map = gz.Einx(op="mean", pattern="c y x -> y x")
        >>> # Matched-filter scoring against a signature matrix.
        >>> score = gz.Einx(op="dot", pattern="band y x, sig band -> sig y x")
        >>> scores = score(reflectance_gt, signatures)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        op: str,
        pattern: str,
        spatial_axes: tuple[str, str] | list[str] = ("y", "x"),
        **op_kwargs: Any,
    ) -> None:
        self.op = op
        self.pattern = pattern
        self.spatial_axes = tuple(spatial_axes)
        if len(self.spatial_axes) != 2:
            raise ValueError(
                f"Einx: spatial_axes must name exactly two axes; got {spatial_axes!r}"
            )
        self.op_kwargs = op_kwargs
        self._fn = _resolve_op(op)
        # Also validates bracket balance eagerly (raises ValueError).
        self._survives = spatial_survives(pattern, self.spatial_axes)

    def _apply(
        self, gt: GeoTensor | np.ndarray, *extra: GeoTensor | np.ndarray
    ) -> GeoTensor | np.ndarray:
        arrays = [np.asarray(a) for a in (gt, *extra)]
        out = np.asarray(self._fn(self.pattern, *arrays, **self.op_kwargs))
        if self._survives:
            return wrap_like(gt, out)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "pattern": self.pattern,
            "spatial_axes": list(self.spatial_axes),
            **jsonable(self.op_kwargs),
        }


class CHWtoHWC(Einx):
    """Reorder a ``(C, H, W)`` cube to channels-last ``(H, W, C)``.

    Display / ML-interop helper (matplotlib ``imshow``, most vision
    frameworks expect channels-last). The output is always a plain
    ``np.ndarray``: with channels trailing, the spatial axes no longer
    sit in the trailing-two positions a ``GeoTensor`` requires.

    Examples:
        >>> import geotoolz as gz
        >>> rgb = gz.CHWtoHWC()(composite_gt)  # (H, W, C) ndarray  # doctest: +SKIP
    """

    def __init__(self) -> None:
        super().__init__(op="id", pattern="c y x -> y x c")

    def get_config(self) -> dict[str, Any]:
        return {}


class HWCtoCHW(Einx):
    """Reorder a channels-last ``(H, W, C)`` array to ``(C, H, W)``.

    Inverse of `CHWtoHWC`, for bringing external channels-last arrays
    into geotoolz's channel-first convention. Input is typically a
    plain array (channels-last carriers can't be GeoTensors), and the
    output is a plain array — rewrapping needs a reference carrier's
    metadata, e.g. ``wrap_like(reference_gt, out)`` downstream.

    Examples:
        >>> import geotoolz as gz
        >>> chw = gz.HWCtoCHW()(hwc_array)
    """

    def __init__(self) -> None:
        super().__init__(op="id", pattern="y x c -> c y x")
        # A channels-last input can't be a well-formed GeoTensor (its
        # trailing dims aren't the spatial grid), so never attempt a
        # metadata rewrap even though the output pattern ends in `y x`.
        self._survives = False

    def get_config(self) -> dict[str, Any]:
        return {}


class PerBandReduce(Einx):
    """Reduce each band over its spatial extent to one scalar.

    ``(C, H, W)`` in, ``(C,)`` out — band-wise statistics for QA
    summaries, normalization fitting, or feature vectors. Spatial
    structure is consumed, so the result is always a plain
    ``np.ndarray`` regardless of carrier.

    Args:
        reduce: einx reduction name: ``"mean"`` (default), ``"sum"``,
            ``"max"``, ``"min"``, ``"std"``, ``"var"``, ...

    Examples:
        >>> import geotoolz as gz
        >>> band_means = gz.PerBandReduce()(reflectance_gt)  # (C,)  # doctest: +SKIP
    """

    def __init__(self, *, reduce: str = "mean") -> None:
        super().__init__(op=reduce, pattern="c y x -> c")

    def get_config(self) -> dict[str, Any]:
        return {"reduce": self.op}


class SpatialPool(Operator):
    """Downsample the spatial grid by integer factors, updating the transform.

    Non-overlapping block pooling via einx
    (``"c (y py) (x px) -> c y x"``). This is the one einx preset that
    *changes* the geotransform rather than preserving or dropping it: a
    ``GeoTensor`` input returns a ``GeoTensor`` whose pixel size is
    scaled by the pool factors (same origin, coarser grid). Plain-array
    input returns a plain array. Accepts ``(H, W)`` or ``(C, H, W)``.

    Args:
        reduce: einx reduction applied per block: ``"mean"`` (default),
            ``"max"``, ``"min"``, ``"sum"``, ...
        factor: Pool factor — one int for square blocks or a
            ``(row, col)`` pair. Spatial dims must divide evenly;
            otherwise a ``ValueError`` names the offending axis (crop
            or pad first, e.g. with ``geom.CropTo`` / ``geom.PadTo``).

    Examples:
        >>> import geotoolz as gz
        >>> coarse = gz.SpatialPool(reduce="mean", factor=4)(scene_gt)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        reduce: str = "mean",
        factor: int | tuple[int, int] | list[int] = 2,
    ) -> None:
        self.reduce = reduce
        fy, fx = (factor, factor) if isinstance(factor, int) else tuple(factor)
        if fy < 1 or fx < 1:
            raise ValueError(f"SpatialPool: factors must be >= 1; got {factor!r}")
        self.factor = (int(fy), int(fx))
        self._fn = _resolve_op(reduce)

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        fy, fx = self.factor
        h, w = arr.shape[-2:]
        if h % fy or w % fx:
            raise ValueError(
                f"SpatialPool: spatial shape ({h}, {w}) is not divisible by "
                f"factor {self.factor}; crop or pad first"
            )
        pattern = (
            "(y py) (x px) -> y x" if arr.ndim == 2 else "c (y py) (x px) -> c y x"
        )
        out = np.asarray(self._fn(pattern, arr, py=fy, px=fx))
        transform = getattr(gt, "transform", None)
        if transform is None:
            return out
        pooled_transform = transform * type(transform).scale(fx, fy)
        from georeader.geotensor import GeoTensor as _GeoTensor

        return _GeoTensor(
            out,
            transform=pooled_transform,
            crs=gt.crs,
            fill_value_default=gt.fill_value_default,
            attrs=gt.attrs,
        )

    def get_config(self) -> dict[str, Any]:
        return {"reduce": self.reduce, "factor": list(self.factor)}
