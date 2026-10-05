"""RS-safe augmentation operators.

Seeding semantics (shared by every operator here):

- ``op = Op(seed=s)`` holds a private ``np.random.Generator`` seeded from
  ``s`` (created lazily on the first call). Each call draws from it, so
  successive calls apply *different* augmentations, while two operators
  built with the same ``s`` produce the same sequence of draws. With
  ``seed=None`` the generator is seeded from OS entropy.
- ``op(x, seed=k)`` is a deterministic one-off draw from a fresh generator
  seeded with ``k``; it does not advance the operator's own stream.
- ``get_config`` only ever reports the constructor ``seed``: a reloaded
  (or freshly built) operator restarts its stream from the beginning.
- The generator state travels with ``copy`` / ``pickle``, so replicas of an
  already-seeded operator (e.g. dataloader workers) repeat each other's
  draws unless each is given its own ``seed``.

Time stacks: a ``(T, C, H, W)`` input is one sample. The band axis is
``-3``, so per-band draws (factors, sigmas, dropout, permutations) are made
once per band and shared by every frame, and geometric draws (flips,
rotations, crops, shifts) move all frames together; per-pixel noise is
drawn for the full stack. Nodata is judged per frame.

Nodata semantics (see :mod:`geotoolz._src.valid`): radiometric operators
never perturb fill pixels -- every pixel invalid under
:func:`~geotoolz._src.valid.valid_pixels` (non-finite, or equal to the
carrier's ``fill_value_default`` in any band) holds the output's fill value
(``NaN`` for plain float arrays), and any statistic they compute (a band
mean, a brightness percentile) is taken over valid pixels only. Nodata
handling never consumes random draws, so a given seed yields the same draws
with or without fill pixels.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine
from jaxtyping import Shaped
from pipekit import Operator, Sequential
from pipekit._base.operator import require_operator
from rasterio.windows import Window

from geotoolz._src.bands import band_count, resolve_bands
from geotoolz._src.config import (
    as_tuple,
    jsonable,
    mapping_from_pairs,
    mapping_to_pairs,
)
from geotoolz._src.geo import require_geotensor, require_grid_match
from geotoolz._src.shape import BAND_AXIS, gather_inputs
from geotoolz._src.valid import carrier_fill_value, restore_fill, valid_pixels
from geotoolz._src.wrap import adopt_attrs, wrap_like
from geotoolz.augment._src.array import (
    cloud_alpha,
    rayleigh_weights,
    rot90_transform,
    sun_angle_scale,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


Range = tuple[float, float]
ScalarOrRange = float | Range
BRIGHT_CLOUD_PERCENTILE = 98.0


def _call_rng(op: Operator, seed: int | None) -> np.random.Generator:
    """Return the generator that drives one ``op`` call.

    A per-call ``seed`` gives a fresh, one-off generator (a deterministic
    draw that leaves the operator's own stream untouched). Otherwise the
    draw comes from a generator held privately on the instance, created
    lazily from ``op.seed`` on first use, so successive calls continue one
    stream: reproducible across instances built with the same seed, yet
    different from call to call. The generator lives outside the
    constructor parameters, so ``get_config`` stays the pure-JSON seed;
    a reloaded operator starts its stream from the beginning. Changing
    ``op.seed`` after construction restarts the stream from the new seed.
    """
    if seed is not None:
        return np.random.default_rng(seed)
    state = op.__dict__.get("_rng_state")
    if state is None or state[0] != op.seed:
        state = (op.seed, np.random.default_rng(op.seed))
        op.__dict__["_rng_state"] = state
    return state[1]


def _check_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_range(value: ScalarOrRange, name: str) -> None:
    """Validate a scalar-or-range parameter at construction time.

    For tuples, requires ``lo <= hi``. For scalars, no ordering check.
    """
    if isinstance(value, tuple):
        lo, hi = value
        if hi < lo:
            raise ValueError(f"{name} range must satisfy lo <= hi; got ({lo}, {hi})")


def _validate_probability_range(value: ScalarOrRange, name: str) -> None:
    """Validate a scalar-or-range that must lie inside ``[0, 1]``.

    Tuples must satisfy ``0 <= lo <= hi <= 1``; scalars must satisfy
    ``0 <= value <= 1``.
    """
    if isinstance(value, tuple):
        lo, hi = value
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"{name} range must satisfy 0 <= lo <= hi <= 1; got ({lo}, {hi})"
            )
    else:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}")


def _apply_accepts_seed(op: Operator) -> bool:
    """Return True if ``op._apply`` takes a ``seed`` keyword.

    The per-call ``seed`` is forwarded to ``_apply`` (via ``op(x, seed=...)``),
    so that signature -- not ``__init__`` -- decides whether it is accepted:
    an explicit ``seed`` parameter or a ``**kwargs`` catch-all.
    """
    try:
        params = inspect.signature(op._apply).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    param = params.get("seed")
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _sample_uniform(rng: np.random.Generator, value: ScalarOrRange, name: str) -> float:
    if isinstance(value, tuple):
        lo, hi = value
        if hi < lo:
            raise ValueError(f"{name} range must be ordered as (min, max).")
        return float(rng.uniform(lo, hi))
    return float(value)


def _sample_nonnegative(
    rng: np.random.Generator, value: ScalarOrRange, name: str
) -> float:
    sampled = _sample_uniform(rng, value, name)
    if sampled < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return sampled


def _band_count(arr: Shaped[np.ndarray, "*dims"]) -> int:
    """Number of bands: ``shape[-3]`` (a 2-D map is one band)."""
    return band_count(arr.shape)


def _band_shape(arr: Shaped[np.ndarray, "*dims"]) -> tuple[int, ...]:
    """Shape of a per-band factor broadcasting as ``(..., C, 1, 1)``.

    The band axis is ``-3`` (``(C, H, W)`` or ``(T, C, H, W)``), so one
    factor per band is shared by every frame of a time stack.
    """
    shape = [1] * arr.ndim
    if arr.ndim >= 3:
        shape[BAND_AXIS] = arr.shape[BAND_AXIS]
    return tuple(shape)


def _broadcast_valid(
    valid: np.ndarray, shape: tuple[int, ...]
) -> Shaped[np.ndarray, "*dims"]:
    """Broadcast an ``(H, W)`` / per-frame ``(T, H, W)`` mask against ``shape``."""
    if valid.ndim == 3 and len(shape) == 4:
        valid = valid[:, None]
    return np.broadcast_to(valid, shape)


def _cast_like(
    out: Shaped[np.ndarray, "*dims"], dtype: np.dtype[Any]
) -> Shaped[np.ndarray, "*dims"]:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_):
        # Pixel-rearranging ops (flip, crop, dropout, ...) keep a mask a mask;
        # radiometric math on a boolean carrier has no meaningful cast back.
        if out.dtype == np.bool_:
            return out
        raise TypeError(
            "radiometric augmentations are undefined for boolean carriers; "
            "cast the mask to a numeric dtype first."
        )
    if np.issubdtype(dtype, np.integer):
        if not np.issubdtype(out.dtype, np.integer):
            # Round to the nearest DN; a bare cast truncates (-0.5 DN bias).
            out = np.rint(out)
        info = np.iinfo(dtype.name)
        out = np.clip(out, info.min, info.max)
    return out.astype(dtype, copy=False)


def _valid(arr: Shaped[np.ndarray, "*dims"], gt: Any) -> np.ndarray | None:
    """Spatial validity mask of ``gt`` (``None`` when every pixel is valid).

    ``None`` flags the fast path: operators then run exactly the math they
    always did. Inputs with fewer than two dims have no pixel grid and are
    treated as all-valid. A ``(T, C, H, W)`` stack gets a per-frame
    ``(T, H, W)`` mask, so nodata in one frame does not blank the others.
    """
    if arr.ndim < 2:
        return None
    valid = valid_pixels(gt, keep_time=True)
    return None if valid.all() else valid


def _cast_and_wrap(
    gt: GeoTensor | np.ndarray,
    out: np.ndarray,
    valid: np.ndarray | None = None,
) -> GeoTensor | np.ndarray:
    """Cast ``out`` back to the input dtype and rewrap it like ``gt``.

    When ``valid`` is given, invalid pixels are reset to the output's fill
    value -- the fill ``wrap_like`` inherits from ``gt`` (``NaN`` for a plain
    float array) -- after the cast, so the fill survives clipping/rounding.
    """
    out = _cast_like(out, np.asarray(gt).dtype)
    if valid is not None:
        out = restore_fill(out, valid, carrier_fill_value(gt))
    return wrap_like(gt, out)


def _new_geotensor(
    gt: GeoTensor, out: np.ndarray, transform: Affine
) -> GeoTensor | np.ndarray:
    """Rewrap a pixel-rearranged ``out`` like ``gt`` on a new ``transform``."""
    return wrap_like(gt, _cast_like(out, np.asarray(gt).dtype), transform=transform)


class Compose(Sequential):
    """Apply augmentations sequentially behind a pipeline probability gate.

    A :class:`pipekit.Sequential` whose whole chain runs with probability
    ``p`` and whose children can be seeded from one top-level seed.

    Seeding follows the module contract. When ``Compose`` is seeded — by
    a per-call ``seed`` (one-off) or a constructor ``seed`` (its own
    stream, advancing each call) — it owns the randomness of the whole
    chain: it draws one child seed per augmentation and forwards it as
    that child's per-call ``seed``, so the same top-level seed always
    reproduces the same chain of inner draws and the children's own
    streams are left untouched. An unseeded ``Compose`` forwards nothing,
    so each child draws from its own stream (honouring any child
    ``seed``). A child receives a ``seed`` only when its ``_apply``
    accepts one (the per-call ``seed`` goes to ``_apply``, not
    ``__init__``), so deterministic operators and samplers whose seed is
    constructor-only are called plainly.

    ``get_config`` emits the children's nested configs (via
    :class:`~pipekit.Sequential`) under ``augmentations``. ``Compose`` is
    a container, so it is not ``forbid_in_yaml`` itself; a flagged child
    is found structurally by pipekit's walker.

    Composition: ``compose | op`` keeps the gate (``Sequential([compose,
    op])``), but pipekit flattens a ``Sequential`` on the right of ``|``,
    so ``op | compose`` would splice the children in *without* the gate —
    write ``Sequential([op, compose])`` instead. Slicing (``compose[1:]``)
    likewise returns an ungated ``Sequential`` of the children.

    The carrier is whatever the children accept: a plain ``np.ndarray``
    input passes through unchanged when the probability check skips the
    pipeline, and is otherwise handed to the children as-is.

    Args:
        augmentations: Operators applied in order.
        p: Probability of applying the whole pipeline. Default ``1.0``.
        seed: Seed of the pipeline's own stream (see above); ``None``
            leaves each child to its own stream.

    Raises:
        TypeError: If a child is not an ``Operator`` (e.g. a reloaded
            nested-config dict) or is a stateful operator.

    Examples:
        >>> import geotoolz as gz
        >>> pipe = gz.augment.Compose(
        ...     augmentations=[
        ...         gz.augment.RandomFlip(),
        ...         gz.augment.GaussianNoise(sigma=0.01),
        ...     ],
        ...     p=1.0,
        ...     seed=0,
        ... )
        >>> out = pipe(patch)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        augmentations: list[Operator],
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        _check_probability(p, "p")
        for i, op in enumerate(augmentations):
            # A YAML / Hydra loader hands back the nested {"class", "config"}
            # payloads as dicts; fail here rather than at apply time.
            require_operator(op, "Compose", f"augmentations[{i}]")
            if getattr(op, "_is_stateful", False):
                raise TypeError(
                    f"Compose.augmentations[{i}] is a stateful operator "
                    f"({type(op).__name__}); Compose does not thread state."
                )
        super().__init__(list(augmentations))
        self.p = p
        self.seed = seed

    @property
    def augmentations(self) -> list[Operator]:
        """The child operators, in application order."""
        return self.operators

    def _apply(  # ty: ignore[invalid-method-override]
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        seeded = seed is not None or self.seed is not None
        rng = _call_rng(self, seed)
        if rng.random() >= self.p:
            return gt

        out = gt
        child_seeds = rng.integers(0, np.iinfo(np.int64).max, len(self.operators))
        for op, child_seed in zip(self.operators, child_seeds, strict=True):
            if seeded and _apply_accepts_seed(op):
                out = op(out, seed=int(child_seed))
            else:
                out = op(out)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "augmentations": super().get_config()["operators"],
            "p": self.p,
            "seed": self.seed,
        }

    def __or__(self, other: Operator) -> Sequential:
        """``compose | op`` -> ``Sequential([compose, op])`` (the gate is kept)."""
        if not isinstance(other, Operator):
            return NotImplemented
        if isinstance(other, Sequential) and not isinstance(other, Compose):
            return Sequential([self, *other.operators])
        return Sequential([self, other])

    def __repr__(self) -> str:
        inner = ", ".join(repr(op) for op in self.operators)
        return f"Compose(augmentations=[{inner}], p={self.p!r}, seed={self.seed!r})"

    def describe(self, indent: int = 0) -> str:
        pad = "  " * indent
        body = super().describe(indent).split("\n", 1)
        head = f"{pad}Compose(p={self.p!r}, seed={self.seed!r}, ["
        return "\n".join([head, *body[1:]]) if len(body) > 1 else f"{pad}{self!r}"


class RandomFlip(Operator):
    """Randomly flip a GeoTensor horizontally and/or vertically.

    Preserves the CRS and updates the affine ``transform`` so the output
    pixel grid still maps to the same physical extent (identical
    ``bounds``/footprint; the mirrored axis is anchored at the far pixel
    edge). Each axis flips independently with probability
    ``p_horizontal`` / ``p_vertical``.

    The pixel math is metadata-free, so a plain ``np.ndarray`` input is
    also accepted and returns the flipped plain array (no transform
    bookkeeping).

    Args:
        p_horizontal: Probability of flipping the last (x) axis.
        p_vertical: Probability of flipping the second-to-last (y) axis.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        p_horizontal: float = 0.5,
        p_vertical: float = 0.5,
        seed: int | None = None,
    ) -> None:
        _check_probability(p_horizontal, "p_horizontal")
        _check_probability(p_vertical, "p_vertical")
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        arr = np.asarray(gt)
        out = arr
        transform = getattr(gt, "transform", None)
        height, width = arr.shape[-2], arr.shape[-1]

        if rng.random() < self.p_horizontal:
            out = np.flip(out, axis=-1)
            if transform is not None:
                transform = (
                    transform * Affine.translation(width, 0) * Affine.scale(-1, 1)
                )

        if rng.random() < self.p_vertical:
            out = np.flip(out, axis=-2)
            if transform is not None:
                transform = (
                    transform * Affine.translation(0, height) * Affine.scale(1, -1)
                )

        if out is arr:
            return gt
        # np.flip returns a view; copy so the output never aliases the input.
        out = out.copy()
        if transform is None:
            return out
        return _new_geotensor(gt, out, transform)


class RandomRotate90(Operator):
    """Randomly rotate by 90, 180, or 270 degrees.

    Uses ``np.rot90`` over the trailing two axes and composes the input
    ``transform`` with the matching rigid rotation so every output pixel
    still maps to its world location and the footprint is unchanged.

    The pixel math is metadata-free, so a plain ``np.ndarray`` input is
    also accepted and returns the rotated plain array (no transform
    bookkeeping).

    Args:
        p: Probability of rotating; when triggered, k is drawn uniformly
            from {1, 2, 3} quarter-turns.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomRotate90(p=1.0, seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(self, *, p: float = 0.75, seed: int | None = None) -> None:
        _check_probability(p, "p")
        self.p = p
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        if rng.random() >= self.p:
            return gt

        k = int(rng.integers(1, 4))
        arr = np.asarray(gt)
        # np.rot90 returns a view; copy so the output never aliases the input.
        out = np.rot90(arr, k=k, axes=(-2, -1)).copy()
        transform = getattr(gt, "transform", None)
        if transform is None:
            return out
        height, width = arr.shape[-2], arr.shape[-1]
        return _new_geotensor(gt, out, rot90_transform(transform, height, width, k))


class RandomCrop(Operator):
    """Randomly crop a spatial window and update the transform.

    Delegates to ``gt.isel`` so the output transform's translation reflects
    the crop origin (i.e. ``transform * (left, top)`` equals the new
    upper-left world coordinate). CRS, dtype and band metadata are
    preserved.

    The crop itself is pixel-space math, so a plain ``np.ndarray`` input
    is also accepted and returns the cropped plain array.

    Args:
        size: ``(height, width)`` of the cropped window in pixels.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomCrop(size=(3, 4), seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(self, *, size: tuple[int, int], seed: int | None = None) -> None:
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("size entries must be positive.")
        self.size = size
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        crop_h, crop_w = self.size
        arr = np.asarray(gt)
        height, width = arr.shape[-2], arr.shape[-1]
        if crop_h > height or crop_w > width:
            raise ValueError("size must fit within the GeoTensor spatial shape.")

        rng = _call_rng(self, seed)
        top = int(rng.integers(0, height - crop_h + 1))
        left = int(rng.integers(0, width - crop_w + 1))
        isel = getattr(gt, "isel", None)
        if isel is not None:
            return adopt_attrs(
                gt,
                isel({"y": slice(top, top + crop_h), "x": slice(left, left + crop_w)}),
            )
        return arr[..., top : top + crop_h, left : left + crop_w]

    def get_config(self) -> dict[str, Any]:
        return {"size": jsonable(self.size), "seed": self.seed}


class RandomShift(Operator):
    """Randomly shift a fixed-size spatial window in pixel units.

    Delegates to ``gt.read_from_window(..., boundless=True)`` so pixels
    shifted in from outside the original extent are padded with the
    carrier's ``fill_value_default`` and the transform tracks the new
    window origin. Geo-dependent: requires a georeferenced ``GeoTensor``
    input; a plain array raises ``TypeError``.

    Args:
        max_shift: ``(max_dy, max_dx)`` maximum absolute shift per axis,
            in pixels. Each shift is drawn uniformly from
            ``[-max, +max]``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(self, *, max_shift: tuple[int, int], seed: int | None = None) -> None:
        if max_shift[0] < 0 or max_shift[1] < 0:
            raise ValueError("max_shift entries must be non-negative.")
        self.max_shift = max_shift
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        require_geotensor(gt, "RandomShift")
        max_y, max_x = self.max_shift
        rng = _call_rng(self, seed)
        dy = int(rng.integers(-max_y, max_y + 1)) if max_y else 0
        dx = int(rng.integers(-max_x, max_x + 1)) if max_x else 0
        if dx == 0 and dy == 0:
            return gt
        window = Window(col_off=dx, row_off=dy, width=gt.width, height=gt.height)
        return adopt_attrs(gt, gt.read_from_window(window, boundless=True))

    def get_config(self) -> dict[str, Any]:
        return {"max_shift": jsonable(self.max_shift), "seed": self.seed}


class BrightnessJitter(Operator):
    """Scale reflectance values by a sampled scalar or per-band factor.

    Pure per-pixel math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind, cast back to the
    input dtype.

    Nodata: fill / non-finite pixels are left untouched and hold the
    output's fill value (see the module docstring).

    Args:
        factor: ``(lo, hi)`` range the multiplicative factor is drawn
            from. Default ``(0.9, 1.1)``.
        per_band: Draw an independent factor per band (the band
            axis, ``-3``) instead of one shared factor. Default ``True``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        factor: Range = (0.9, 1.1),
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        factor = as_tuple(factor)
        _validate_range(factor, "factor")
        self.factor = factor
        self.per_band = per_band
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        arr = np.asarray(gt)
        if self.per_band:
            factors = rng.uniform(self.factor[0], self.factor[1], _band_count(arr))
            factors = factors.reshape(_band_shape(arr))
        else:
            factors = _sample_uniform(rng, self.factor, "factor")
        return _cast_and_wrap(
            gt, arr.astype(np.float64, copy=False) * factors, _valid(arr, gt)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": jsonable(self.factor),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class ContrastJitter(Operator):
    """Scale deviations from the per-band spatial mean.

    Pure per-pixel math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind, cast back to the
    input dtype.

    Nodata: the per-band mean is taken over valid pixels only, and fill /
    non-finite pixels hold the output's fill value (see the module
    docstring).

    Args:
        factor: ``(lo, hi)`` range the contrast factor is drawn from.
            Default ``(0.9, 1.1)``.
        per_band: Draw an independent factor per band (the band
            axis, ``-3``) instead of one shared factor. Default ``True``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        factor: Range = (0.9, 1.1),
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        factor = as_tuple(factor)
        _validate_range(factor, "factor")
        self.factor = factor
        self.per_band = per_band
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        arr = np.asarray(gt)
        data = arr.astype(np.float64, copy=False)
        valid = _valid(arr, gt)
        if valid is None:
            mean = np.mean(data, axis=(-2, -1), keepdims=True)
        else:
            masked = np.where(_broadcast_valid(valid, data.shape), data, np.nan)
            with warnings.catch_warnings():
                # An all-nodata band has no mean; its pixels are all refilled.
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(masked, axis=(-2, -1), keepdims=True)
        if self.per_band:
            factors = rng.uniform(self.factor[0], self.factor[1], _band_count(arr))
            factors = factors.reshape(_band_shape(arr))
        else:
            factors = _sample_uniform(rng, self.factor, "factor")
        return _cast_and_wrap(gt, (data - mean) * factors + mean, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": jsonable(self.factor),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class GaussianNoise(Operator):
    """Add zero-mean Gaussian sensor noise.

    Pure per-pixel math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind, cast back to the
    input dtype.

    Nodata: fill / non-finite pixels are left untouched and hold the
    output's fill value (see the module docstring).

    Args:
        sigma: Noise standard deviation, either a scalar or a
            ``(lo, hi)`` range to sample from. Default ``0.01``.
        per_band: Draw an independent sigma per band (the band
            axis, ``-3``) instead of one shared sigma. Default ``True``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        sigma: ScalarOrRange = 0.01,
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        sigma = as_tuple(sigma)
        _validate_range(sigma, "sigma")
        self.sigma = sigma
        self.per_band = per_band
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        arr = np.asarray(gt)
        if self.per_band:
            sigmas = np.array(
                [
                    _sample_nonnegative(rng, self.sigma, "sigma")
                    for _ in range(_band_count(arr))
                ]
            ).reshape(_band_shape(arr))
        else:
            sigmas = _sample_nonnegative(rng, self.sigma, "sigma")
        noise = rng.normal(0.0, sigmas, size=arr.shape)
        return _cast_and_wrap(
            gt, arr.astype(np.float64, copy=False) + noise, _valid(arr, gt)
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "sigma": jsonable(self.sigma),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class SpeckleNoise(Operator):
    """Apply multiplicative Gaussian speckle noise, useful for SAR.

    Pure per-pixel math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind, cast back to the
    input dtype.

    Nodata: fill / non-finite pixels are left untouched and hold the
    output's fill value (see the module docstring).

    Args:
        sigma: Speckle standard deviation, either a scalar or a
            ``(lo, hi)`` range to sample from. Default ``0.05``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(self, *, sigma: ScalarOrRange = 0.05, seed: int | None = None) -> None:
        sigma = as_tuple(sigma)
        _validate_range(sigma, "sigma")
        self.sigma = sigma
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        sigma = _sample_nonnegative(rng, self.sigma, "sigma")
        arr = np.asarray(gt)
        noise = rng.normal(0.0, sigma, size=arr.shape)
        return _cast_and_wrap(
            gt, arr.astype(np.float64, copy=False) * (1.0 + noise), _valid(arr, gt)
        )

    def get_config(self) -> dict[str, Any]:
        return {"sigma": jsonable(self.sigma), "seed": self.seed}


class BandDropout(Operator):
    """Fill each band independently with probability ``p``.

    Pure per-band math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind. A 2-D input is
    treated as a single band.

    Nodata: pixels that were fill / non-finite on input keep the output's
    fill value rather than ``fill_value`` (see the module docstring).

    Args:
        p: Per-band dropout probability. Default ``0.1``.
        fill_value: Value written into dropped bands. Default ``0.0``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self, *, p: float = 0.1, fill_value: float = 0.0, seed: int | None = None
    ) -> None:
        _check_probability(p, "p")
        self.p = p
        self.fill_value = fill_value
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        arr = np.asarray(gt)
        out = np.array(arr, copy=True)
        if arr.ndim < 3:
            if rng.random() < self.p:
                out[...] = self.fill_value
            return _cast_and_wrap(gt, out, _valid(arr, gt))

        mask = rng.random(arr.shape[BAND_AXIS]) < self.p
        out[..., mask, :, :] = self.fill_value
        return _cast_and_wrap(gt, out, _valid(arr, gt))


class BandJitter(Operator):
    """Permute bands within explicitly configured groups.

    Metadata-dependent: band names are resolved through the carrier's
    ``attrs`` by the package-wide resolver (``band_names``, then
    ``descriptions``, then ``bands``), so named groups on a plain
    ``np.ndarray`` raise ``TypeError`` and on a GeoTensor without those
    names raise ``ValueError``. Integer indices work on any carrier.

    Args:
        groups: Mapping of group label to the band names (or integer
            indices) permuted within that group, or the equivalent
            ``[[label, names], ...]`` pairs that ``get_config`` emits.
            ``None`` or empty disables the op (identity).
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        groups: dict[str, list[str]] | list[list[Any]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.groups = mapping_from_pairs(groups)
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        if not self.groups:
            return gt

        arr = np.asarray(gt)
        if arr.ndim < 3:
            return gt

        n_bands = arr.shape[BAND_AXIS]
        groups = [resolve_bands(gt, group) for group in self.groups.values()]
        rng = _call_rng(self, seed)
        out = np.array(arr, copy=True)
        for indices in groups:
            for index in indices:
                if not 0 <= index < n_bands:
                    raise ValueError(f"Band index {index} is outside [0, {n_bands}).")
            if len(indices) > 1:
                out[..., indices, :, :] = arr[..., rng.permutation(indices), :, :]
        return _cast_and_wrap(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"groups": mapping_to_pairs(self.groups), "seed": self.seed}


class SunAngleJitter(Operator):
    """Rescale TOA reflectance for a simulated solar-zenith change.

    Metadata-dependent: the base solar zenith angle is read from the
    carrier's ``attrs`` (``solar_zenith_angle`` or ``sza_deg``), so a
    plain ``np.ndarray`` input (or a GeoTensor without those attrs)
    raises ``ValueError``.

    Nodata: fill / non-finite pixels are left untouched and hold the
    output's fill value (see the module docstring).

    Args:
        delta_sza_deg: Solar-zenith perturbation in degrees, either a
            scalar or a ``(lo, hi)`` range to sample from. Default
            ``(-5.0, 5.0)``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        delta_sza_deg: ScalarOrRange = (-5.0, 5.0),
        seed: int | None = None,
    ) -> None:
        self.delta_sza_deg = as_tuple(delta_sza_deg)
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _call_rng(self, seed)
        delta = _sample_uniform(rng, self.delta_sza_deg, "delta_sza_deg")
        attrs = getattr(gt, "attrs", None) or {}
        sza_value = attrs.get("solar_zenith_angle", attrs.get("sza_deg"))
        if sza_value is None:
            raise ValueError(
                "SunAngleJitter requires `solar_zenith_angle` or `sza_deg` in "
                "gt.attrs; got neither."
            )
        scale = sun_angle_scale(float(sza_value), delta)
        arr = np.asarray(gt)
        return _cast_and_wrap(
            gt, arr.astype(np.float64, copy=False) * scale, _valid(arr, gt)
        )

    def get_config(self) -> dict[str, Any]:
        return {"delta_sza_deg": jsonable(self.delta_sza_deg), "seed": self.seed}


class AtmosphericHaze(Operator):
    """Add a sampled haze term following an inverse fourth-power spectrum.

    Wavelengths are read from ``attrs["wavelengths"]`` (the package-wide
    key, in nanometers; values below ``10`` are treated as micrometers and
    converted to nanometers). Accepts a ``GeoTensor`` or a plain
    ``np.ndarray``; when the carrier has no ``wavelengths`` attr, a
    default 450-850 nm linspace is assumed.

    Nodata: fill / non-finite pixels are left untouched and hold the
    output's fill value (see the module docstring).

    Args:
        intensity: Haze amplitude added to the shortest wavelength,
            either a scalar or a ``(lo, hi)`` range to sample from.
            Default ``(0.0, 0.05)``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self, *, intensity: ScalarOrRange = (0.0, 0.05), seed: int | None = None
    ) -> None:
        self.intensity = as_tuple(intensity)
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        intensity = _sample_nonnegative(rng, self.intensity, "intensity")
        if intensity == 0.0:
            return gt

        arr = np.asarray(gt)
        attrs = getattr(gt, "attrs", None) or {}
        wavelengths = attrs.get("wavelengths")
        weights = rayleigh_weights(wavelengths, _band_count(arr)).reshape(
            _band_shape(arr)
        )
        return _cast_and_wrap(
            gt,
            arr.astype(np.float64, copy=False) + intensity * weights,
            _valid(arr, gt),
        )

    def get_config(self) -> dict[str, Any]:
        return {"intensity": jsonable(self.intensity), "seed": self.seed}


class SimulatedClouds(Operator):
    """Overlay a smooth synthetic cloud field onto reflectance imagery.

    Pure per-pixel math: accepts a ``GeoTensor`` or a plain
    ``np.ndarray`` and returns the same carrier kind, cast back to the
    input dtype.

    Nodata: the cloud brightness (max / percentile of the scene) is taken
    over valid pixels only, and fill / non-finite pixels hold the output's
    fill value (see the module docstring).

    Args:
        coverage: Fraction of pixels covered by cloud, either a scalar
            or a ``(lo, hi)`` range in ``[0, 1]``. Default ``(0.0, 0.3)``.
        feather: Gaussian smoothing sigma (pixels) applied to the random
            cloud field; ``0`` disables smoothing. Default ``5``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.
    """

    def __init__(
        self,
        *,
        coverage: ScalarOrRange = (0.0, 0.3),
        feather: int = 5,
        seed: int | None = None,
    ) -> None:
        if feather < 0:
            raise ValueError("feather must be non-negative.")
        coverage = as_tuple(coverage)
        _validate_probability_range(coverage, "coverage")
        self.coverage = coverage
        self.feather = feather
        self.seed = seed

    def _apply(
        self, gt: GeoTensor | np.ndarray, *, seed: int | None = None
    ) -> GeoTensor | np.ndarray:
        rng = _call_rng(self, seed)
        coverage = _sample_uniform(rng, self.coverage, "coverage")
        if coverage == 0.0:
            return gt

        arr = np.asarray(gt)
        alpha = cloud_alpha(
            rng.normal(size=arr.shape[-2:]), coverage, feather=self.feather
        )
        alpha = alpha.reshape((1,) * (arr.ndim - 2) + alpha.shape)
        valid = _valid(arr, gt)
        scene = arr if valid is None else arr[_broadcast_valid(valid, arr.shape)]
        # Assume [0, 1] reflectance if max <= 1; otherwise approximate bright clouds.
        cloud_value = (
            1.0
            if scene.size == 0 or np.nanmax(scene) <= 1.0
            else np.nanpercentile(scene, BRIGHT_CLOUD_PERCENTILE)
        )
        out = arr.astype(np.float64, copy=False) * (1.0 - alpha) + cloud_value * alpha
        return _cast_and_wrap(gt, out, valid)

    def get_config(self) -> dict[str, Any]:
        return {
            "coverage": jsonable(self.coverage),
            "feather": self.feather,
            "seed": self.seed,
        }


class CutMix(Operator):
    """Paste a random rectangle from a pool donor on the input's pixel grid.

    Called as ``op(gt, *pool)`` -- the donors are positional carriers after
    the input (or one list of them, ``op(gt, [d1, d2])``), so the operator
    wires into a ``pipekit.Graph`` as ``CutMix()(Input("x"), Input("d1"),
    Input("d2"))``. With no donors the input passes through unchanged.

    Every donor must share the input's pixel grid exactly
    (:func:`geotoolz._src.geo.grid_matches` with ``spatial_only=False``:
    the full shape, and -- when both carry georeferencing -- the CRS and
    every ``transform`` coefficient, origin included). The paste is
    pixel-space math: pixel ``(i, j)`` of the donor lands on pixel
    ``(i, j)`` of the input, so requiring the same grid keeps the result
    geographically coherent (e.g. a donor from another acquisition date
    or sensor over the same tile). A donor at a different origin, scale
    or CRS is rejected rather than pasted at the wrong location.

    Plain ``np.ndarray`` carriers (input and/or donors) are accepted:
    the georeferencing check applies only when both the input and the
    drawn donor carry a ``transform``; plain arrays are compared by shape.

    Sampling: with probability ``p`` one donor is drawn uniformly from
    the pool; the rectangle's height and width are drawn independently
    and uniformly from ``[1, H]`` and ``[1, W]``, and its top-left corner
    uniformly among the positions that keep it inside the raster. This
    is *not* the Beta(alpha, alpha) area-ratio convention of Yun et al.
    (2019) and no label-mixing ratio is returned (see #141).

    Nodata: donor pixels that are fill / non-finite under the *donor's*
    fill value are written as the input's fill value, so a pasted hole
    stays a hole for the output carrier.

    Args:
        p: Probability of applying the paste. Default ``0.5``.
        seed: Seed of the operator's own draw stream, which advances on
            every call; a per-call ``seed`` makes a one-off draw instead.

    Raises:
        ValueError: At apply time, if any donor is not on the input's
            pixel grid.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.CutMix(p=1.0, seed=0)
        >>> out = op(patch, donor)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        p: float = 0.5,
        seed: int | None = None,
    ) -> None:
        _check_probability(p, "p")
        self.p = p
        self.seed = seed

    def _apply(
        self,
        gt: GeoTensor | np.ndarray,
        *pool: GeoTensor | np.ndarray,
        seed: int | None = None,
    ) -> GeoTensor | np.ndarray:
        donors = gather_inputs(pool, "CutMix") if pool else []
        for idx, candidate in enumerate(donors):
            require_grid_match(
                gt,
                candidate,
                "CutMix",
                names=("input", f"donor {idx}"),
                spatial_only=False,
            )
        rng = _call_rng(self, seed)
        if not donors or rng.random() >= self.p:
            return gt

        donor = donors[int(rng.integers(0, len(donors)))]
        arr = np.asarray(gt)
        donor_arr = np.asarray(donor)

        height, width = arr.shape[-2], arr.shape[-1]
        cut_h = int(rng.integers(1, height + 1))
        cut_w = int(rng.integers(1, width + 1))
        top = int(rng.integers(0, height - cut_h + 1))
        left = int(rng.integers(0, width - cut_w + 1))

        out = np.array(arr, copy=True)
        region = (slice(top, top + cut_h), slice(left, left + cut_w))
        out[..., region[0], region[1]] = donor_arr[..., region[0], region[1]]
        donor_valid = _valid(donor_arr, donor)
        valid = None
        if donor_valid is not None:
            # Input pixels outside the rectangle keep their own values
            # (fills included); only donor holes need the input's fill.
            valid = np.ones(donor_valid.shape, dtype=bool)
            valid[..., region[0], region[1]] = donor_valid[..., region[0], region[1]]
        return _cast_and_wrap(gt, out, valid)
