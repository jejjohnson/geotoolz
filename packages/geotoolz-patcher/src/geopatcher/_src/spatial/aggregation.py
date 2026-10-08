"""`spatial.aggregation.Aggregation` — local patch results → global field.

The aggregation step is the inverse of `split`: take an iterable of
patches (each annotated with its indices and weights) and reconstruct a
single global field. The streaming-safe families are monoidal folds
over one or more accumulators; the non-streaming ones (`spatial.aggregation.Median`,
`spatial.aggregation.Mode`, `spatial.aggregation.Learned`) need a per-cell history and
accumulate in memory.

The `streaming_safe` class flag advertises which aggregations support
the disk-backed path (a target zarr store) — `spatial.aggregation.OverlapAdd` is the
canonical streaming-safe member; `spatial.aggregation.Median` triggers a warning if the
caller asks for streaming.

See ``docs/patcher/patching.md`` §"Streaming aggregations" for the framing.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._extras import missing_extra
from geopatcher._src._serialize import config_from_fields


COG_WRITER = "cog"
ZARR_WRITER = "zarr"
DEFAULT_COG_BLOCKSIZE = 512
HASH_BITS = 64


# ---------------------------------------------------------------------------
# Base + helpers
# ---------------------------------------------------------------------------


class Aggregation:
    """Base for local → global merging strategies.

    Subclasses implement ``merge(patches, domain) -> field-like``.

    ``streaming_safe = True`` means the aggregation can be reduced one
    patch at a time without keeping per-cell history; the disk-backed
    `spatial.aggregation.OverlapAdd(streaming=True, target_path=...)` path lights up
    automatically for these.
    """

    streaming_safe: ClassVar[bool] = False
    forbid_in_yaml: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {}


def _domain_array_shape(domain: Any) -> tuple[int, ...]:
    """Best-effort full-array shape for a `Domain`.

    For raster: ``domain.shape``. For grid: ``tuple(domain.shape)``.
    Other domains don't have a natural dense shape — those aggregations
    don't apply, and the caller will land in `spatial.aggregation.ByIndex` instead.
    """
    if hasattr(domain, "shape"):
        return tuple(domain.shape)
    raise TypeError(f"can't infer dense shape from {type(domain).__name__}")


def _is_window(indices: Any) -> bool:
    """Whether ``indices`` place a patch by raster window (the trailing ``(H, W)``)."""
    from geopatcher._src.spatial.geometry import _MaskedWindow

    if isinstance(indices, _MaskedWindow):
        indices = indices.window
    return hasattr(indices, "row_off") and hasattr(indices, "col_off")


def _dense_layout(
    patches: Iterable[Any],
    domain: Any,
    cells: Callable[[Any], tuple[int, ...]] = np.shape,
) -> tuple[tuple[int, ...], Iterator[Any]]:
    """Accumulator shape of a dense merge, and the patches to fold into it.

    The domain fixes the grid, the trailing ``(H, W)``; the patches fix
    what each grid cell holds. Raster-window patches set the leading
    (band / time) axes from their own data, so an operator that turns
    four bands into one index merges into ``(1, H, W)`` (or ``(H, W)``
    for a 2-D map) instead of being broadcast across the domain's four
    bands. The first patch decides, and a later window patch whose
    leading axes differ raises `ValueError`. ``{dim: slice}`` (grid)
    patches index the domain's own axes and keep its shape.

    Args:
        patches: The patches to merge (any iterable; consumed once).
        domain: The domain they were drawn from.
        cells: The per-cell array shape of a patch's ``data`` — without the
            class axis of `spatial.aggregation.SoftVote` or the
            ``(mu, var)`` pair of `spatial.aggregation.InvVarWeightedMean`.

    Returns:
        ``(shape, patches)``: the accumulator shape and an iterator over
        every patch, the inspected first one included.
    """
    shape = _domain_array_shape(domain)
    stream = iter(patches)
    first = next(stream, None)
    if first is None:
        return shape, iter(())
    rest = itertools.chain([first], stream)
    first_cells = tuple(cells(first.data))
    if not _is_window(first.indices) or len(first_cells) < 2:
        return shape, rest
    shape = first_cells[:-2] + shape[-2:]
    return shape, _same_leading(rest, shape[:-2], cells)


def _same_leading(
    patches: Iterator[Any],
    leading: tuple[int, ...],
    cells: Callable[[Any], tuple[int, ...]],
) -> Iterator[Any]:
    """Yield ``patches``; raise if a window patch's leading axes are not ``leading``."""
    for p in patches:
        found = tuple(cells(p.data))[:-2]
        if _is_window(p.indices) and found != leading:
            raise ValueError(
                f"patches disagree on their leading (band / time) axes: {found} "
                f"after {leading}; every patch merged together must have the "
                "same shape apart from its window"
            )
        yield p


def _not_nan(array: np.ndarray) -> np.ndarray:
    """``array``-shaped bool: ``True`` where the value is not NaN.

    Infinities are real values — only NaN marks a missing sample (an
    ``on_error="mask"`` patch, a nodata hole, …). Integer / bool arrays
    have no NaN.
    """
    if np.issubdtype(array.dtype, np.inexact):
        return ~np.isnan(array)
    return np.ones(array.shape, dtype=bool)


def _as_float(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _label_dtype(fill_value: float) -> type[np.generic]:
    """Output dtype of a label aggregation: ``int64`` iff the fill is integral."""
    if isinstance(fill_value, bool):
        raise TypeError("fill_value must be a number, not a bool")
    if isinstance(fill_value, int | np.integer):
        return np.int64
    if math.isfinite(fill_value) and float(fill_value).is_integer():
        return np.int64
    return np.float64


def _with_fill(out: np.ndarray, covered: np.ndarray, fill_value: float) -> np.ndarray:
    """``out`` where ``covered``, else ``fill_value`` — always float64."""
    return np.where(covered, out, np.float64(fill_value))


@dataclass(frozen=True)
class _Placement:
    """Where a patch lands in the accumulator, cropped to the domain.

    A patch's ``indices`` may overhang the domain (``boundary="pad"`` /
    ``"reflect"`` chips, or any caller-built window), while numpy would
    silently clip a too-long slice — and read a negative start from the
    end. Every dense aggregation therefore writes only the in-domain
    part: ``acc[placement.acc] ⊕= f(placement.crop(data))``, with the
    weights cropped by the same ``placement.crop``, and counts only the
    cells ``placement.valid(data)`` keeps — not NaN, and inside the
    interior mask of a `_MaskedWindow`.

    Attributes:
        acc: Slicer into the domain-shaped accumulator, clipped to
            ``[0, n)`` on every sliced axis.
        chip: Matching slicer into the patch's own data / weights —
            the offset of the in-domain part within the chip.
        mask: In-domain part of a `_MaskedWindow`'s interior mask
            (``True`` = inside), or ``None`` when every cell counts.
    """

    acc: tuple[Any, ...]
    chip: tuple[Any, ...]
    mask: np.ndarray | None = None

    def crop(self, array: Any) -> Any:
        """Crop a chip-shaped ``array`` (data or weights) to the in-domain part.

        ``None`` passes through. An array with fewer axes than the slicer
        (2-D weights broadcast over a ``(band, H, W)`` chip) is cropped on
        its trailing axes only.
        """
        if array is None:
            return None
        arr = np.asarray(array)
        chip = self.chip
        if chip and chip[0] is not Ellipsis and arr.ndim < len(chip):
            chip = chip[len(chip) - arr.ndim :]
        return arr[chip]

    def valid(self, data: np.ndarray) -> np.ndarray:
        """Cells of the cropped ``data`` that count: not NaN and inside the mask.

        The mask broadcasts against ``data`` on the trailing axes, so a
        2-D polygon mask covers every band of a ``(band, H, W)`` chip.
        """
        ok = _not_nan(data)
        if self.mask is not None:
            ok = ok & self.mask
        return ok


def _clip_axis(start: int, stop: int, length: int) -> tuple[slice, slice]:
    """``(acc_slice, chip_slice)`` for ``[start, stop)`` clipped to ``[0, length)``."""
    lo = min(max(start, 0), length)
    hi = min(max(stop, lo), length)
    return slice(lo, hi), slice(lo - start, hi - start)


def _resolve_indices(indices: Any, shape: tuple[int, ...]) -> _Placement | None:
    """Map a patch's indices to an in-domain `_Placement`.

    The accumulator slicer targets the **trailing** axes for a raster
    window — leading band/time dims pass through via ``Ellipsis``, so
    ``acc[placement.acc]`` selects ``(..., row_slice, col_slice)`` on a
    3-D ``(band, H, W)`` array — and the leading axes, in dict order, for
    a ``GridDomain`` indexer.

    - rasterio.windows.Window: ``(..., row_slice, col_slice)``
    - ``dict[str, slice]``: one entry per dim in dict order (no ellipsis);
      non-slice entries (integer / list indexers) pass through unclipped
    - ``_MaskedWindow``: resolve the underlying window and carry its
      interior mask, cropped like the data, as ``placement.mask``

    Returns ``None`` when the patch has no in-domain cell (the caller
    skips it).

    Raises:
        TypeError: ``indices`` is not a dense placement (a point-index
            array, a polygon id, …). Dense aggregations refuse it rather
            than silently merging an empty field.
    """
    from geopatcher._src.spatial.geometry import _MaskedWindow

    if isinstance(indices, _MaskedWindow):
        placement = _resolve_indices(indices.window, shape)
        if placement is None:
            return None
        mask = np.asarray(placement.crop(indices.mask), dtype=bool)
        return _Placement(acc=placement.acc, chip=placement.chip, mask=mask)
    if hasattr(indices, "row_off") and hasattr(indices, "col_off"):
        r0, c0 = int(indices.row_off), int(indices.col_off)
        h, w = int(indices.height), int(indices.width)
        acc_r, chip_r = _clip_axis(r0, r0 + h, int(shape[-2]))
        acc_c, chip_c = _clip_axis(c0, c0 + w, int(shape[-1]))
        if acc_r.stop <= acc_r.start or acc_c.stop <= acc_c.start:
            return None
        return _Placement(acc=(Ellipsis, acc_r, acc_c), chip=(Ellipsis, chip_r, chip_c))
    if isinstance(indices, dict):
        acc: list[Any] = []
        chip: list[Any] = []
        for axis, index in enumerate(indices.values()):
            if not (isinstance(index, slice) and index.step in (None, 1)):
                acc.append(index)
                if not isinstance(index, int | np.integer):
                    chip.append(slice(None))
                continue
            length = int(shape[axis])
            start = 0 if index.start is None else int(index.start)
            stop = length if index.stop is None else int(index.stop)
            acc_s, chip_s = _clip_axis(start, stop, length)
            if acc_s.stop <= acc_s.start:
                return None
            acc.append(acc_s)
            chip.append(chip_s)
        return _Placement(acc=tuple(acc), chip=tuple(chip))
    raise TypeError(
        "dense aggregations need raster / grid patch indices (a rasterio "
        "Window, a {dim: slice} dict or a masked window), got "
        f"{type(indices).__name__}; use spatial.aggregation.ByIndex for ragged "
        "geometries"
    )


def _placed(p: Any, shape: tuple[int, ...]) -> tuple[_Placement, np.ndarray] | None:
    """``(placement, cropped float64 data)`` for one patch, ``None`` if off-domain."""
    pl = _resolve_indices(p.indices, shape)
    if pl is None:
        return None
    return pl, _as_float(pl.crop(p.data))


# ---------------------------------------------------------------------------
# Exact streaming aggregations
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Sum(Aggregation):
    """Per-cell sum across patches (NaN / masked samples skipped).

    Args:
        fill_value: Written into cells no valid sample reached
            (uncovered, all-NaN or outside every mask). Default NaN;
            pass e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        acc = np.zeros(shape, dtype=np.float64)
        covered = np.zeros(shape, dtype=bool)
        for p in patches:
            placed = _placed(p, shape)
            if placed is None:
                continue
            pl, x = placed
            valid = pl.valid(x)
            acc[pl.acc] += np.where(valid, x, 0.0)
            covered[pl.acc] |= valid
        return _with_fill(acc, covered, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Max(Aggregation):
    """Per-cell maximum across patches (NaN / masked samples skipped).

    Args:
        fill_value: Written into cells no valid sample reached. Default
            NaN; pass e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        return _extreme(patches, domain, np.fmax, -np.inf, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Min(Aggregation):
    """Per-cell minimum across patches (NaN / masked samples skipped).

    Args:
        fill_value: Written into cells no valid sample reached. Default
            NaN; pass e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        return _extreme(patches, domain, np.fmin, np.inf, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _extreme(
    patches: Iterable[Any],
    domain: Any,
    op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    start: float,
    fill_value: float,
) -> np.ndarray:
    """Shared `spatial.aggregation.Max` / `spatial.aggregation.Min` fold (``op`` is
    ``np.fmax`` / ``np.fmin``)."""
    shape, patches = _dense_layout(patches, domain)
    acc = np.full(shape, start, dtype=np.float64)
    covered = np.zeros(shape, dtype=bool)
    for p in patches:
        placed = _placed(p, shape)
        if placed is None:
            continue
        pl, x = placed
        valid = pl.valid(x)
        block = acc[pl.acc]
        acc[pl.acc] = np.where(valid, op(block, x), block)
        covered[pl.acc] |= valid
    return _with_fill(acc, covered, fill_value)


@dataclass(eq=False)
class WeightedSum(Aggregation):
    """Per-cell weighted sum — each patch's window weights multiply in.

    Args:
        weight_fn: Optional callable ``(patch) -> Array`` overriding the
            per-patch weights. ``None`` uses ``patch.weights`` directly.
        fill_value: Written into cells whose accumulated weight of valid
            samples is not positive (uncovered, all-NaN, outside every
            mask, or a taper's zero edge). Default NaN; pass e.g. the
            domain's nodata to override.

    Note:
        With a ``weight_fn`` the instance is ``forbid_in_yaml``: the
        callable has no JSON form, so ``get_config()`` records only its
        name, a debug record that `from_config` refuses to rebuild.
        Without one the config rebuilds the aggregation.
    """

    weight_fn: Callable[[Any], np.ndarray] | None = None
    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.weight_fn is None:
            return
        if not callable(self.weight_fn):
            raise TypeError(
                "spatial.aggregation.WeightedSum.weight_fn must be a callable or "
                "None, got "
                f"{self.weight_fn!r} — a config naming a weight_fn cannot be "
                "rebuilt; construct it in code."
            )
        # Per-instance flag (shadowing the ClassVar): only a weight_fn makes
        # the config unfaithful, so `spatial.aggregation.WeightedSum()` stays
        # rebuildable.
        object.__setattr__(self, "forbid_in_yaml", True)

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        acc = np.zeros(shape, dtype=np.float64)
        wsum = np.zeros(shape, dtype=np.float64)
        for p in patches:
            placed = _placed(p, shape)
            if placed is None:
                continue
            pl, x = placed
            raw_w = self.weight_fn(p) if self.weight_fn else p.weights
            w = np.float64(1.0) if raw_w is None else _as_float(pl.crop(raw_w))
            valid = pl.valid(x)
            acc[pl.acc] += np.where(valid, x * w, 0.0)
            wsum[pl.acc] += np.where(valid, w, 0.0)
        return _with_fill(acc, wsum > 0, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        fn = self.weight_fn
        return {
            "weight_fn": None if fn is None else getattr(fn, "__name__", repr(fn)),
            **config_from_fields(self, exclude=("weight_fn",)),
        }


# ---------------------------------------------------------------------------
# Compound monoidal aggregations
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Mean(Aggregation):
    """Per-cell mean — runs `spatial.aggregation.Sum` and a count accumulator in
    parallel.

    NaN samples and cells outside a `_MaskedWindow` mask are not counted.

    Args:
        fill_value: Written into cells no valid sample reached. Default
            NaN; pass e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        total = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        for p in patches:
            placed = _placed(p, shape)
            if placed is None:
                continue
            pl, x = placed
            valid = pl.valid(x)
            total[pl.acc] += np.where(valid, x, 0.0)
            count[pl.acc] += valid
        with np.errstate(invalid="ignore", divide="ignore"):
            return _with_fill(total / count, count > 0, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Variance(Aggregation):
    """Per-cell sample variance via Welford's online algorithm.

    Returns the unbiased estimate (``ddof=1``, as ``np.nanvar``); NaN and
    masked samples are not counted.

    Args:
        fill_value: Written into cells with fewer than two valid samples,
            where the ``ddof=1`` variance is undefined. Default NaN; pass
            e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        mean = np.zeros(shape, dtype=np.float64)
        m2 = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        for p in patches:
            placed = _placed(p, shape)
            if placed is None:
                continue
            pl, x = placed
            sl = pl.acc
            valid = pl.valid(x)
            x_clean = np.where(valid, x, 0.0)
            next_count = count[sl] + valid
            delta = np.where(valid, x_clean - mean[sl], 0.0)
            denom = np.where(next_count > 0, next_count, 1.0)
            mean[sl] += np.where(next_count > 0, delta / denom, 0.0)
            m2[sl] += np.where(valid, delta * (x_clean - mean[sl]), 0.0)
            count[sl] = next_count
        with np.errstate(invalid="ignore", divide="ignore"):
            return _with_fill(m2 / (count - 1), count > 1, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _global_values(p: Any, shape: tuple[int, ...] | None) -> np.ndarray:
    """A patch's valid samples, flattened, for the global reducers.

    On a dense (raster / grid) domain only the chip's in-domain part
    counts — ``"pad"`` / ``"reflect"`` fill is context, not data — and
    only inside a `_MaskedWindow` mask. NaN never counts.
    """
    if shape is None:
        x = _as_float(p.data).reshape(-1)
        return x[_not_nan(x)]
    placed = _placed(p, shape)
    if placed is None:
        return np.empty(0, dtype=np.float64)
    pl, x = placed
    return x[np.broadcast_to(pl.valid(x), x.shape)]


def _dense_shape_or_none(domain: Any) -> tuple[int, ...] | None:
    return tuple(domain.shape) if hasattr(domain, "shape") else None


@dataclass(eq=False)
class MeanStd(Aggregation):
    """Global mean and sample standard deviation across patch data.

    NaN samples are skipped. On a dense (raster / grid) domain only each
    chip's in-domain, in-mask part counts, so pad / reflect fill is not
    data; on other domains every non-NaN sample counts.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, float]:
        shape = _dense_shape_or_none(domain)
        count = 0
        mean = 0.0
        m2 = 0.0
        for p in patches:
            x = _global_values(p, shape)
            if x.size == 0:
                continue
            batch_count = int(x.size)
            batch_mean = float(np.mean(x))
            batch_m2 = float(np.sum((x - batch_mean) ** 2))
            next_count = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta * delta * count * batch_count / next_count
            mean += delta * batch_count / next_count
            count = next_count
        if count == 0:
            raise ValueError("spatial.aggregation.MeanStd requires at least one value")
        var = m2 / (count - 1) if count > 1 else 0.0
        return {"mean": mean, "std": float(np.sqrt(var))}


@dataclass(eq=False)
class MinMax(Aggregation):
    """Global minimum and maximum across patch data.

    NaN samples are skipped. On a dense (raster / grid) domain only each
    chip's in-domain, in-mask part counts.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, float]:
        shape = _dense_shape_or_none(domain)
        min_value = np.inf
        max_value = -np.inf
        seen = False
        for p in patches:
            x = _global_values(p, shape)
            if x.size == 0:
                continue
            min_value = min(min_value, float(np.min(x)))
            max_value = max(max_value, float(np.max(x)))
            seen = True
        if not seen:
            raise ValueError("spatial.aggregation.MinMax requires at least one value")
        return {"min": min_value, "max": max_value}


@dataclass(eq=False)
class OverlapAdd(Aggregation):
    """spatial.window.Window-weighted overlap-add — the canonical chip-stitching
    aggregator.

    Computes ``Σ wᵢ xᵢ / Σ wᵢ`` per cell. When stride < patch size and a
    `spatial.window.Hann` (or similar partition-of-unity) window is used, the resulting
    field equals the original (modulo the operator's effect) — the
    standard inference-time stitching pattern.

    With ``streaming=True`` and a ``target_path`` the two accumulators
    live in a chunked on-disk zarr store instead of RAM: each patch is a
    read-modify-write of the blocks it touches, and the final
    normalisation runs block by block, so peak memory is one patch plus
    one block — never the whole field.

    Args:
        streaming: If ``True`` and ``target_path`` is set, accumulate
            on disk (see ``writer``) rather than in RAM.
        target_path: With ``writer="zarr"``, the directory holding the
            ``rec.zarr`` (result) and ``wsum.zarr`` (accumulated weight)
            arrays; with ``writer="cog"``, the output GeoTIFF path.
        chunks: Zarr chunk shape of the streaming accumulators — required
            for ``writer="zarr"``; pass the patch geometry's size (e.g.
            ``chunks=geometry.size``) so each patch writes whole blocks.
            Right-aligned against the domain shape: leading band / time
            dims missing from it get their full extent. With
            ``writer="cog"`` it defaults to the COG block size.
        shard_shape: Optional zarr v3 shard shape (right-aligned like
            ``chunks``).
        writer: ``"zarr"`` (default) returns the result ``zarr.Array``;
            ``"cog"`` streams through a temporary zarr store and converts
            it, block by block, into a Cloud-Optimized GeoTIFF (GDAL
            ``COG`` driver: tiled, overviews, ``nodata = fill_value``),
            returning ``target_path``.
        cog: ``writer="cog"`` options — ``blocksize`` (default 512),
            ``compress`` (default ``"DEFLATE"``), ``bigtiff`` (default
            ``"IF_SAFER"``); any other key is forwarded upper-cased as a
            GDAL COG creation option (e.g. ``overview_resampling``).
        normalize_by_window: Divide by the accumulated weight at the end
            (default ``True``). Set to ``False`` for the raw weighted
            sum.
        fill_value: Written into cells whose accumulated weight of valid
            samples is zero — uncovered, all-NaN, outside every mask, or
            only reached by a taper's zero edge (the leading row / column
            under a periodic `spatial.window.Hann`). Default NaN; pass e.g. the
            domain's nodata to override. It is also the COG's nodata.
        dtype: Floating dtype of the streaming accumulators and output
            (default ``"float32"``). The in-RAM path is float64.
        overwrite: Replace an existing store / file at ``target_path``.
            Default ``False``: a second merge onto the same path raises
            `FileExistsError` instead of silently destroying the first.
    """

    streaming: bool = False
    target_path: str | None = None
    chunks: tuple[int, ...] | None = None
    shard_shape: tuple[int, ...] | None = None
    writer: str = "zarr"
    cog: dict[str, Any] | None = None
    normalize_by_window: bool = True
    fill_value: float = math.nan
    dtype: str = "float32"
    overwrite: bool = False

    streaming_safe: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.writer not in (ZARR_WRITER, COG_WRITER):
            raise ValueError(
                f"writer must be {ZARR_WRITER!r} or {COG_WRITER!r}, got {self.writer!r}"
            )
        if not np.issubdtype(np.dtype(self.dtype), np.floating):
            raise ValueError(
                f"dtype must be a floating dtype to hold the fill, got {self.dtype!r}"
            )

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        if not (self.streaming and self.target_path):
            return self._merge_in_memory(patches, domain)
        if self.writer == COG_WRITER:
            return self._merge_cog(patches, domain, self.target_path)
        if self.chunks is None:
            raise ValueError(
                "spatial.aggregation.OverlapAdd(streaming=True) needs chunks= — pass "
                "the "
                "patch geometry's size (e.g. chunks=geometry.size) so each "
                "patch writes whole blocks"
            )
        return self._merge_streaming(patches, domain, self.target_path, self.chunks)

    def _merge_in_memory(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        acc = np.zeros(shape, dtype=np.float64)
        wsum = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            contrib, weight = _weighted_block(pl, p, np.float64, len(shape))
            acc[pl.acc] += contrib
            wsum[pl.acc] += weight
        out = acc
        if self.normalize_by_window:
            with np.errstate(invalid="ignore", divide="ignore"):
                out = acc / wsum
        return _with_fill(out, wsum > 0, self.fill_value)

    def _merge_streaming(
        self,
        patches: Iterable[Any],
        domain: Any,
        root: str,
        chunks: tuple[int, ...],
        *,
        overwrite: bool | None = None,
    ) -> Any:
        shape, patches = _dense_layout(patches, domain)
        chunks = _right_align(chunks, shape, "chunks")
        shards = (
            None
            if self.shard_shape is None
            else _right_align(self.shard_shape, shape, "shard_shape")
        )
        replace = self.overwrite if overwrite is None else overwrite
        dtype = np.dtype(self.dtype)
        rec, wsum = (
            _open_zarr_array(
                f"{root}/{name}.zarr",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                shard_shape=shards,
                overwrite=replace,
            )
            for name in ("rec", "wsum")
        )
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            contrib, weight = _weighted_block(pl, p, dtype.type, len(shape))
            rec[pl.acc] = np.asarray(rec[pl.acc]) + contrib
            wsum[pl.acc] = np.asarray(wsum[pl.acc]) + weight
        # Normalise block by block (one shard, or one chunk, at a time) so
        # the final pass never loads a full-domain accumulator.
        for blk in _blocks(shape, shards or chunks):
            weight = np.asarray(wsum[blk])
            out = np.asarray(rec[blk])
            if self.normalize_by_window:
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = out / weight
            rec[blk] = _with_fill(out, weight > 0, self.fill_value)
        return rec

    def _merge_cog(self, patches: Iterable[Any], domain: Any, target: str) -> str:
        import os
        import tempfile

        if os.path.exists(target) and not self.overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True to replace it"
            )
        options = dict(self.cog or {})
        blocksize = int(options.pop("blocksize", DEFAULT_COG_BLOCKSIZE))
        shape = _domain_array_shape(domain)
        if len(shape) not in (2, 3):
            raise ValueError(
                "COG writer expects a 2-D domain or a 3-D band-first domain"
            )
        chunks = self.chunks or tuple(min(blocksize, n) for n in shape[-2:])
        parent = os.path.dirname(os.path.abspath(target))
        with tempfile.TemporaryDirectory(dir=parent, prefix=".geopatcher-") as tmp:
            rec = self._merge_streaming(patches, domain, tmp, chunks, overwrite=True)
            _zarr_to_cog(
                rec,
                domain,
                target,
                blocksize=blocksize,
                fill_value=self.fill_value,
                scratch=os.path.join(tmp, "stage.tif"),
                options=options,
            )
        return target

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _weighted_block(
    pl: _Placement, p: Any, dtype: type[np.floating], ndim: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(Σ-contribution w·x, weight w)`` of one patch, zero where invalid."""
    x = np.asarray(pl.crop(p.data), dtype=dtype)
    if x.ndim > ndim:
        raise ValueError(
            f"patch data has {x.ndim} dims but the domain has {ndim}; "
            "an overlap-add patch cannot carry extra axes"
        )
    w = (
        np.asarray(pl.crop(p.weights), dtype=dtype)
        if p.weights is not None
        else np.ones_like(x)
    )
    valid = pl.valid(x)
    return np.where(valid, x * w, 0), np.where(valid, w, 0)


def _right_align(
    block: tuple[int, ...], shape: tuple[int, ...], name: str
) -> tuple[int, ...]:
    """Right-align a block shape against ``shape``; missing leading dims are whole."""
    block = tuple(int(b) for b in block)
    if len(block) > len(shape) or any(b < 1 for b in block):
        raise ValueError(
            f"{name}={block} does not fit a {len(shape)}-D domain of shape {shape}"
        )
    return tuple(shape[: len(shape) - len(block)]) + block


def _blocks(
    shape: tuple[int, ...], block: tuple[int, ...]
) -> Iterator[tuple[slice, ...]]:
    """Every ``block``-shaped slicer tiling ``shape`` (edge blocks partial)."""
    starts = [range(0, n, b) for n, b in zip(shape, block, strict=True)]
    for corner in itertools.product(*starts):
        yield tuple(
            slice(s, min(s + b, n))
            for s, b, n in zip(corner, block, shape, strict=True)
        )


def _open_zarr_array(
    path: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    shard_shape: tuple[int, ...] | None,
    overwrite: bool,
) -> Any:
    """Create a zero-filled zarr v3 array, refusing to replace one unless asked."""
    try:
        import zarr
    except ImportError as exc:
        raise missing_extra(
            "spatial.aggregation.OverlapAdd(streaming=True)", "streaming"
        ) from exc
    from zarr.errors import (
        ContainsArrayAndGroupError,
        ContainsArrayError,
        ContainsGroupError,
    )

    try:
        return zarr.create_array(
            path,
            shape=shape,
            chunks=chunks,
            shards=shard_shape,
            dtype=dtype,
            fill_value=0.0,
            overwrite=overwrite,
        )
    except (ContainsArrayError, ContainsGroupError, ContainsArrayAndGroupError) as exc:
        raise FileExistsError(
            f"{path} already holds a zarr store; pass overwrite=True to replace it"
        ) from exc


def _zarr_to_cog(
    rec: Any,
    domain: Any,
    target: str,
    *,
    blocksize: int,
    fill_value: float,
    scratch: str,
    options: dict[str, Any],
) -> None:
    """Convert a 2-D / band-first 3-D zarr result into a COG, block by block.

    GDAL's ``COG`` driver is copy-only (it lays out overviews and tiles in
    a single pass over a finished source), so the result is first copied
    window by window into a tiled scratch GeoTIFF, then converted with
    ``rasterio.shutil.copy(driver="COG")``. Neither step holds more than
    one block of the field in Python memory.
    """
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.windows import Window

    shape = tuple(rec.shape)
    if len(shape) not in (2, 3):
        raise ValueError("COG writer expects a 2-D array or a 3-D band-first array")
    height, width = shape[-2:]
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1 if len(shape) == 2 else shape[0],
        "dtype": np.dtype(rec.dtype).name,
        "crs": getattr(domain, "crs", None),
        "transform": getattr(domain, "transform", rasterio.Affine.identity()),
        "nodata": fill_value,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(scratch, "w", **profile) as dst:
        for blk in _blocks((height, width), (blocksize, blocksize)):
            rows, cols = blk
            data = np.asarray(rec[(Ellipsis, rows, cols)])
            window = Window.from_slices(rows, cols)
            dst.write(data[np.newaxis] if data.ndim == 2 else data, window=window)
    creation = {
        "BLOCKSIZE": blocksize,
        "COMPRESS": options.pop("compress", "DEFLATE"),
        "BIGTIFF": options.pop("bigtiff", "IF_SAFER"),
        **{key.upper(): value for key, value in options.items()},
    }
    rio_copy(scratch, target, driver="COG", **creation)


@dataclass(eq=False)
class InvVarWeightedMean(Aggregation):
    """Bayesian inverse-variance weighting for overlapping local posteriors.

    Each patch produces ``(mu, var)`` — i.e. ``patch.data`` is a tuple or
    a dict with ``"mu"`` / ``"var"`` keys. Returns a dict ``{"mu":
    global_mu, "var": global_var}`` with Kalman-optimal combination::

        μ_global = Σ wᵢ μᵢ / σᵢ² / Σ wᵢ / σᵢ²
        σ²_global = 1 / Σ wᵢ / σᵢ²

    A sample with ``var == 0`` is exact: it has infinite precision, so a
    cell with any such sample takes the (``w``-weighted mean of the) exact
    ``mu`` and ``var = 0``. Samples with NaN ``mu`` / ``var``, negative
    ``var``, non-positive weight or outside a `_MaskedWindow` mask are
    skipped.

    Args:
        fill_value: Written into both ``mu`` and ``var`` of cells no valid
            sample reached. Default NaN; pass e.g. the domain's nodata to
            override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, np.ndarray]:
        shape, patches = _dense_layout(
            patches, domain, lambda data: np.shape(_unpack_mu_var(data)[0])
        )
        mu_acc = np.zeros(shape, dtype=np.float64)
        prec = np.zeros(shape, dtype=np.float64)
        exact_mu = np.zeros(shape, dtype=np.float64)
        exact_w = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            mu, var = (_as_float(pl.crop(part)) for part in _unpack_mu_var(p.data))
            w = (
                _as_float(pl.crop(p.weights))
                if p.weights is not None
                else np.ones_like(mu)
            )
            ok = pl.valid(mu) & _not_nan(var) & (w > 0)
            exact = ok & (var == 0)
            regular = ok & (var > 0)
            inv_var = np.divide(
                w, var, out=np.zeros(np.broadcast(w, var).shape), where=regular
            )
            mu_acc[pl.acc] += np.where(regular, inv_var * mu, 0.0)
            prec[pl.acc] += inv_var
            exact_mu[pl.acc] += np.where(exact, w * mu, 0.0)
            exact_w[pl.acc] += np.where(exact, w, 0.0)
        is_exact = exact_w > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            mu_g = np.where(is_exact, exact_mu / exact_w, mu_acc / prec)
            var_g = np.where(is_exact, 0.0, 1.0 / prec)
        covered = is_exact | (prec > 0)
        return {
            "mu": _with_fill(mu_g, covered, self.fill_value),
            "var": _with_fill(var_g, covered, self.fill_value),
        }

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _unpack_mu_var(data: Any) -> tuple[Any, Any]:
    if isinstance(data, tuple):
        return data
    if isinstance(data, dict):
        return data["mu"], data["var"]
    raise TypeError(
        "spatial.aggregation.InvVarWeightedMean expects each patch's data to be a "
        "(mu, var) "
        "tuple or a {'mu': ..., 'var': ...} mapping."
    )


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class HardVote(Aggregation):
    """Per-cell majority vote — patches carry integer class predictions.

    Values outside ``[0, n_classes)``, NaN and samples outside a
    `_MaskedWindow` mask cast no vote. **Ties** go to the lowest class
    index (``np.argmax``).

    Args:
        n_classes: Total number of classes ``K``. Accumulator shape is
            ``(K, *domain_shape)``.
        fill_value: Label of cells that received no vote. Default ``-1``
            (never a class); the output is ``int64`` for an integral fill
            and ``float64`` otherwise (e.g. ``fill_value=float("nan")``).
    """

    n_classes: int
    fill_value: int | float = -1

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        votes = np.zeros((self.n_classes, *shape), dtype=np.int64)
        for p in patches:
            placed = _placed(p, shape)
            if placed is None:
                continue
            pl, x = placed
            valid = pl.valid(x) & np.isfinite(x)
            cls = np.where(valid, x, -1).astype(np.int64)
            for k in range(self.n_classes):
                votes[k][pl.acc] += cls == k
        out = _with_fill(np.argmax(votes, axis=0), votes.any(axis=0), self.fill_value)
        return out.astype(_label_dtype(self.fill_value))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SoftVote(Aggregation):
    """Per-cell soft vote — patches carry per-class probabilities.

    Each patch's data has shape ``(n_classes, ...)``: the class axis
    first, then the cell axes. The per-class probabilities accumulate and
    the argmax across the class axis is returned, on the cell axes the
    patches carry — ``(K, h, w)`` probabilities on a ``(band, H, W)``
    domain give an ``(H, W)`` label map. A cell counts only
    where none of its class probabilities is NaN and it lies inside a
    `_MaskedWindow` mask. **Ties** go to the lowest class index.

    Args:
        n_classes: Total number of classes ``K``.
        fill_value: Label of cells no valid probability reached. Default
            ``-1``; the output is ``int64`` for an integral fill and
            ``float64`` otherwise.
    """

    n_classes: int
    fill_value: int | float = -1

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain, lambda data: np.shape(data)[1:])
        acc = np.zeros((self.n_classes, *shape), dtype=np.float64)
        covered = np.zeros(shape, dtype=bool)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            probs = _as_float(p.data)
            # (K, h, w) on an N-D domain → (K, 1, …, 1, h, w): the class
            # axis stays first and the cell axes stay trailing.
            missing = len(shape) - (probs.ndim - 1)
            if missing > 0:
                probs = probs.reshape(
                    (probs.shape[0], *([1] * missing), *probs.shape[1:])
                )
            # The leading class axis is kept whole on both sides.
            probs = probs[(slice(None), *pl.chip)]
            cell_ok = pl.valid(probs).all(axis=0)
            acc[(slice(None), *pl.acc)] += np.where(cell_ok, probs, 0.0)
            covered[pl.acc] |= cell_ok
        out = _with_fill(np.argmax(acc, axis=0), covered, self.fill_value)
        return out.astype(_label_dtype(self.fill_value))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class ByIndex(Aggregation):
    """Don't merge — return the ``[(anchor, data), …]`` pairs in patch order.

    The natural choice for ragged geometries (`spatial.geometry.RadiusGraph`,
    `spatial.geometry.KNNGraph`, `spatial.geometry.PolygonIntersection`) where the
    per-patch
    outputs aren't laid out on a regular grid. A list of pairs rather
    than a ``dict``: `GridDomain` anchors are ``dict``s and graph / array
    anchors are numpy arrays (both unhashable), and two patches may share
    an anchor — the same convention as `SpatioTemporalPatcher.merge`.
    Build ``dict(out)`` yourself when the anchors are hashable and unique.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> list[tuple[Any, Any]]:
        return [(p.anchor, p.data) for p in patches]


# ---------------------------------------------------------------------------
# Honest non-streamable
# ---------------------------------------------------------------------------


def _overlap_stack(
    patches: Iterable[Any], shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell stack of every valid sample, as deep as the maximum overlap.

    Returns ``(stack, count)``: ``stack`` is ``(D, *shape)`` float64 where
    ``D`` is the largest number of valid samples any cell received, NaN
    past each cell's ``count``. Memory is ``O(D * cells)`` — bounded by
    the overlap, not by the number of patches. The patches are walked
    once and their (cropped) data kept until the stack is filled.
    """
    placed: list[tuple[_Placement, Any]] = []
    count = np.zeros(shape, dtype=np.int64)
    for p in patches:
        pl = _resolve_indices(p.indices, shape)
        if pl is None:
            continue
        data = pl.crop(p.data)
        placed.append((pl, data))
        count[pl.acc] += pl.valid(_as_float(data))
    depth = int(count.max(initial=0))
    # One scratch slot past the deepest level absorbs the invalid samples.
    stack = np.full((depth + 1, *shape), np.nan, dtype=np.float64)
    level = np.zeros(shape, dtype=np.int64)
    for pl, data in placed:
        x = _as_float(data)
        lvl = level[pl.acc]
        valid = np.broadcast_to(pl.valid(x), lvl.shape)
        full = (slice(None), *pl.acc)
        block = stack[full]
        np.put_along_axis(
            block,
            np.where(valid, lvl, depth)[np.newaxis],
            np.broadcast_to(x, lvl.shape)[np.newaxis],
            axis=0,
        )
        stack[full] = block
        level[pl.acc] = lvl + valid
    return stack[:depth], count


@dataclass(eq=False)
class Median(Aggregation):
    """Per-cell median — exact, requires per-cell history.

    NaN and masked samples are skipped. The per-cell history is a stack
    as deep as the largest overlap (not one full-domain layer per patch).

    ``streaming_safe = False`` and there is no streamable per-cell
    substitute. ``spatial.aggregation.ApproxQuantile(q=0.5)`` is *not* one: it is a
    global sketch returning one scalar for the whole field.

    Args:
        fill_value: Written into cells no valid sample reached. Default
            NaN; pass e.g. the domain's nodata to override.
    """

    fill_value: float = math.nan

    streaming_safe: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        stack, count = _overlap_stack(patches, shape)
        if stack.shape[0] == 0:
            return np.full(shape, self.fill_value, dtype=np.float64)
        with warnings.catch_warnings():
            # Uncovered cells are an all-NaN column; they get the fill below.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(stack, axis=0)
        return _with_fill(median, count > 0, self.fill_value)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Mode(Aggregation):
    """Per-cell exact mode — not streamable. Use `spatial.aggregation.HardVote` for
    streaming.

    NaN and masked samples are skipped. **Ties** go to the smallest
    value. The per-cell history is a stack as deep as the largest
    overlap, and the mode is computed vectorised over it.

    Args:
        fill_value: Written into cells no valid sample reached. Default
            NaN, so the output is float64; an integral fill (e.g. ``-1``)
            makes it ``int64`` — values are then truncated to integers,
            which suits the class labels a mode is meant for.
    """

    fill_value: int | float = math.nan

    streaming_safe: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape, patches = _dense_layout(patches, domain)
        stack, count = _overlap_stack(patches, shape)
        out = _with_fill(_arraywise_mode(stack), count > 0, self.fill_value)
        return out.astype(_label_dtype(self.fill_value))

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _arraywise_mode(stack: np.ndarray) -> np.ndarray:
    """Mode along axis 0 of a NaN-padded float stack; ties → smallest value.

    Sort each column, measure the running length of every run of equal
    values, and take the value where that length first peaks: the first
    run to reach the longest length is the smallest of the tied values.
    All-NaN columns return NaN.
    """
    if stack.shape[0] == 0:
        return np.full(stack.shape[1:], np.nan, dtype=np.float64)
    ordered = np.sort(stack, axis=0)  # NaN sorts last
    pos = np.arange(ordered.shape[0]).reshape(-1, *([1] * (ordered.ndim - 1)))
    new_run = np.ones(ordered.shape, dtype=bool)
    new_run[1:] = ordered[1:] != ordered[:-1]
    run_start = np.maximum.accumulate(np.where(new_run, pos, 0), axis=0)
    run_length = np.where(np.isnan(ordered), 0, pos - run_start + 1)
    best = np.argmax(run_length, axis=0)[np.newaxis]
    return np.take_along_axis(ordered, best, axis=0)[0]


@dataclass(eq=False)
class Learned(Aggregation):
    """Caller-supplied merge — ``model(patches, domain) -> field``.

    Carries closures, so ``forbid_in_yaml = True``. Streaming behaviour is
    the model's responsibility, so ``streaming_safe = False``.
    """

    model: Callable[[Iterable[Any], Any], Any]

    streaming_safe: ClassVar[bool] = False
    forbid_in_yaml: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        return self.model(patches, domain)


# ---------------------------------------------------------------------------
# Approximate streaming sketches
# ---------------------------------------------------------------------------


class _SketchAggregation(Aggregation):
    """Shared ``merge(patches, domain)`` loop for global sketch reducers.

    Sketches are **global** reducers: ``merge`` folds every finite value of
    every patch into one bounded summary and returns a scalar / dict / small
    array describing the whole field — never an ``(H, W)`` field. They are
    not per-cell substitutes for `spatial.aggregation.Median` /
    `spatial.aggregation.Mode`; for a
    streamable per-cell majority use `spatial.aggregation.HardVote`.

    Every ``merge(patches)`` call starts from fresh state (``_reset``), so
    reusing one instance across ``merge()`` / ``reduce()`` calls never
    leaks values from an earlier call. Incremental accumulation goes
    through ``update`` / ``update_many`` / ``merge_state`` (or
    ``merge(other_sketch)``), which do not reset.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any = None) -> Any:
        del domain
        if isinstance(patches, self.__class__):
            self.merge_state(patches)
            return self
        self._reset()
        for patch in patches:
            self.update(patch)
        return self.finalize()

    def _reset(self) -> None:
        """Rebuild all accumulator state (and RNG, if any) from the config.

        Run at each ``merge(patches)`` entry and from ``__post_init__``.
        Stochastic sketches rebuild ``default_rng(seed)`` here — the same
        convention as the samplers, which rebuild it per call.
        """
        raise NotImplementedError

    def update(self, patch: Any) -> None:
        self.update_many(_patch_values(patch))

    def update_many(self, values: Iterable[Any]) -> None:
        raise NotImplementedError

    def finalize(self) -> Any:
        raise NotImplementedError

    def merge_state(self, other: Any) -> None:
        """Fold another sketch of the same type and configuration into this one."""
        raise NotImplementedError

    def _check_mergeable(self, other: Any, *names: str) -> None:
        if type(other) is not type(self):
            raise TypeError(
                f"cannot merge {type(other).__name__} into {type(self).__name__}"
            )
        for name in names:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                raise ValueError(
                    f"cannot merge {type(self).__name__} sketches with different "
                    f"{name} ({mine!r} vs {theirs!r})"
                )


def _patch_values(patch: Any) -> np.ndarray:
    array = np.asarray(patch.data)
    values = array.reshape(-1)
    return (
        values[np.isfinite(values)] if np.issubdtype(array.dtype, np.number) else values
    )


class _ReservoirSketch(_SketchAggregation):
    """Algorithm R reservoir of capacity ``k`` plus an exact uniform union.

    Subclasses are dataclasses declaring ``k: int`` and ``seed: int | None``.
    """

    k: int
    seed: int | None
    _sample: list[Any]
    _seen: int
    _rng: np.random.Generator

    def _reset(self) -> None:
        self._sample = []
        self._seen = 0
        self._rng = np.random.default_rng(self.seed)

    def _coerce(self, value: Any) -> Any:
        return _python_scalar(value)

    def update_many(self, values: Iterable[Any]) -> None:
        for value in values:
            item = self._coerce(value)
            self._seen += 1
            if len(self._sample) < self.k:
                self._sample.append(item)
                continue
            j = int(self._rng.integers(0, self._seen))
            if j < self.k:
                self._sample[j] = item

    def merge_state(self, other: Any) -> None:
        """Replace the sample with a uniform ``k``-sample of the union.

        Each reservoir is a uniform ``min(k, seen)``-subset of its stream.
        The number of merged slots drawn from ``self`` is hypergeometric in
        the two stream sizes (drawn slot by slot, so it is exact for any
        ``seen``), and those slots are a uniform subset of ``self``'s
        sample (likewise for ``other``). The result is a uniform
        ``min(k, seen_a + seen_b)``-subset of the concatenated streams, and
        ``_seen`` becomes ``seen_a + seen_b`` so Algorithm R continues
        correctly afterwards.
        """
        self._check_mergeable(other, "k")
        n_a, n_b = self._seen, other._seen
        size = min(self.k, n_a + n_b)
        take_a = 0
        rem_a, rem_b = n_a, n_b
        for _ in range(size):
            if self._rng.random() * (rem_a + rem_b) < rem_a:
                take_a += 1
                rem_a -= 1
            else:
                rem_b -= 1
        take_b = size - take_a
        pick_a = self._rng.choice(len(self._sample), size=take_a, replace=False)
        pick_b = self._rng.choice(len(other._sample), size=take_b, replace=False)
        merged = [self._sample[i] for i in pick_a] + [other._sample[i] for i in pick_b]
        self._rng.shuffle(merged)
        self._sample = merged
        self._seen = n_a + n_b


@dataclass(eq=False)
class ApproxQuantile(_ReservoirSketch):
    """Global approximate quantile(s) from a uniform reservoir of ``k`` values.

    A **global** reducer: ``merge`` returns ``{str(float(q)): value}`` for
    the whole field, not a per-cell quantile field. There is no streamable
    per-cell median; `spatial.aggregation.Median` stays the exact in-RAM option.

    Args:
        q: Quantile or list of quantiles in ``[0, 1]`` (int or float).
        k: Reservoir size — number of values retained for the estimate.
        seed: RNG seed. ``None`` (the default, as for the samplers) draws
            fresh OS entropy on every ``merge``; pass an int to make
            repeated ``merge()`` calls on one instance reproducible.
    """

    q: float | list[float] = 0.5
    k: int = 200
    seed: int | None = None
    _sample: list[Any] = field(default_factory=list, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        for q in self._quantiles():
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"q must be in [0, 1], got {q!r}")
        self._reset()

    def _quantiles(self) -> list[float]:
        if isinstance(self.q, (list, tuple)):
            return [float(q) for q in self.q]
        return [float(self.q)]

    def _coerce(self, value: Any) -> Any:
        return float(value)

    def finalize(self) -> dict[str, float]:
        if not self._sample:
            return {}
        values = np.asarray(self._sample, dtype=np.float64)
        return {str(q): float(np.quantile(values, q)) for q in self._quantiles()}

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _hll_alpha(m: int) -> float:
    """HyperLogLog bias-correction constant ``alpha_m`` (Flajolet et al. 2007)."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


@dataclass(eq=False)
class ApproxCardinality(_SketchAggregation):
    """Global approximate unique-value count via HyperLogLog.

    A **global** reducer: ``merge`` returns one float for the whole field.
    """

    p: int = 14
    _registers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 4 <= self.p <= 16:
            raise ValueError("p must be between 4 and 16")
        self._reset()

    def _reset(self) -> None:
        self._registers = np.zeros(1 << self.p, dtype=np.uint8)

    def update_many(self, values: Iterable[Any]) -> None:
        for value in values:
            h = _hash64(value)
            idx = h & ((1 << self.p) - 1)
            w = h >> self.p
            rank = (
                (HASH_BITS - self.p) - w.bit_length() + 1
                if w
                else HASH_BITS - self.p + 1
            )
            self._registers[idx] = max(int(self._registers[idx]), rank)

    def finalize(self) -> float:
        m = 1 << self.p
        estimate = (
            _hll_alpha(m) * m * m / np.sum(2.0 ** (-self._registers.astype(float)))
        )
        zeros = int(np.count_nonzero(self._registers == 0))
        if estimate <= 2.5 * m and zeros > 0:
            estimate = m * math.log(m / zeros)
        return float(estimate)

    def merge_state(self, other: ApproxCardinality) -> None:
        self._check_mergeable(other, "p")
        self._registers = np.maximum(self._registers, other._registers)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class ApproxMode(_SketchAggregation):
    """Global approximate heavy hitters via Misra-Gries counters.

    A **global** reducer: ``merge`` returns ``{value: count}`` for the
    whole field, not a per-cell mode. For a streamable per-cell majority
    use `spatial.aggregation.HardVote`.
    """

    k: int = 16
    _counts: dict[Any, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")

    def _reset(self) -> None:
        self._counts = {}

    def update_many(self, values: Iterable[Any]) -> None:
        for value in values:
            key = _python_scalar(value)
            if key in self._counts:
                self._counts[key] += 1
            elif len(self._counts) < self.k:
                self._counts[key] = 1
            else:
                drop = []
                for item in self._counts:
                    self._counts[item] -= 1
                    if self._counts[item] == 0:
                        drop.append(item)
                for item in drop:
                    del self._counts[item]

    def finalize(self) -> dict[Any, int]:
        return dict(
            sorted(self._counts.items(), key=lambda item: item[1], reverse=True)
        )

    def merge_state(self, other: ApproxMode) -> None:
        """Mergeable Misra-Gries union (Agarwal et al. 2012).

        Sum the counters; if more than ``k`` survive, subtract the
        ``(k + 1)``-th largest count from all and drop the non-positive ones.
        """
        self._check_mergeable(other, "k")
        counts = dict(self._counts)
        for key, count in other._counts.items():
            counts[key] = counts.get(key, 0) + count
        if len(counts) > self.k:
            cut = sorted(counts.values(), reverse=True)[self.k]
            counts = {key: c - cut for key, c in counts.items() if c > cut}
        self._counts = counts

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class StreamingHistogram(_SketchAggregation):
    """Global online histogram with at most ``bins`` centroids.

    A **global** reducer: ``merge`` returns ``{"centers", "counts"}`` for
    the whole field.
    """

    bins: int = 64
    _centers: list[float] = field(default_factory=list, init=False, repr=False)
    _counts: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("bins must be >= 1")

    def _reset(self) -> None:
        self._centers = []
        self._counts = []

    def update_many(self, values: Iterable[Any]) -> None:
        for value in values:
            self._centers.append(float(value))
            self._counts.append(1)
            while len(self._centers) > self.bins:
                self._merge_closest_bins()

    def finalize(self) -> dict[str, np.ndarray]:
        order = np.argsort(self._centers)
        return {
            "centers": np.asarray(self._centers, dtype=np.float64)[order],
            "counts": np.asarray(self._counts, dtype=np.int64)[order],
        }

    def merge_state(self, other: StreamingHistogram) -> None:
        """Add ``other``'s weighted centroids, then re-compress to ``bins``."""
        self._check_mergeable(other, "bins")
        self._centers = [*self._centers, *other._centers]
        self._counts = [*self._counts, *other._counts]
        while len(self._centers) > self.bins:
            self._merge_closest_bins()

    def _merge_closest_bins(self) -> None:
        order = np.argsort(self._centers)
        centers = [self._centers[i] for i in order]
        counts = [self._counts[i] for i in order]
        idx = min(
            range(len(centers) - 1),
            key=lambda i: abs(centers[i + 1] - centers[i]),
        )
        count = counts[idx] + counts[idx + 1]
        center = (
            centers[idx] * counts[idx] + centers[idx + 1] * counts[idx + 1]
        ) / count
        centers[idx : idx + 2] = [center]
        counts[idx : idx + 2] = [count]
        self._centers = centers
        self._counts = counts

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class Reservoir(_ReservoirSketch):
    """Uniform global reservoir sample of ``k`` values (Vitter's Algorithm R).

    A **global** reducer: ``merge`` returns a 1-D array of at most ``k``
    values drawn uniformly from every finite value of every patch.

    Args:
        k: Reservoir size.
        seed: RNG seed. ``None`` (the default, as for the samplers) draws
            fresh OS entropy on every ``merge``; pass an int to make
            repeated ``merge()`` calls on one instance reproducible.
    """

    k: int = 100
    seed: int | None = None
    _sample: list[Any] = field(default_factory=list, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        self._reset()

    def finalize(self) -> np.ndarray:
        return np.asarray(self._sample)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _hash64(value: Any) -> int:
    key = repr(_python_scalar(value)).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------------------
# Streaming-safety check (used by SpatialPatcher.merge)
# ---------------------------------------------------------------------------


def _warn_if_unsafe_streaming(aggregation: Any, *, stacklevel: int = 2) -> None:
    """Warn (or raise under `set_strict`) for a non-streaming aggregation.

    Serves every patcher family: ``aggregation`` is anything with a
    ``streaming_safe`` flag (spatial or temporal). ``stacklevel`` is
    forwarded to `warnings.warn`, so callers can point the warning at the
    user's line rather than at geopatcher internals.
    """
    if getattr(aggregation, "streaming_safe", False):
        return
    from geopatcher._src.config import get_strict

    msg = (
        f"{type(aggregation).__name__} has streaming_safe = False — "
        "the merge is happening in-RAM."
    )
    if isinstance(aggregation, Aggregation):
        msg += (
            " Per-cell streaming alternatives: "
            "Mode->HardVote, Learned->patcher.two_pass; Median has none (the "
            "Approx* sketches are global reducers returning one summary for "
            "the whole field, not an (H, W) field)."
        )
    msg += " See docs/patcher/patching.md §'Streaming aggregations'."
    if get_strict():
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)
