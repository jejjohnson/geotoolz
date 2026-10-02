"""`SpatialAggregation` — local patch results → global field.

The aggregation step is the inverse of `split`: take an iterable of
patches (each annotated with its indices and weights) and reconstruct a
single global field. The streaming-safe families are monoidal folds
over one or more accumulators; the non-streaming ones (`SpatialMedian`,
`SpatialMode`, `SpatialLearned`) need a per-cell history and accumulate in memory.

The `streaming_safe` class flag advertises which aggregations support
the disk-backed path (a target zarr / memmap) — `SpatialOverlapAdd` is the
canonical streaming-safe member; `SpatialMedian` triggers a warning if the
caller asks for streaming.

See ``docs/patching.md`` §"Streaming aggregations" for the framing.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from geopatcher._src._serialize import config_from_fields


COG_WRITER = "cog"
DEFAULT_COG_BLOCKSIZE = 512
HASH_BITS = 64


# ---------------------------------------------------------------------------
# Base + helpers
# ---------------------------------------------------------------------------


class SpatialAggregation:
    """Base for local → global merging strategies.

    Subclasses implement ``merge(patches, domain) -> field-like``.

    ``streaming_safe = True`` means the aggregation can be reduced one
    patch at a time without keeping per-cell history; the disk-backed
    `SpatialOverlapAdd(streaming=True, target_path=...)` path lights up
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
    don't apply, and the caller will land in `SpatialByIndex` instead.
    """
    if hasattr(domain, "shape"):
        return tuple(domain.shape)
    raise TypeError(f"can't infer dense shape from {type(domain).__name__}")


def _replace_nan_with_zero(array: np.ndarray) -> np.ndarray:
    """Replace only NaN with zero; preserve infinities as real values."""
    return np.where(np.isnan(array), 0.0, array)


@dataclass(frozen=True)
class _Placement:
    """Where a patch lands in the accumulator, cropped to the domain.

    A patch's ``indices`` may overhang the domain (``boundary="pad"`` /
    ``"reflect"`` chips, or any caller-built window), while numpy would
    silently clip a too-long slice — and read a negative start from the
    end. Every dense aggregation therefore writes only the in-domain
    part: ``acc[placement.acc] ⊕= f(placement.crop(data))``, with the
    weights cropped by the same ``placement.crop``.

    Attributes:
        acc: Slicer into the domain-shaped accumulator, clipped to
            ``[0, n)`` on every sliced axis.
        chip: Matching slicer into the patch's own data / weights —
            the offset of the in-domain part within the chip.
    """

    acc: tuple[Any, ...]
    chip: tuple[Any, ...]

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
    - ``_MaskedWindow``: resolve the underlying rasterio Window

    Returns ``None`` when the indices are not a dense placement (the
    caller skips the patch) or when the patch has no in-domain cell.
    """
    from geopatcher._src.spatial.geometry import _MaskedWindow

    if isinstance(indices, _MaskedWindow):
        return _resolve_indices(indices.window, shape)
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
    return None


# ---------------------------------------------------------------------------
# Exact streaming aggregations
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SpatialSum(SpatialAggregation):
    """Per-cell sum across patches."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            acc[sl] += _replace_nan_with_zero(
                np.asarray(pl.crop(p.data), dtype=np.float64)
            )
        return acc


@dataclass(eq=False)
class SpatialMax(SpatialAggregation):
    """Per-cell maximum across patches."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.full(shape, -np.inf, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            # fmax ignores NaN to support masked patches.
            acc[sl] = np.fmax(acc[sl], np.asarray(pl.crop(p.data), dtype=np.float64))
        return acc


@dataclass(eq=False)
class SpatialMin(SpatialAggregation):
    """Per-cell minimum across patches."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.full(shape, np.inf, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            # fmin ignores NaN to support masked patches.
            acc[sl] = np.fmin(acc[sl], np.asarray(pl.crop(p.data), dtype=np.float64))
        return acc


@dataclass(eq=False)
class SpatialWeightedSum(SpatialAggregation):
    """Per-cell weighted sum — each patch's window weights multiply in.

    Args:
        weight_fn: Optional callable ``(patch) -> Array`` overriding the
            per-patch weights. ``None`` uses ``patch.weights`` directly.
    """

    weight_fn: Callable[[Any], np.ndarray] | None = None

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            w = pl.crop(self.weight_fn(p) if self.weight_fn else p.weights)
            data = np.asarray(pl.crop(p.data), dtype=np.float64)
            if w is None:
                acc[sl] += _replace_nan_with_zero(data)
            else:
                acc[sl] += _replace_nan_with_zero(
                    data * np.asarray(w, dtype=np.float64)
                )
        return acc


# ---------------------------------------------------------------------------
# Compound monoidal aggregations
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SpatialMean(SpatialAggregation):
    """Per-cell mean — runs `SpatialSum` and a count accumulator in parallel."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        total = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            data = np.asarray(pl.crop(p.data), dtype=np.float64)
            valid = ~np.isnan(data)
            total[sl] += _replace_nan_with_zero(data)
            count[sl] += valid
        with np.errstate(invalid="ignore"):
            return np.where(count > 0, total / count, 0.0)


@dataclass(eq=False)
class SpatialVariance(SpatialAggregation):
    """Per-cell sample variance via Welford's online algorithm.

    Returns the unbiased estimate (``ddof=1``); cells touched fewer than
    two patches return ``0.0``.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        mean = np.zeros(shape, dtype=np.float64)
        m2 = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            x = np.asarray(pl.crop(p.data), dtype=np.float64)
            valid = ~np.isnan(x)
            x_clean = _replace_nan_with_zero(x)
            next_count = count[sl] + valid
            delta = np.where(valid, x_clean - mean[sl], 0.0)
            denom = np.where(next_count > 0, next_count, 1.0)
            mean[sl] += np.where(next_count > 0, delta / denom, 0.0)
            m2[sl] += np.where(valid, delta * (x_clean - mean[sl]), 0.0)
            count[sl] = next_count
        with np.errstate(invalid="ignore"):
            return np.where(count > 1, m2 / (count - 1), 0.0)


@dataclass(eq=False)
class SpatialMeanStd(SpatialAggregation):
    """Global mean and sample standard deviation across patch data."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, float]:
        count = 0
        mean = 0.0
        m2 = 0.0
        for p in patches:
            x = np.asarray(p.data, dtype=np.float64).reshape(-1)
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
            raise ValueError("SpatialMeanStd requires at least one value")
        var = m2 / (count - 1) if count > 1 else 0.0
        return {"mean": mean, "std": float(np.sqrt(var))}


@dataclass(eq=False)
class SpatialMinMax(SpatialAggregation):
    """Global minimum and maximum across patch data."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, float]:
        min_value = np.inf
        max_value = -np.inf
        seen = False
        for p in patches:
            x = np.asarray(p.data, dtype=np.float64)
            if x.size == 0:
                continue
            min_value = min(min_value, float(np.min(x)))
            max_value = max(max_value, float(np.max(x)))
            seen = True
        if not seen:
            raise ValueError("SpatialMinMax requires at least one value")
        return {"min": min_value, "max": max_value}


@dataclass(eq=False)
class SpatialOverlapAdd(SpatialAggregation):
    """SpatialWindow-weighted overlap-add — the canonical chip-stitching aggregator.

    Computes ``Σ wᵢ xᵢ / Σ wᵢ`` per cell. When stride < patch size and a
    `SpatialHann` (or similar partition-of-unity) window is used, the resulting
    field equals the original (modulo the operator's effect) — the
    standard inference-time stitching pattern.

    Args:
        streaming: If ``True`` and ``target_path`` is set, accumulate
            into a chunked on-disk zarr store rather than in RAM.
        target_path: Filesystem path for the disk-backed accumulators.
        chunks: Zarr chunk shape for the streaming accumulators. When
            ``None`` (the default), the chunk shape is derived from the
            first patch's data shape, right-aligned against the domain
            shape so any leading band/time dims pick up their full extent
            as a chunk. Provide an explicit value to override.
        normalize_by_window: Divide by the accumulated weight at the end
            (default ``True``). Set to ``False`` for the raw weighted
            sum.
    """

    streaming: bool = False
    target_path: str | None = None
    chunks: tuple[int, ...] | None = None
    shard_shape: tuple[int, ...] | None = None
    writer: str = "zarr"
    cog: dict[str, Any] | None = None
    normalize_by_window: bool = True

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        if self.streaming and self.target_path and self.writer == COG_WRITER:
            result = self._merge_in_memory(patches, domain)
            return _write_cog(result, domain, self.target_path, self.cog)
        if self.streaming and self.target_path:
            return self._merge_streaming(patches, domain)
        return self._merge_in_memory(patches, domain)

    def _merge_in_memory(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.zeros(shape, dtype=np.float64)
        wsum = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            x = np.asarray(pl.crop(p.data), dtype=np.float64)
            w = (
                np.asarray(pl.crop(p.weights), dtype=np.float64)
                if p.weights is not None
                else np.ones_like(x)
            )
            valid = ~np.isnan(x)
            acc[sl] += _replace_nan_with_zero(x) * w * valid
            wsum[sl] += w * valid
        if not self.normalize_by_window:
            return acc
        with np.errstate(invalid="ignore"):
            return np.where(wsum > 0, acc / wsum, 0.0)

    def _merge_streaming(self, patches: Iterable[Any], domain: Any) -> Any:
        import itertools

        shape = _domain_array_shape(domain)
        # Peek the first patch so the default chunk shape matches its data
        # shape, rather than degenerating to the whole-array chunk that
        # would defeat the whole point of streaming.
        patches_iter = iter(patches)
        try:
            first = next(patches_iter)
        except StopIteration:
            # No patches → return an empty zero-filled zarr array.
            return _open_zarr_array(
                f"{self.target_path}/rec.zarr",
                shape=shape,
                chunks=shape,
                dtype="float32",
                fill_value=0.0,
                shard_shape=self.shard_shape,
            )
        first_data = np.asarray(first.data)
        # Right-align the data shape against the domain shape so leading
        # band/time dims pick up their full extent as a chunk.
        if self.chunks is not None:
            chunks: tuple[int, ...] = tuple(self.chunks)
        else:
            chunks = tuple(shape[: -len(first_data.shape)]) + tuple(first_data.shape)
        # `zarr.open` returns `Array | Group`; with mode="w" and a `shape`/`dtype`
        # it always returns an Array — but ty can't narrow that, so we cast.
        rec = _open_zarr_array(
            f"{self.target_path}/rec.zarr",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=0.0,
            shard_shape=self.shard_shape,
        )
        wsum = _open_zarr_array(
            f"{self.target_path}/wsum.zarr",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=0.0,
            shard_shape=self.shard_shape,
        )
        # Push the peeked patch back to the front of the iterator.
        patches = itertools.chain([first], patches_iter)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            x = np.asarray(pl.crop(p.data), dtype=np.float32)
            w = (
                np.asarray(pl.crop(p.weights), dtype=np.float32)
                if p.weights is not None
                else np.ones_like(x)
            )
            valid = ~np.isnan(x)
            rec[sl] = np.asarray(rec[sl]) + _replace_nan_with_zero(x) * w * valid
            wsum[sl] = np.asarray(wsum[sl]) + w * valid
        if self.normalize_by_window:
            arr = np.asarray(rec[:])
            wt = np.asarray(wsum[:])
            with np.errstate(invalid="ignore"):
                rec[:] = np.where(wt > 0, arr / wt, 0.0)
        return rec

    def get_config(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "target_path": self.target_path,
            "chunks": list(self.chunks) if self.chunks else None,
            "shard_shape": list(self.shard_shape) if self.shard_shape else None,
            "writer": self.writer,
            "cog": self.cog,
            "normalize_by_window": self.normalize_by_window,
        }


def _open_zarr_array(
    path: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    fill_value: float,
    shard_shape: tuple[int, ...] | None,
) -> Any:
    import zarr

    if shard_shape is not None:
        # Sharded creation goes through the zarr v3 `create_array` API —
        # `zarr.open` has no `shards` parameter on any zarr release, so
        # routing shards through it silently produced unsharded stores.
        create_array = getattr(zarr, "create_array", None)
        if create_array is not None:
            return create_array(
                path,
                shape=shape,
                chunks=chunks,
                shards=shard_shape,
                dtype=dtype,
                fill_value=fill_value,
                overwrite=True,
            )
        warnings.warn(
            "installed zarr does not support sharding (requires zarr >= 3); "
            "writing unsharded output",
            RuntimeWarning,
            stacklevel=2,
        )
    return zarr.open(
        path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
    )


def _write_cog(
    array: np.ndarray, domain: Any, target_path: str, cog: dict[str, Any] | None
) -> str:
    import rasterio

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        write_data = data[np.newaxis, ...]
    elif data.ndim == 3:
        write_data = data
    else:
        raise ValueError("COG writer expects a 2-D array or a 3-D band-first array")

    options = dict(cog or {})
    blocksize = options.pop("blocksize", DEFAULT_COG_BLOCKSIZE)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": write_data.shape[-2],
        "width": write_data.shape[-1],
        "count": write_data.shape[0],
        "dtype": "float32",
        "crs": getattr(domain, "crs", None),
        "transform": getattr(domain, "transform", rasterio.Affine.identity()),
        "tiled": True,
        "compress": options.pop("compress", "DEFLATE"),
        "blockxsize": blocksize,
        "blockysize": blocksize,
        # GDAL creation option casing.
        "BIGTIFF": options.pop("bigtiff", "IF_SAFER"),
    }
    profile.update(options)
    with rasterio.open(target_path, "w", **profile) as dst:
        dst.write(write_data)
    return target_path


@dataclass(eq=False)
class SpatialInvVarWeightedMean(SpatialAggregation):
    """Bayesian inverse-variance weighting for overlapping local posteriors.

    Each patch produces ``(mu, var)`` — i.e. ``patch.data`` is a tuple or
    a dict with ``"mu"`` / ``"var"`` keys. Returns a dict ``{"mu":
    global_mu, "var": global_var}`` with Kalman-optimal combination::

        μ_global = Σ wᵢ μᵢ / σᵢ² / Σ wᵢ / σᵢ²
        σ²_global = 1 / Σ wᵢ / σᵢ²
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[str, np.ndarray]:
        shape = _domain_array_shape(domain)
        mu_acc = np.zeros(shape, dtype=np.float64)
        prec = np.zeros(shape, dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            mu, var = (pl.crop(part) for part in _unpack_mu_var(p.data))
            w = (
                np.asarray(pl.crop(p.weights), dtype=np.float64)
                if p.weights is not None
                else np.ones_like(np.asarray(mu))
            )
            inv_var = w / np.asarray(var, dtype=np.float64)
            mu_acc[sl] += inv_var * np.asarray(mu, dtype=np.float64)
            prec[sl] += inv_var
        with np.errstate(invalid="ignore"):
            mu_g = np.where(prec > 0, mu_acc / prec, 0.0)
            var_g = np.where(prec > 0, 1.0 / prec, np.inf)
        return {"mu": mu_g, "var": var_g}


def _unpack_mu_var(data: Any) -> tuple[Any, Any]:
    if isinstance(data, tuple):
        return data
    if isinstance(data, dict):
        return data["mu"], data["var"]
    raise TypeError(
        "SpatialInvVarWeightedMean expects each patch's data to be a (mu, var) "
        "tuple or a {'mu': ..., 'var': ...} mapping."
    )


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SpatialHardVote(SpatialAggregation):
    """Per-cell majority vote — patches carry integer class predictions.

    Args:
        n_classes: Total number of classes ``K``. Accumulator shape is
            ``(K, *domain_shape)``.
    """

    n_classes: int

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        votes = np.zeros((self.n_classes, *shape), dtype=np.int64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            cls = np.asarray(pl.crop(p.data), dtype=np.int64)
            for k in range(self.n_classes):
                votes[k][sl] += (cls == k).astype(np.int64)
        return np.argmax(votes, axis=0)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialSoftVote(SpatialAggregation):
    """Per-cell soft vote — patches carry per-class probabilities.

    Each patch's data has shape ``(n_classes, ...)``; the per-class
    probabilities accumulate, and the final argmax across the class axis
    is returned.
    """

    n_classes: int

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.zeros((self.n_classes, *shape), dtype=np.float64)
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            # The leading class axis is kept whole on both sides.
            probs = np.asarray(p.data, dtype=np.float64)[(slice(None), *pl.chip)]
            full_sl = (slice(None), *sl)
            acc[full_sl] += probs
        return np.argmax(acc, axis=0)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SpatialByIndex(SpatialAggregation):
    """Don't merge — return a ``{anchor: data}`` mapping.

    The natural choice for ragged geometries (`SpatialRadiusGraph`,
    `SpatialKNNGraph`, `SpatialPolygonIntersection`) where the per-patch outputs
    aren't laid out on a regular grid.
    """

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> dict[Any, Any]:
        out: dict[Any, Any] = {}
        for p in patches:
            out[p.anchor] = p.data
        return out


# ---------------------------------------------------------------------------
# Honest non-streamable
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SpatialMedian(SpatialAggregation):
    """Per-cell median — exact, requires per-cell history.

    ``streaming_safe = False`` and there is no streamable per-cell
    substitute. ``SpatialApproxQuantile(q=0.5)`` is *not* one: it is a
    global sketch returning one scalar for the whole field.
    """

    streaming_safe: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        bucket: list[np.ndarray] = []
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            full = np.full(shape, np.nan, dtype=np.float64)
            full[sl] = np.asarray(pl.crop(p.data), dtype=np.float64)
            bucket.append(full)
        if not bucket:
            return np.zeros(shape, dtype=np.float64)
        stack = np.stack(bucket, axis=0)
        return np.nanmedian(stack, axis=0)


@dataclass(eq=False)
class SpatialMode(SpatialAggregation):
    """Per-cell exact mode — not streamable. Use `SpatialHardVote` for streaming."""

    streaming_safe: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        bucket: list[np.ndarray] = []
        for p in patches:
            pl = _resolve_indices(p.indices, shape)
            if pl is None:
                continue
            sl = pl.acc
            full = np.full(shape, np.iinfo(np.int64).min, dtype=np.int64)
            full[sl] = np.asarray(pl.crop(p.data), dtype=np.int64)
            bucket.append(full)
        if not bucket:
            return np.zeros(shape, dtype=np.int64)
        stack = np.stack(bucket, axis=0)
        return _arraywise_mode(stack)


def _arraywise_mode(stack: np.ndarray) -> np.ndarray:
    """SpatialMode along axis 0 of an integer stack, ignoring the sentinel."""
    sentinel = np.iinfo(np.int64).min
    out = np.zeros(stack.shape[1:], dtype=np.int64)
    flat = stack.reshape(stack.shape[0], -1)
    for i in range(flat.shape[1]):
        col = flat[:, i]
        col = col[col != sentinel]
        if col.size == 0:
            continue
        vals, counts = np.unique(col, return_counts=True)
        out.reshape(-1)[i] = int(vals[np.argmax(counts)])
    return out


@dataclass(eq=False)
class SpatialLearned(SpatialAggregation):
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


class _SketchAggregation(SpatialAggregation):
    """Shared ``merge(patches, domain)`` loop for global sketch reducers.

    Sketches are **global** reducers: ``merge`` folds every finite value of
    every patch into one bounded summary and returns a scalar / dict / small
    array describing the whole field — never an ``(H, W)`` field. They are
    not per-cell substitutes for `SpatialMedian` / `SpatialMode`; for a
    streamable per-cell majority use `SpatialHardVote`.

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
class SpatialApproxQuantile(_ReservoirSketch):
    """Global approximate quantile(s) from a uniform reservoir of ``k`` values.

    A **global** reducer: ``merge`` returns ``{str(float(q)): value}`` for
    the whole field, not a per-cell quantile field. There is no streamable
    per-cell median; `SpatialMedian` stays the exact in-RAM option.

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
class SpatialApproxCardinality(_SketchAggregation):
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

    def merge_state(self, other: SpatialApproxCardinality) -> None:
        self._check_mergeable(other, "p")
        self._registers = np.maximum(self._registers, other._registers)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


@dataclass(eq=False)
class SpatialApproxMode(_SketchAggregation):
    """Global approximate heavy hitters via Misra-Gries counters.

    A **global** reducer: ``merge`` returns ``{value: count}`` for the
    whole field, not a per-cell mode. For a streamable per-cell majority
    use `SpatialHardVote`.
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

    def merge_state(self, other: SpatialApproxMode) -> None:
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
class SpatialStreamingHistogram(_SketchAggregation):
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

    def merge_state(self, other: SpatialStreamingHistogram) -> None:
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
class SpatialReservoir(_ReservoirSketch):
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


def _warn_if_unsafe_streaming(aggregation: SpatialAggregation) -> None:
    if aggregation.streaming_safe:
        return
    from geopatcher._src.config import get_strict

    msg = (
        f"{type(aggregation).__name__} has streaming_safe = False — "
        "the merge is happening in-RAM. Per-cell streaming alternatives: "
        "Mode->HardVote, Learned->patcher.two_pass; Median has none (the "
        "Approx* sketches are global reducers returning one summary for the "
        "whole field, not an (H, W) field). See docs/patcher/patching.md "
        "§'Streaming aggregations'."
    )
    if get_strict():
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
