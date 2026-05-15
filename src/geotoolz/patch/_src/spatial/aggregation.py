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

See ``scaling.md`` §"Streaming-Compatible Aggregations" for the
mathematical framing.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


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


def _resolve_indices(indices: Any) -> tuple[Any, ...] | None:
    """Map a patch's indices object to a numpy slicer.

    Returns a tuple that targets the **trailing** axes of the accumulator —
    leading band/time dims pass through via ``Ellipsis``. So
    ``acc[_resolve_indices(idx)]`` selects ``(..., row_slice, col_slice)``
    on a 3-D `(band, H, W)` array, not just `(row, col, :)`.

    - rasterio.windows.Window: ``(..., row_slice, col_slice)``
    - ``dict[str, slice]``: ``tuple(values)`` in dict order (no ellipsis)
    - ``_MaskedWindow``: resolve the underlying rasterio Window
    - ``None``: return ``None`` (caller falls back to anchor dispatch)
    """
    from geotoolz.patch._src.spatial.geometry import _MaskedWindow

    if hasattr(indices, "row_off") and hasattr(indices, "col_off"):
        r0 = int(indices.row_off)
        c0 = int(indices.col_off)
        h = int(indices.height)
        w = int(indices.width)
        return (Ellipsis, slice(r0, r0 + h), slice(c0, c0 + w))
    if isinstance(indices, _MaskedWindow):
        return _resolve_indices(indices.window)
    if isinstance(indices, dict):
        return tuple(indices.values())
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            acc[sl] += np.asarray(p.data, dtype=np.float64)
        return acc


@dataclass(eq=False)
class SpatialMax(SpatialAggregation):
    """Per-cell maximum across patches."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.full(shape, -np.inf, dtype=np.float64)
        for p in patches:
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            acc[sl] = np.maximum(acc[sl], np.asarray(p.data, dtype=np.float64))
        return acc


@dataclass(eq=False)
class SpatialMin(SpatialAggregation):
    """Per-cell minimum across patches."""

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.full(shape, np.inf, dtype=np.float64)
        for p in patches:
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            acc[sl] = np.minimum(acc[sl], np.asarray(p.data, dtype=np.float64))
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            w = self.weight_fn(p) if self.weight_fn else p.weights
            data = np.asarray(p.data, dtype=np.float64)
            if w is None:
                acc[sl] += data
            else:
                acc[sl] += data * np.asarray(w, dtype=np.float64)
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            total[sl] += np.asarray(p.data, dtype=np.float64)
            count[sl] += 1.0
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            x = np.asarray(p.data, dtype=np.float64)
            count[sl] += 1.0
            delta = x - mean[sl]
            mean[sl] += delta / count[sl]
            m2[sl] += delta * (x - mean[sl])
        with np.errstate(invalid="ignore"):
            return np.where(count > 1, m2 / (count - 1), 0.0)


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
    normalize_by_window: bool = True

    streaming_safe: ClassVar[bool] = True

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        if self.streaming and self.target_path:
            return self._merge_streaming(patches, domain)
        return self._merge_in_memory(patches, domain)

    def _merge_in_memory(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        acc = np.zeros(shape, dtype=np.float64)
        wsum = np.zeros(shape, dtype=np.float64)
        for p in patches:
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            w = (
                np.asarray(p.weights, dtype=np.float64)
                if p.weights is not None
                else np.ones_like(np.asarray(p.data))
            )
            x = np.asarray(p.data, dtype=np.float64)
            acc[sl] += x * w
            wsum[sl] += w
        if not self.normalize_by_window:
            return acc
        with np.errstate(invalid="ignore"):
            return np.where(wsum > 0, acc / wsum, 0.0)

    def _merge_streaming(self, patches: Iterable[Any], domain: Any) -> Any:
        import itertools

        import zarr

        shape = _domain_array_shape(domain)
        # Peek the first patch so the default chunk shape matches its data
        # shape, rather than degenerating to the whole-array chunk that
        # would defeat the whole point of streaming.
        patches_iter = iter(patches)
        try:
            first = next(patches_iter)
        except StopIteration:
            # No patches → return an empty zero-filled zarr array.
            return zarr.open(
                f"{self.target_path}/rec.zarr",
                mode="w",
                shape=shape,
                chunks=shape,
                dtype="float32",
                fill_value=0.0,
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
        rec: Any = zarr.open(
            f"{self.target_path}/rec.zarr",
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=0.0,
        )
        wsum: Any = zarr.open(
            f"{self.target_path}/wsum.zarr",
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=0.0,
        )
        # Push the peeked patch back to the front of the iterator.
        patches = itertools.chain([first], patches_iter)
        for p in patches:
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            w = (
                np.asarray(p.weights, dtype=np.float32)
                if p.weights is not None
                else np.ones_like(np.asarray(p.data), dtype=np.float32)
            )
            x = np.asarray(p.data, dtype=np.float32)
            rec[sl] = np.asarray(rec[sl]) + x * w
            wsum[sl] = np.asarray(wsum[sl]) + w
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
            "normalize_by_window": self.normalize_by_window,
        }


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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            mu, var = _unpack_mu_var(p.data)
            w = (
                np.asarray(p.weights, dtype=np.float64)
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            cls = np.asarray(p.data, dtype=np.int64)
            for k in range(self.n_classes):
                votes[k][sl] += (cls == k).astype(np.int64)
        return np.argmax(votes, axis=0)

    def get_config(self) -> dict[str, Any]:
        return {"n_classes": self.n_classes}


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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            probs = np.asarray(p.data, dtype=np.float64)
            full_sl = (slice(None), *sl)
            acc[full_sl] += probs
        return np.argmax(acc, axis=0)

    def get_config(self) -> dict[str, Any]:
        return {"n_classes": self.n_classes}


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

    ``streaming_safe = False``: callers asking for the disk-backed path
    get a warning pointing at ``SpatialApproxQuantile(q=0.5)`` as the streamable
    substitute.
    """

    streaming_safe: ClassVar[bool] = False

    def merge(self, patches: Iterable[Any], domain: Any) -> np.ndarray:
        shape = _domain_array_shape(domain)
        bucket: list[np.ndarray] = []
        for p in patches:
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            full = np.full(shape, np.nan, dtype=np.float64)
            full[sl] = np.asarray(p.data, dtype=np.float64)
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
            sl = _resolve_indices(p.indices)
            if sl is None:
                continue
            full = np.full(shape, np.iinfo(np.int64).min, dtype=np.int64)
            full[sl] = np.asarray(p.data, dtype=np.int64)
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
# Approximate streaming (stubs — full implementations in v0.2)
# ---------------------------------------------------------------------------


class _ApproxStub(SpatialAggregation):
    """Shared NotImplementedError for the sketch family."""

    streaming_safe: ClassVar[bool] = True

    _substitute: ClassVar[str] = ""

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} is reserved for v0.2 (needs the "
            f"{self._substitute} backend). Use the documented substitute or "
            "implement a `SpatialCustom`-style wrapper in the meantime."
        )


@dataclass(eq=False)
class SpatialApproxQuantile(_ApproxStub):
    """Streaming quantile via t-digest / KLL (placeholder).

    Use `SpatialMedian` (non-streaming) or supply a custom `SpatialAggregation` until
    v0.2 lands the t-digest backend.
    """

    q: float = 0.5
    _substitute: ClassVar[str] = "t-digest (e.g. `tdigest`)"


@dataclass(eq=False)
class SpatialApproxCardinality(_ApproxStub):
    """Streaming unique-value count via HyperLogLog (placeholder)."""

    p: int = 14
    _substitute: ClassVar[str] = "HyperLogLog (e.g. `datasketch.HyperLogLog`)"


@dataclass(eq=False)
class SpatialApproxMode(_ApproxStub):
    """Streaming mode via Misra-Gries / Space-Saving (placeholder)."""

    k: int = 16
    _substitute: ClassVar[str] = "Misra-Gries / Space-Saving"


@dataclass(eq=False)
class SpatialStreamingHistogram(_ApproxStub):
    """Streaming binned histogram (placeholder)."""

    bins: int = 64
    _substitute: ClassVar[str] = "equi-width or t-digest-backed histogram"


@dataclass(eq=False)
class SpatialReservoir(_ApproxStub):
    """SpatialReservoir sampling of size ``k`` (placeholder)."""

    k: int = 100
    _substitute: ClassVar[str] = "reservoir sampling (Algorithm R / L)"


# ---------------------------------------------------------------------------
# Streaming-safety check (used by SpatialPatcher.merge)
# ---------------------------------------------------------------------------


def _warn_if_unsafe_streaming(aggregation: SpatialAggregation) -> None:
    if not aggregation.streaming_safe:
        warnings.warn(
            f"{type(aggregation).__name__} has streaming_safe = False — "
            "the merge is happening in-RAM. For streaming alternatives see "
            "scaling.md (Median->ApproxQuantile, Mode->HardVote or "
            "ApproxMode, Learned->two-pass).",
            RuntimeWarning,
            stacklevel=2,
        )
