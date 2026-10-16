"""Lazy, indexable windowed-Zarr training dataset for pipekit-train.

`XarrayWindowDataset` samples ``(input, target)`` windows along a dimension
(default ``"time"``) from one or more **lazy** ``xarray.Dataset``/Zarr stores,
never materialising the whole store. It is a pipekit-train `TrainingDataset`
for ERA5-scale data: a window is ``source.isel({dim: slice})`` — random access
— so the dataset is indexable and rides the Equinox adapter's Grain path,
inheriting shuffle + checkpoint/resume for free.

Two read paths share one class:

- **Per-example** (default): ``__getitem__`` reads + collates one window. The
  adapter's ``MapDataset → shuffle → repeat → batch`` chain drives it.
- **Block reader** (``block_reader=True``): coalesced reads + window shuffle +
  multi-host sharding, reached through the adapter's ``build_batch_iter`` hook
  so it works through the same ``TrainingLoop``.

The temporal sampling math is `geopatcher.temporal.stencils`; the read
machinery (ported from terrax) is `geopatcher._src.xreader`. It moved here
from ``pipekit_train.xarray_window``, so pipekit does not import geopatcher.

Install the ``[train]`` extra (pipekit-train with its Equinox adapter,
xarray and zarr)::

    from geopatcher.integrations.train import XarrayWindowDataset
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from geopatcher._src._extras import missing_extra


try:
    import xarray
    from pipekit_train.dataset import Split, TrainingDataset, hash_config
except ImportError as _e:  # pragma: no cover - exercised when [train] is missing
    raise missing_extra("geopatcher.integrations.train", "train") from _e

from geopatcher._src import xreader as _xreader
from geopatcher._src._digest import array_digest
from geopatcher.temporal.stencils import (
    build_sampling_slices,
    coord_step,
    stencil_offsets,
    valid_origin_points,
)


__all__ = ["XarrayWindowDataset"]


# Split → (start_tenth, stop_tenth) over a tenths denominator, mirroring
# SimulationDataset so the integer split sizes are exact (80 / 10 / 10).
#: The two source leaves every dataset pairs up, as ``(input, target)``.
_LEAVES = frozenset({"input", "target"})

_SPLIT_TENTHS: dict[Split, tuple[int, int]] = {
    "train": (0, 8),
    "val": (8, 9),
    "test": (9, 10),
}


def _intersect_valid_origins(
    source: dict[str, xarray.Dataset],
    stencils: dict[str, Any],
    sample_dim: str,
) -> np.ndarray:
    """Origins for which every leaf's stencil fits — intersection across leaves."""
    per_leaf = [
        valid_origin_points(ds[sample_dim].values, stencils[name])
        for name, ds in source.items()
    ]
    origins = per_leaf[0]
    for nxt in per_leaf[1:]:
        origins = np.intersect1d(origins, nxt)
    return origins


def _default_guard(
    source: dict[str, xarray.Dataset],
    stencils: dict[str, Any],
    sample_dim: str,
) -> int:
    """Widest stencil *reach* from the origin, expressed in source-steps.

    Dropping this many origins on the interior side of each split boundary
    guarantees no window straddles a train/val/test boundary, since origins sit
    at source resolution. The reach is the furthest a window extends from its
    origin in *either* direction, not its span: a single-point offset stencil
    (e.g. a target at ``+24h``) has zero span but reaches 24 steps across the
    boundary.

    The step and the window offsets come from `coord_step` and
    `stencil_offsets`, the same conversions `build_sampling_slices` uses, so
    the guard and the sampled windows cannot disagree.
    """
    guard = 0
    for name, ds in source.items():
        points = ds[sample_dim].values
        if len(points) < 2:  # no step to measure; no window can straddle
            continue
        lo, hi, _ = stencil_offsets(stencils[name], coord_step(points))
        # A window at origin index i covers indices [i + lo, i + hi).
        guard = max(guard, abs(lo), abs(hi - 1))
    return guard


def _source_view(ds: xarray.Dataset, sample_dim: str) -> dict[str, Any]:
    """What a leaf's examples look like beyond the store token and the origins.

    Two views of one store version (``ds.isel(x=slice(...))``, a spatial
    transpose) share variable names, stencils and origins but return
    different values or shapes. Dimension order, sizes and the index
    coordinates (in memory, so no store read) tell them apart.
    """
    return {
        "dims": {str(name): list(map(str, var.dims)) for name, var in ds.items()},
        "sizes": {str(dim): int(size) for dim, size in ds.sizes.items()},
        "coords": {
            str(name): array_digest(ds[name].values)
            for name in ds.xindexes
            if name != sample_dim
        },
    }


def _split_origins(origins: np.ndarray, split: Split, guard: int) -> np.ndarray:
    """Contiguous tenths split with a guard band on interior boundaries.

    Contiguous (not random) so splits don't temporally interleave; the guard
    band drops ``guard`` origins on each interior side so windows near a
    boundary don't leak across splits.
    """
    origins = np.asarray(origins)
    n = len(origins)
    start_t, stop_t = _SPLIT_TENTHS[split]
    start = n * start_t // 10
    end = n * stop_t // 10
    if start_t != 0:  # a split exists below → trim the low side
        start += guard
    if stop_t != 10:  # a split exists above → trim the high side
        end -= guard
    if start >= end:
        return origins[:0]
    return origins[start:end]


class XarrayWindowDataset(TrainingDataset):
    """Lazy, indexable ``(input, target)`` windows sampled from a Zarr store.

    Args:
        source: dict of lazy ``xarray.Dataset``s, e.g.
            ``{"input": ds, "target": ds}``, opened with ``chunks=None``.
        stencils: matching dict of stencils (one per source leaf).
        sample_origins: coordinate values to anchor windows on. ``None`` →
            derived via ``valid_origin_points`` (intersection across leaves).
        sample_dim: dimension to sample along (default ``"time"``).
        store_version: opaque token identifying the store contents for hashing
            (Zarr ``.zmetadata`` token or user-supplied). Required for a stable
            content-recipe hash.
        block_reader: when ``True``, the dataset advertises ``build_batch_iter``
            so the adapter drives the cloud-scale block reader (coalesced reads
            + window shuffle + sharding). Default ``False`` (per-example path).
        buffer_size_in_bytes: shuffle-buffer byte budget for the block reader
            (ignored unless ``block_reader=True``).
        buffer_diversity: blocks represented in the shuffle buffer (ignored
            unless ``block_reader=True``).
        shard_index: this host's shard for the block reader. When it and
            ``shard_count`` are both ``None`` and ``block_reader=True``, both
            are auto-detected from JAX (``process_index`` / ``process_count``)
            at build time.
        shard_count: number of shards for the block reader (see
            ``shard_index``).
        split_guard: origins to drop on the interior side of each split
            boundary. ``None`` → auto-derive from the widest stencil span.
        seed: RNG seed, the standard `TrainingDataset` knob.
        split: ``"train"``, ``"val"`` or ``"test"``, the standard
            `TrainingDataset` knob.
    """

    forbid_in_yaml: ClassVar[bool] = True  # source carries an open store
    __config_mixin_auto__ = False

    def __init__(
        self,
        source: dict[str, xarray.Dataset],
        stencils: dict[str, Any],
        sample_origins: np.typing.ArrayLike | None = None,
        *,
        sample_dim: str = "time",
        store_version: str,
        block_reader: bool = False,
        buffer_size_in_bytes: float = 1e10,
        buffer_diversity: int = 10,
        shard_index: int | None = None,
        shard_count: int | None = None,
        split_guard: int | None = None,
        seed: int = 0,
        split: Split = "train",
    ) -> None:
        super().__init__(seed=seed, split=split)
        if set(source) != set(stencils):
            raise ValueError(
                "source and stencils must have matching keys; got "
                f"{sorted(source)} vs {sorted(stencils)}."
            )
        if set(source) != _LEAVES:
            raise ValueError(
                "source and stencils must have exactly the keys "
                f"{sorted(_LEAVES)}; got {sorted(source)}."
            )
        if split_guard is not None and (
            isinstance(split_guard, bool)
            or not isinstance(split_guard, (int, np.integer))
            or split_guard < 0
        ):
            raise ValueError(
                f"split_guard must be a non-negative integer or None; got "
                f"{split_guard!r}."
            )
        if shard_index is not None or shard_count is not None:
            _xreader.check_shards(shard_index, shard_count)
        # Prepare each leaf (drop static vars + lead with sample_dim). The
        # resulting data_vars order is the channel order we collate + hash.
        self.source = {
            name: _xreader._prepare_xarray_source(ds, sample_dim)
            for name, ds in source.items()
        }
        self.stencils = stencils
        self.sample_dim = sample_dim
        self.store_version = store_version
        self.var_order = {name: list(ds.data_vars) for name, ds in self.source.items()}
        self.block_reader = block_reader
        self.buffer_size_in_bytes = buffer_size_in_bytes
        self.buffer_diversity = buffer_diversity
        self.shard_index = shard_index
        self.shard_count = shard_count
        # Block-reader sources handed out by build_batch_iter; close() shuts
        # their loader pools down too.
        self._block_sources: list[_xreader._PytreeSource] = []

        if sample_origins is not None:
            self._all_origins = np.asarray(sample_origins)
        else:
            self._all_origins = _intersect_valid_origins(
                self.source, stencils, sample_dim
            )
        self._split_guard = (
            split_guard
            if split_guard is not None
            else _default_guard(self.source, stencils, sample_dim)
        )
        self.sample_origins = _split_origins(
            self._all_origins, split, self._split_guard
        )
        self._grain_source = self._build_grain_source()

    # --- construction helpers ----------------------------------------------

    def _build_grain_source(self) -> _xreader._PytreeSource:
        def _slices(ds: xarray.Dataset, stencil: Any) -> list[slice]:
            # A degenerate split (e.g. a guard band wider than a narrow band)
            # yields no origins; build_sampling_slices indexes origins[0], so
            # short-circuit to a length-0 source instead of crashing.
            if len(self.sample_origins) == 0:
                return []
            return build_sampling_slices(
                ds[self.sample_dim].values, self.sample_origins, stencil
            )

        return _xreader._PytreeSource(
            {
                name: _xreader._XarraySliceSource(
                    ds, _slices(ds, self.stencils[name]), self.sample_dim
                )
                for name, ds in self.source.items()
            }
        )

    def _collate_pair(self, pair: dict[str, Any]) -> tuple[Any, Any]:
        return (
            _xreader.collate(pair["input"], self.var_order["input"]),
            _xreader.collate(pair["target"], self.var_order["target"]),
        )

    def _resolve_shards(self) -> tuple[int, int]:
        if self.shard_index is not None and self.shard_count is not None:
            return self.shard_index, self.shard_count
        if self.shard_index is not None or self.shard_count is not None:
            raise ValueError("set both or neither of shard_index and shard_count")
        try:  # auto-detect a multi-host JAX setup
            import jax
        except ImportError:  # no JAX: a single host reads everything
            return 0, 1
        # A JAX that is installed but fails to report its processes is a
        # broken distributed setup; let it raise rather than have every host
        # read the full dataset.
        return jax.process_index(), jax.process_count()

    # --- indexable contract → rides the adapter's Grain path ---------------

    def __len__(self) -> int:
        return len(self._grain_source)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self._collate_pair(self._grain_source[index])

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    # --- cloud-scale path, reachable via the adapter hook ------------------

    def build_batch_iter(self, batch_size: int, seed: int) -> Any:
        """Adapter hook (see ``adapters/equinox.py`` ``_BatchSource``).

        Returns a checkpointable Grain *iterator* yielding collated, batched
        ``(X, Y)`` pairs via the block reader, or ``None`` to decline so the
        adapter falls back to the per-example indexable path. Declines unless
        ``block_reader=True``.
        """
        if not self.block_reader:
            return None
        shard_index, shard_count = self._resolve_shards()
        iter_ds, block_source = _xreader.training_iterator(
            self.source,
            self.stencils,
            self.sample_origins,
            sample_dim=self.sample_dim,
            num_epochs=None,
            buffer_size_in_bytes=self.buffer_size_in_bytes,
            buffer_diversity=self.buffer_diversity,
            shard_index=shard_index,
            shard_count=shard_count,
            seed=seed,
        )
        self._block_sources.append(block_source)
        iter_ds = iter_ds.map(self._collate_pair).batch(batch_size, drop_remainder=True)
        return iter(iter_ds)

    # --- split handling -----------------------------------------------------

    def with_split(self, split: Split) -> XarrayWindowDataset:
        """Return a clone of this dataset restricted to ``split``.

        The clone builds its own reader sources (and, lazily, its own
        loader thread pools); the parent keeps its own. Call `close`
        on datasets you no longer iterate to release their pools.
        """
        clone = XarrayWindowDataset.__new__(XarrayWindowDataset)
        clone.__dict__.update(self.__dict__)
        clone.split = split
        clone.sample_origins = _split_origins(
            self._all_origins, split, self._split_guard
        )
        clone._grain_source = clone._build_grain_source()
        clone._block_sources = []
        return clone

    def close(self) -> None:
        """Release the loader thread pools held by this dataset's sources.

        Each reader source lazily creates a large `ThreadPoolExecutor` on
        first read; without an explicit close those threads are only
        reclaimed by the garbage collector. This covers the per-example
        source and every block-reader source `build_batch_iter` handed
        out. Safe to call repeatedly; the pools are rebuilt on the next
        read if the dataset is used again.
        """
        self._grain_source.close()
        for source in self._block_sources:
            source.close()

    # --- reproducibility ----------------------------------------------------

    def content_hash(self) -> str:
        # NB: block_reader / buffer / shard params are deliberately NOT hashed —
        # they change iteration order / sharding, not the logical (x, y) set.
        return hash_config(
            "XarrayWindowDataset",
            {
                "store_version": self.store_version,
                "stencils": {k: s.get_config() for k, s in self.stencils.items()},
                "var_order": self.var_order,
                "views": {
                    name: _source_view(ds, self.sample_dim)
                    for name, ds in self.source.items()
                },
                "origins_sha": array_digest(self.sample_origins),
                "sample_dim": self.sample_dim,
                "seed": self.seed,
                "split": self.split,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "sample_dim": self.sample_dim,
            "store_version": self.store_version,
            "stencils": {k: s.get_config() for k, s in self.stencils.items()},
            "var_order": self.var_order,
            "n_origins": len(self.sample_origins),
            "block_reader": self.block_reader,
            "seed": self.seed,
            "split": self.split,
        }
