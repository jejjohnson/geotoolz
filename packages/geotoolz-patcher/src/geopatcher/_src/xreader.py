# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright 2026 J. Emmanuel Johnson, distributed under the
# MIT license that covers the rest of geotoolz (first made in pipekit-train,
# moved here with `geopatcher.integrations.train`). Modifications:
#   - the source/stencil pytree is constrained to a flat ``dict[str, ...]``,
#     so the upstream ``jax.tree`` machinery is replaced with plain dict
#     comprehensions and the hard ``jax`` dependency is dropped;
#   - ``_XarraySliceSource`` / ``_XarrayBlockSource`` no longer subclass
#     ``grain.RandomAccessDataSource`` so this module imports with only
#     ``xarray`` + ``numpy`` (grain is needed solely for the block reader);
#   - a new ``collate`` helper stacks a loaded dataset's data variables into
#     a single ndarray along a leading channel axis (the conversion the
#     pipekit Equinox adapter needs — it is not xarray-aware).
"""Xarray data reader for windowed-Zarr training — ported from terrax.

Upstream: ``neuralgcm/terrax`` → ``terrax/xreader/iterators.py`` (Apache-2.0,
© Google LLC; original author Stephan Hoyer). The header above records the
provenance and the modifications.

This module provides the lazy, indexable Grain source machinery behind
`geopatcher.integrations.train.XarrayWindowDataset`:

- `_XarraySliceSource` — per-example random-access reader (the simple path).
- `_XarrayBlockSource` + `group_slices` + `_split_block` — coalesced block
  reader (the cloud-scale path, behind `training_iterator`).
- `_PytreeSource` — zips a flat dict of aligned sources into structurally
  matched samples.
- `collate` — xarray.Dataset slice → stacked ndarray (channels axis 0).

The temporal sampling math (`build_sampling_slices`, `valid_origin_points`)
is `geopatcher.temporal.stencils`.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
from collections.abc import Hashable
from typing import Any

import numpy as np

from geopatcher._src._extras import missing_extra


try:
    import xarray
except ImportError as _e:  # pragma: no cover - exercised when [train] is missing
    raise missing_extra("geopatcher's windowed-Zarr reader", "train") from _e


PyTree = Any


# ---------------------------------------------------------------------
# Collation — the pipekit-specific adaptation
# ---------------------------------------------------------------------


def collate(dataset: xarray.Dataset, var_order: list[Hashable]) -> np.ndarray:
    """Stack a loaded sample's data variables into one ndarray.

    The pipekit Equinox adapter (`adapters/equinox.py`) is not xarray-aware:
    Grain ``.batch()`` ``np.stack``s elements, ``_BatchSource.next_batch``
    calls ``jnp.asarray``, and the synthesised loss does
    ``jax.vmap(model)(x)``. All three require a plain ndarray. This stacks the
    data variables along a **leading channel axis** in the fixed ``var_order``
    captured at dataset construction, so channel ``k`` always means the same
    variable across samples (and feeds ``content_hash``).

    Args:
        dataset: a loaded (in-memory) ``xarray.Dataset`` for one window,
            already transposed so ``sample_dim`` leads each variable.
        var_order: data-variable names in channel order.

    Returns:
        ``ndarray`` of shape ``(len(var_order), *window_shape)``.
    """
    return np.stack([np.asarray(dataset[name].values) for name in var_order], axis=0)


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------


def _drop_static_vars(dataset: xarray.Dataset, sample_dim: str) -> xarray.Dataset:
    """Drop fields that are static and do not vary across the sample dimension."""
    vars_to_drop = [k for k, v in dataset.items() if sample_dim not in v.dims]
    return dataset.drop_vars(vars_to_drop)


def _prepare_xarray_source(source: xarray.Dataset, sample_dim: str) -> xarray.Dataset:
    """Prepare a single ``xarray.Dataset`` for reading.

    Drops static variables and transposes ``sample_dim`` to the front so every
    realised window has ``sample_dim`` leading. The resulting ``data_vars``
    order is the channel order used by `collate` and folded into the dataset's
    content hash.
    """
    # TODO(shoyer, upstream): support multiple sample dimensions.
    if sample_dim not in source.dims:
        raise ValueError(
            f"source does not include variables with a {sample_dim!r} "
            f"dimension:\n{source}"
        )
    if sample_dim not in source.xindexes:
        # Origins and windows are computed from the sample coordinate when the
        # dataset is built; an index keeps it in memory, so building one never
        # reads the store. Without it (`create_default_indexes=False`) that
        # read would hit a remote Zarr at construction.
        raise ValueError(
            f"the {sample_dim!r} coordinate must have an in-memory index; "
            f"open the store with default indexes (the xarray default) or "
            f"call `set_xindex({sample_dim!r})` first"
        )
    source = _drop_static_vars(source, sample_dim)
    if not source.data_vars:
        raise ValueError(
            f"source has no data variables along {sample_dim!r} (all of them "
            f"are static), so there is nothing to sample:\n{source}"
        )
    source = source.transpose(sample_dim, ...)
    return source


def _example_nbytes(
    sources: dict[str, xarray.Dataset],
    stencils: dict[str, Any],
    sample_dim: str,
) -> int:
    """Bytes in a single (multi-leaf) sample — sizes the shuffle buffers."""
    total = 0
    for name, source in sources.items():
        elements_per_sample = stencils[name].points.size
        bytes_per_element = source.isel({sample_dim: slice(1)}).nbytes
        total += elements_per_sample * bytes_per_element
    return total


# ---------------------------------------------------------------------
# Thread-pool loader
# ---------------------------------------------------------------------


class _PoolLoader:
    """Dataset loader using a large thread pool for concurrency.

    A separate thread reads each data variable, which suffices for maximum
    concurrency against a Zarr store (the store uses internal per-chunk
    concurrency underneath). The pool is owned by this object; call
    `close()` to release its threads deterministically instead of waiting
    for the garbage collector.
    """

    def __init__(self, max_workers: int = 100) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers)

    def __call__(self, dataset: xarray.Dataset) -> xarray.Dataset:
        arrays = self._executor.map(lambda var: var.values, dataset.values())
        return dataset.copy(data=dict(zip(dataset, arrays, strict=True)))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------
# Random-access sources
# ---------------------------------------------------------------------


@dataclasses.dataclass(repr=False, eq=False)
class _XarraySliceSource:
    """Random-access reader for per-example slices of an ``xarray.Dataset``."""

    source: xarray.Dataset
    slices: list[slice]
    sample_dim: str

    @functools.cached_property
    def loader(self) -> _PoolLoader:
        return _PoolLoader()

    def __getitem__(self, index: int) -> xarray.Dataset:
        selection = self.source.isel({self.sample_dim: self.slices[index]})
        return self.loader(selection)

    def __len__(self) -> int:
        return len(self.slices)

    def close(self) -> None:
        """Shut down the lazily created loader thread pool, if any."""
        loader = self.__dict__.pop("loader", None)
        if loader is not None:
            loader.close()

    def __getstate__(self) -> dict[str, Any]:
        # Cannot pickle the loader because it includes a thread pool; it is a
        # cached_property and rebuilds lazily after unpickling.
        return {
            "source": self.source,
            "slices": self.slices,
            "sample_dim": self.sample_dim,
        }


@dataclasses.dataclass(eq=False)
class _BlockResult:
    block: xarray.Dataset
    sub_slices: list[slice]


@dataclasses.dataclass(repr=False, eq=False)
class _XarrayBlockSource:
    """Random-access reader for coalesced *blocks* of an ``xarray.Dataset``."""

    source: xarray.Dataset
    groups: list[tuple[slice, list[slice]]]
    sample_dim: str

    @functools.cached_property
    def loader(self) -> _PoolLoader:
        return _PoolLoader()

    def __getitem__(self, index: int) -> _BlockResult:
        block_slice, sub_slices = self.groups[index]
        selection = self.source.isel({self.sample_dim: block_slice})
        return _BlockResult(self.loader(selection), sub_slices)

    def __len__(self) -> int:
        return len(self.groups)

    def close(self) -> None:
        """Shut down the lazily created loader thread pool, if any."""
        loader = self.__dict__.pop("loader", None)
        if loader is not None:
            loader.close()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "groups": self.groups,
            "sample_dim": self.sample_dim,
        }


@dataclasses.dataclass(eq=False)
class _PytreeSource:
    """Random-access source zipping a flat dict of aligned sources.

    One index draws a structurally matched ``{name: leaf_i}`` sample. All
    leaves must report the same length.
    """

    sources: dict[str, Any]

    def __len__(self) -> int:
        lengths = {len(source) for source in self.sources.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"all sources must have the same length, but got lengths: {lengths}"
            )
        [length] = lengths
        return length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {name: source[index] for name, source in self.sources.items()}

    def close(self) -> None:
        """Close every leaf source that supports it (releases thread pools)."""
        for source in self.sources.values():
            close = getattr(source, "close", None)
            if close is not None:
                close()


# ---------------------------------------------------------------------
# Block grouping / splitting
# ---------------------------------------------------------------------


def group_starts(leaf_slices: list[list[slice]], group_size: int) -> list[int]:
    """Example index at which each block starts, shared by every leaf.

    A block holds at most ``group_size`` examples, and in **every** leaf its
    span is at most ``group_size`` of that leaf's windows. Dense origins fill
    a block by count; sparse origins (gaps between windows) end it early, so
    a block never reads the timesteps between two far-apart windows. The
    boundaries are decided once across all leaves: grouping each leaf on its
    own would split leaves of different window widths at different examples
    and pair the wrong inputs and targets.
    """
    if not leaf_slices or not leaf_slices[0]:
        return []
    max_spans = [
        group_size * max(s.stop - s.start for s in slices) for slices in leaf_slices
    ]
    starts = [0]
    for i in range(1, len(leaf_slices[0])):
        first = starts[-1]
        if i - first == group_size or any(
            slices[i].stop - slices[first].start > max_span
            for slices, max_span in zip(leaf_slices, max_spans, strict=True)
        ):
            starts.append(i)
    return starts


def group_slices(
    slices: list[slice], group_size: int, starts: list[int] | None = None
) -> list[tuple[slice, list[slice]]]:
    """Partition a list of slices into groups for reading in parallel.

    Each group is ``(block_slice, sub_slices)``: one contiguous super-slice to
    read once, plus the per-example sub-slices (relative to the block start) to
    carve it back up in memory. The block reads every point in its span; the
    sub-slices preserve each window's original ``step`` so a strided stencil
    (e.g. 3-hourly windows on hourly data) yields the same points as the
    per-example ``isel`` path rather than every point in the span.

    ``starts`` are the block boundaries from `group_starts`; pass the ones
    computed across every leaf so all leaves group identically. Without
    them the boundaries come from ``slices`` alone.
    """
    if not slices:
        return []
    if starts is None:
        starts = group_starts([slices], group_size)
    ends = [*starts[1:], len(slices)]
    return [_group(slices[a:b]) for a, b in zip(starts, ends, strict=True)]


def _group(members: list[slice]) -> tuple[slice, list[slice]]:
    block_start = members[0].start
    block = slice(block_start, members[-1].stop)
    sub_slices = [
        slice(s.start - block_start, s.stop - block_start, s.step) for s in members
    ]
    return block, sub_slices


def _split_block(
    sample_dim: str, block_tree: dict[str, _BlockResult]
) -> list[dict[str, xarray.Dataset]]:
    """Apply each leaf's ``sub_slices`` to its block, yielding per-example trees.

    ``block_tree`` is ``{name: _BlockResult}``; all leaves share the same number
    of sub-slices. Returns a list of ``{name: sub_dataset}`` examples.
    """
    keys = list(block_tree)
    n_examples = len(block_tree[keys[0]].sub_slices)
    out: list[dict[str, xarray.Dataset]] = []
    for j in range(n_examples):
        out.append(
            {
                name: block_tree[name].block.isel(
                    {sample_dim: block_tree[name].sub_slices[j]}
                )
                for name in keys
            }
        )
    return out


# ---------------------------------------------------------------------
# Block-reader iterator (cloud-scale path) — lazily needs grain
# ---------------------------------------------------------------------


def _window_shuffle_cls(grain: Any) -> Any:
    """Resolve ``WindowShuffleIterDataset`` across Grain versions.

    Old Grain (0.2.3) didn't expose it as a public API. The fallback reaches
    into a private module via ``import_module`` (a string import) so static
    type checkers don't try to resolve the private path.
    """
    cls = getattr(grain.experimental, "WindowShuffleIterDataset", None)
    if cls is not None:
        return cls
    import importlib

    shuffle = importlib.import_module("grain._src.python.dataset.transformations")
    return shuffle.WindowShuffleIterDataset


def check_shards(shard_index: Any, shard_count: Any) -> tuple[int, int]:
    """Validate and return ``(shard_index, shard_count)`` as plain ints.

    Rejects anything but ``0 <= shard_index < shard_count`` integers.

    A sequence slice ``[shard_index::shard_count]`` silently accepts an
    out-of-range or negative index (it overlaps or reverses other shards),
    so the host-shard meaning has to be checked up front.

    Raises:
        ValueError: Only one of the two is set, either is not an integer,
            ``shard_count < 1`` or ``shard_index`` is outside
            ``[0, shard_count)``.
    """
    if shard_index is None or shard_count is None:
        raise ValueError("must set both or neither of shard_index and shard_count")
    for name, value in (("shard_index", shard_index), ("shard_count", shard_count)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer; got {value!r}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"need 0 <= shard_index < shard_count with shard_count >= 1; got "
            f"shard_index={shard_index}, shard_count={shard_count}"
        )
    return int(shard_index), int(shard_count)


def training_iterator(
    source: dict[str, xarray.Dataset],
    stencils: dict[str, Any],
    sample_origins: np.ndarray,
    *,
    sample_dim: str = "time",
    num_epochs: int | None = 1,
    buffer_size_in_bytes: float = 1e10,
    buffer_diversity: int = 10,
    shard_index: int | None = None,
    shard_count: int | None = None,
    seed: int = 0,
) -> tuple[Any, _PytreeSource]:
    """Block-coalesced, window-shuffled, shardable iterator of sample trees.

    Yields ``{name: xarray.Dataset}`` examples in randomly shuffled order. The
    caller is responsible for collation + batching (e.g. via
    ``XarrayWindowDataset.build_batch_iter``) and for calling
    ``close()`` on the returned block source to release its loader pools.
    An empty ``sample_origins`` gives an empty iterator.

    ``source`` must already be prepared (see ``_prepare_xarray_source``).

    Args:
        source: flat dict of prepared lazy datasets.
        stencils: matching flat dict of stencils.
        sample_origins: 1-D origin coordinates.
        sample_dim: dimension to sample along.
        num_epochs: epochs to read, or ``None`` for indefinite.
        buffer_size_in_bytes: shuffle-buffer byte budget.
        buffer_diversity: blocks represented in the buffer (>= batch size).
        shard_index, shard_count: multi-host sharding; set both or neither.
        seed: RNG seed.

    Returns:
        ``(dataset, block_source)``: the Grain iter dataset, and the
        `_PytreeSource` of block readers it draws from.

    Raises:
        ValueError: The shard arguments are invalid (see `check_shards`).
    """
    import grain

    from geopatcher.temporal.stencils import build_sampling_slices

    if shard_index is None and shard_count is None:
        shard_index, shard_count = 0, 1
    shard_index, shard_count = check_shards(shard_index, shard_count)

    n_examples = len(sample_origins)
    if 0 < n_examples < shard_count:
        raise ValueError(
            f"{n_examples} examples cannot feed {shard_count} shards: some hosts "
            f"would get none and wait forever while the others train"
        )
    bytes_per_example = _example_nbytes(source, stencils, sample_dim)
    shuffle_window = round(buffer_size_in_bytes / bytes_per_example)
    group_size = max(
        round((buffer_size_in_bytes / buffer_diversity) / bytes_per_example), 1
    )
    # Shards take whole blocks, so there must be at least one block per
    # shard: a short split with a large buffer would otherwise leave hosts
    # with an empty (never-yielding) shard.
    group_size = min(group_size, max(n_examples // shard_count, 1))

    # A guard band can empty a split; build_sampling_slices indexes
    # sample_origins[0], so an empty split reads no blocks.
    leaf_slices = {
        name: (
            build_sampling_slices(
                source_points=ds[sample_dim].values,
                sample_origins=sample_origins,
                stencil=stencils[name],
            )
            if len(sample_origins)
            else []
        )
        for name, ds in source.items()
    }
    starts = group_starts(list(leaf_slices.values()), group_size)
    grain_source = _PytreeSource(
        {
            name: _XarrayBlockSource(
                source[name], group_slices(slices, group_size, starts), sample_dim
            )
            for name, slices in leaf_slices.items()
        }
    )

    # The source already uses many threads internally, so Grain's own
    # read concurrency is kept minimal.
    read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=2)

    rng = np.random.default_rng(seed)
    global_seed = int(rng.integers(2**32))
    window_seed = int(rng.integers(2**32))

    # Emitting multiple examples per block needs Grain's lower-level Dataset
    # API. IterDataset (not MapDataset) avoids re-reading the whole block per
    # example.
    dataset = grain.MapDataset.source(grain_source)
    dataset = dataset[shard_index::shard_count]
    dataset = dataset.shuffle(global_seed)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.to_iter_dataset(read_options=read_options)
    dataset = dataset.map(functools.partial(_split_block, sample_dim))

    @dataclasses.dataclass
    class _UnbatchTransform(grain.experimental.FlatMapTransform):
        max_fan_out: int

        def flat_map(self, element: Any) -> Any:
            return element

    dataset = grain.experimental.FlatMapIterDataset(
        dataset, _UnbatchTransform(group_size)
    )
    if shuffle_window:
        dataset = _window_shuffle_cls(grain)(
            dataset, window_size=shuffle_window, seed=window_seed
        )

    return dataset, grain_source
