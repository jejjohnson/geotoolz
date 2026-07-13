"""`IndexedPatchView` — random-access wrapper over a patcher + field pair.

The patcher's canonical surface is `patcher.split(field) → Iterator[Patch]`
(ADR-001). For ML loaders that need integer-indexed random access
(torch `Dataset.__getitem__`, Grain `RandomAccessDataSource.__getitem__`,
xrpatcher's `patcher[i]`) this wrapper exposes `Sequence[Patch]` over a
materialised list of anchors and dispatches to `patcher.patch_at`.

Optional in-memory cache (`cache=True`) mirrors `xrpatcher.XRDAPatcher`'s
``cache`` / ``preload`` flags one-for-one — the integer index is the
cache key, no content hashing. The deeper content-addressed cache is
tracked separately as gh #24.

See ADR-005 for the design choices (Sequence-not-torch-Dataset,
cache-on-view-not-on-patcher).
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, overload

from geopatcher._src.patch import Patch


@dataclass
class IndexedPatchView(Sequence[Patch]):
    """Integer-indexed view over a `SpatialPatcher`'s anchors.

    Wraps a ``(patcher, field)`` pair as a ``Sequence[Patch]`` —
    supports ``len(view)``, ``view[i]``, ``for p in view``, and negative
    indexing. The patches are bit-identical to those `patcher.split(field)`
    yields for the same anchor.

    Cache accesses are guarded by a `threading.Lock`, so a view shared
    across threads (e.g. a torch DataLoader with ``num_workers=0`` plus
    background threads) is safe. Note that a torch DataLoader with
    ``num_workers > 0`` forks worker *processes*: each worker gets its
    own process-local copy of this view, and therefore its own cache —
    entries are not shared back to the parent.

    Args:
        patcher: A patcher exposing ``anchors(field) -> list`` and
            ``patch_at(field, anchor) -> Patch``. `SpatialPatcher`
            satisfies this; other patchers can opt in by providing the
            same two methods.
        field: The `Field` to read from.
        cache: If ``True``, cache patches in memory by integer index after
            the first access (mirrors xrpatcher's ``cache=True``). A
            `PatchCache` instead routes reads through the cross-run,
            content-addressed on-disk cache (gh #24); ``preload`` and
            ``cache_size`` do not apply in that mode.
        preload: If ``True`` and ``cache=True``, eagerly materialise each
            patch's data (via `xarray.DataArray.load` / `dask.compute` /
            numpy passthrough) before caching, so cached entries are
            fully in RAM rather than lazy views. Requires
            ``cache=True``. Mirrors xrpatcher's ``preload=True``.
        cache_size: Optional LRU bound on the number of cached patches.
            ``None`` (default) keeps the cache unbounded, matching
            xrpatcher's behaviour. Requires ``cache=True``.
    """

    patcher: Any
    field: Any
    cache: bool | Any = False
    preload: bool = False
    cache_size: int | None = None
    _anchors: list[Any] = field(default_factory=list, init=False, repr=False)
    _cache: OrderedDict[int, Patch] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _disk_cache: Any = field(default=None, init=False, repr=False, compare=False)
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # A non-bool `cache` is a `PatchCache` (content-addressed, on-disk);
        # `cache=True` is the in-memory index cache. `preload` / `cache_size`
        # only apply to the latter.
        self._disk_cache = None if isinstance(self.cache, bool) else self.cache
        mem_cache = self.cache is True
        if self.preload and not mem_cache:
            raise ValueError("preload=True requires cache=True.")
        if self.cache_size is not None:
            if not mem_cache:
                raise ValueError("cache_size requires cache=True.")
            if self.cache_size < 1:
                raise ValueError("cache_size must be >= 1 (or None for unbounded).")
        anchors = getattr(self.patcher, "anchors", None)
        patch_at = getattr(self.patcher, "patch_at", None)
        if anchors is None or patch_at is None:
            raise TypeError(
                "IndexedPatchView needs a patcher with both `anchors(field)` "
                "and `patch_at(field, anchor)`; got "
                f"{type(self.patcher).__name__}."
            )
        self._anchors = list(anchors(self.field))

    def __len__(self) -> int:
        return len(self._anchors)

    @overload
    def __getitem__(self, idx: int) -> Patch: ...

    @overload
    def __getitem__(self, idx: slice) -> list[Patch]: ...

    def __getitem__(self, idx: int | slice) -> Patch | list[Patch]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self._anchors)))]
        i = int(idx)
        if i < 0:
            i += len(self._anchors)
        if i < 0 or i >= len(self._anchors):
            raise IndexError(
                f"IndexedPatchView index {idx} out of range [0, {len(self._anchors)})"
            )
        if self._disk_cache is not None:
            return self.patcher.patch_at(
                self.field, self._anchors[i], cache=self._disk_cache
            )
        if self.cache:
            with self._cache_lock:
                cached = self._cache.get(i)
                if cached is not None:
                    self._cache.move_to_end(i)
                    return cached
        patch = self.patcher.patch_at(self.field, self._anchors[i])
        if self.cache:
            if self.preload:
                patch = _materialise(patch)
            with self._cache_lock:
                # Two threads may race to build the same index; keep the
                # first stored entry so repeated reads return one object.
                existing = self._cache.get(i)
                if existing is not None:
                    self._cache.move_to_end(i)
                    return existing
                self._cache[i] = patch
                if self.cache_size is not None:
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return patch

    @property
    def anchors(self) -> list[Any]:
        """The materialised anchor list this view dispatches into.

        Same sequence ``patcher.anchors(field)`` returned at construction
        time; exposed so callers can correlate ``view[i]`` with the
        underlying anchor without going through `patch_at`.
        """
        return list(self._anchors)

    def clear_cache(self) -> None:
        """Drop any cached patches; subsequent reads go back through `patch_at`."""
        with self._cache_lock:
            self._cache.clear()


def _materialise(patch: Patch) -> Patch:
    """Best-effort eager-load on lazy patch data.

    Tries `xarray.DataArray.load`, then `dask.array.compute`, then leaves
    the patch untouched (numpy arrays are already in RAM). Returns a new
    Patch with the loaded data; never mutates the input.
    """
    data = patch.data
    for attr in ("load", "compute"):
        loader = getattr(data, attr, None)
        if loader is not None and callable(loader):
            return dataclasses.replace(patch, data=loader())
    return patch


__all__ = ["IndexedPatchView"]
