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
import inspect
import operator
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, overload

from geopatcher._src.patch import Patch


@dataclass
class IndexedPatchView(Sequence[Patch]):
    """Integer-indexed view over a patcher's anchors.

    Wraps a ``(patcher, field)`` pair as a ``Sequence[Patch]`` —
    supports ``len(view)``, ``view[i]``, ``for p in view``, and negative
    indexing. ``view[i]`` is bit-identical to the ``i``-th patch
    `patcher.split(field)` yields.

    Cache accesses are guarded by a `threading.Lock`, so a view shared
    across threads (e.g. a torch DataLoader with ``num_workers=0`` plus
    background threads) is safe.

    The view pickles, so it can be shipped to worker *processes* under
    any start method — ``spawn`` (the macOS / Windows default),
    ``forkserver`` (the Linux default from Python 3.14), ``fork``, and
    Grain's multiprocess ``RandomAccessDataSource``. The anchor list and
    the bound `PatchCache` identity travel with it; the lock is
    recreated and the in-memory ``cache=True`` entries are dropped, so
    each worker starts with its own empty cache (entries are never
    shared back to the parent). The patcher and the field must pickle
    too — every built-in `Field` adapter does.

    Args:
        patcher: A patcher exposing ``anchors(field) -> list`` and
            ``patch_at(field, anchor) -> Patch``. `SpatialPatcher` and
            `TemporalPatcher` satisfy this; other patchers can opt in by
            providing the same two methods. A patcher that also has
            ``patch_anchors(field)`` — one key per patch ``split``
            yields, as `TemporalPatcher` provides for multi-scale
            geometries — is indexed by those keys instead.
        field: The `Field` (or, for `TemporalPatcher`, the series) to
            read from.
        cache: If ``True``, cache patches in memory by integer index after
            the first access (mirrors xrpatcher's ``cache=True``). A
            `PatchCache` instead routes reads through the cross-run,
            content-addressed on-disk cache (gh #24); ``preload`` and
            ``cache_size`` do not apply in that mode, and the patcher's
            ``patch_at`` must accept ``cache=`` (`SpatialPatcher`'s does;
            `TemporalPatcher`'s does not).
        preload: If ``True`` and ``cache=True``, eagerly materialise each
            patch's data (via `xarray.DataArray.load` / `dask.compute` /
            numpy passthrough) before caching, so cached entries are
            fully in RAM rather than lazy views. Requires
            ``cache=True``. Mirrors xrpatcher's ``preload=True``.
        cache_size: Optional LRU bound on the number of cached patches.
            ``None`` (default) keeps the cache unbounded, matching
            xrpatcher's behaviour. Requires ``cache=True``.
        patcher_kwargs: Extra keyword arguments forwarded to every
            ``anchors`` / ``patch_anchors`` / ``patch_at`` call — e.g.
            ``{"time_axis": 1, "coord": times}`` for a `TemporalPatcher`.
    """

    patcher: Any
    field: Any
    cache: bool | Any = False
    preload: bool = False
    cache_size: int | None = None
    patcher_kwargs: dict[str, Any] = field(default_factory=dict)
    _anchors: list[Any] = field(default_factory=list, init=False, repr=False)
    _cache: OrderedDict[int, Patch] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _disk_cache: Any = field(default=None, init=False, repr=False, compare=False)
    _field_id: str | None = field(default=None, init=False, repr=False, compare=False)
    _patch_at_takes_field_id: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # A non-bool `cache` is a `PatchCache` (content-addressed, on-disk);
        # `cache=True` is the in-memory index cache. `preload` / `cache_size`
        # only apply to the latter.
        self._disk_cache = None if isinstance(self.cache, bool) else self.cache
        mem_cache = self.cache is True
        for name, given in (
            ("preload=True", self.preload),
            ("cache_size", self.cache_size is not None),
        ):
            if not given or mem_cache:
                continue
            if self._disk_cache is not None:
                raise ValueError(
                    f"{name} applies only to the in-memory cache (cache=True); "
                    "a PatchCache already serves materialised patches from "
                    f"disk. Drop {name}, or pass cache=True instead."
                )
            raise ValueError(f"{name} requires cache=True.")
        if self.cache_size is not None and self.cache_size < 1:
            raise ValueError("cache_size must be >= 1 (or None for unbounded).")
        anchors = getattr(self.patcher, "patch_anchors", None) or getattr(
            self.patcher, "anchors", None
        )
        patch_at = getattr(self.patcher, "patch_at", None)
        if anchors is None or patch_at is None:
            raise TypeError(
                "IndexedPatchView needs a patcher with both `anchors(field)` "
                "and `patch_at(field, anchor)`; got "
                f"{type(self.patcher).__name__}."
            )
        if self._disk_cache is not None and not _accepts_kwarg(patch_at, "cache"):
            raise TypeError(
                "cache=PatchCache(...) needs a patcher whose "
                "`patch_at(field, anchor, *, cache=...)` accepts a cache; "
                f"{type(self.patcher).__name__}.patch_at does not. Use "
                "cache=True for the in-memory index cache instead."
            )
        self._anchors = list(anchors(self.field, **self.patcher_kwargs))
        self._patch_at_takes_field_id = _accepts_kwarg(patch_at, "field_id")

    def __getstate__(self) -> dict[str, Any]:
        # `threading.Lock` doesn't pickle, and in-memory cache entries are
        # process-local by design: ship the anchors, the bound PatchCache
        # identity and the configuration, not the cached patches.
        state = self.__dict__.copy()
        del state["_cache_lock"]
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._anchors)

    @overload
    def __getitem__(self, idx: int) -> Patch: ...

    @overload
    def __getitem__(self, idx: slice) -> list[Patch]: ...

    def __getitem__(self, idx: int | slice) -> Patch | list[Patch]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self._anchors)))]
        try:
            # `operator.index`, not `int`: ``view[1.9]`` must not silently
            # read patch 1. numpy integers still work.
            i = operator.index(idx)
        except TypeError:
            raise TypeError(
                "IndexedPatchView indices must be integers or slices, not "
                f"{type(idx).__name__}."
            ) from None
        if i < 0:
            i += len(self._anchors)
        if i < 0 or i >= len(self._anchors):
            raise IndexError(
                f"IndexedPatchView index {idx} out of range [0, {len(self._anchors)})"
            )
        kwargs = self.patcher_kwargs
        if self._disk_cache is not None:
            # Only reached when `patch_at` accepts ``cache=`` (checked at
            # construction). A patcher written against the original
            # protocol, ``patch_at(field, anchor, cache=...)``, does not
            # also get ``field_id=``.
            kwargs = {**kwargs, "cache": self._disk_cache}
            if self._patch_at_takes_field_id:
                kwargs["field_id"] = self._disk_field_id()
            return self.patcher.patch_at(self.field, self._anchors[i], **kwargs)
        if self.cache:
            with self._cache_lock:
                cached = self._cache.get(i)
                if cached is not None:
                    self._cache.move_to_end(i)
                    return cached
        patch = self.patcher.patch_at(self.field, self._anchors[i], **kwargs)
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

    def _disk_field_id(self) -> str:
        """The field's `PatchCache` identity, resolved once per view.

        Like the anchor list, it is bound when first needed: deriving it
        per item would stat files — or, for `ObstoreCogField`, send a
        ``HEAD`` — on every ``view[i]``. A source changed after that is
        not noticed by this view; build a new one to pick it up.
        """
        with self._cache_lock:
            if self._field_id is None:
                self._field_id = self._disk_cache.field_id_for(self.field)
            return self._field_id

    @property
    def anchors(self) -> list[Any]:
        """The materialised anchor list this view dispatches into.

        Same sequence ``patcher.anchors(field)`` (or
        ``patcher.patch_anchors(field)`` when the patcher has it) returned
        at construction time; exposed so callers can correlate ``view[i]`` with the
        underlying anchor without going through `patch_at`.
        """
        return list(self._anchors)

    def clear_cache(self) -> None:
        """Drop any cached patches; subsequent reads go back through `patch_at`."""
        with self._cache_lock:
            self._cache.clear()


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """``True`` when ``fn`` takes keyword ``name`` (or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (p.name == name and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params
    )


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
