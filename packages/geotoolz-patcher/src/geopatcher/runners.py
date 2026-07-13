"""Reference runners for applying operators over patch streams."""

from __future__ import annotations

import pickle
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import replace
from typing import Any, Literal

from geopatcher._src.patch import Patch
from geopatcher._src.protocols import Field
from geopatcher._src.spatial.patcher import SpatialPatcher


Backend = Literal["thread", "process"]
ErrorPolicy = Literal["raise", "skip"]


def parallel_map(
    patcher: SpatialPatcher,
    field: Field,
    operator: Callable[[Any], Any],
    *,
    n_workers: int = 8,
    backend: Backend = "thread",
    show_progress: bool = False,
    journal: Any | None = None,
    on_error: ErrorPolicy = "raise",
    batch_size: int = 64,
) -> list[Patch]:
    """Apply ``operator`` to each spatial patch with a reference executor.

    Args:
        patcher: Spatial patcher that defines the anchor schedule.
        field: Field to split into patches.
        operator: Callable applied to each patch's ``data``.
        n_workers: Number of worker threads or processes.
        backend: ``"thread"`` for `ThreadPoolExecutor` or ``"process"`` for
            `ProcessPoolExecutor`.
        show_progress: If ``True``, print a lightweight completion counter
            to stderr.
        journal: Reserved for future `PatchJournal` integration.
        on_error: ``"raise"`` to fail fast, or ``"skip"`` to omit failed
            patches from the returned list.
        batch_size: Maximum number of patches whose reads are coalesced
            into one ``field.select_many`` call when the field supports
            it. No effect on fields without ``select_many``. Default
            64 — keeps peak memory bounded (~``batch_size`` patches'
            worth of array data held simultaneously) while still
            capturing most of the connection-reuse win.

    Returns:
        Patches with ``data`` replaced by ``operator(patch.data)``, ordered by
        the patcher's anchor schedule.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if journal is not None:
        raise NotImplementedError("journal integration is reserved for PatchJournal")
    if backend == "process":
        _ensure_picklable_operator(operator)

    # Duck-type for the batched-read fast path. Fields that implement
    # ``select_many`` (e.g. ``ObstoreCogField``) can fetch every patch's
    # data in one coalesced request; we drive the patcher with a stub
    # that defers reads, then call ``select_many`` in chunks of
    # ``batch_size`` before handing the now-populated patches to the
    # executor. Fields without ``select_many`` go through the original
    # path unchanged.
    if hasattr(field, "select_many") and callable(field.select_many):
        patches = _bulk_select_patches(
            patcher=patcher,
            field=field,
            batch_size=batch_size,
            on_error=on_error,
        )
    else:
        patches = list(patcher.split(field))
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    results: list[tuple[int, Patch]] = []
    total = len(patches)
    done = 0

    def _submit_patch(patch: Patch) -> Patch:
        # ``SpatialPatcher.split`` may attach a non-picklable ``_release``
        # backpressure closure; detach it for transport and let the caller
        # invoke it once the worker returns.
        if backend == "process" and getattr(patch, "_release", None) is not None:
            return replace(patch, _release=None)
        return patch

    with executor_cls(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_apply_operator, i, _submit_patch(patch), operator): (
                i,
                patch,
            )
            for i, patch in enumerate(patches)
        }
        for future in as_completed(futures):
            index, original = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                if on_error == "raise":
                    raise
                warnings.warn(
                    f"parallel_map skipped patch {index} after operator error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            finally:
                # Use ``Patch.close`` so the release closure runs exactly once,
                # whether the worker raised or returned successfully.
                close = getattr(original, "close", None)
                if callable(close):
                    close()
            done += 1
            if show_progress:
                print(f"\r{done}/{total}", end="", file=sys.stderr)

    if show_progress:
        print(file=sys.stderr)
    return [patch for _, patch in sorted(results, key=lambda item: item[0])]


def _apply_operator(
    index: int, patch: Patch, operator: Callable[[Any], Any]
) -> tuple[int, Patch]:
    return index, replace(patch, data=operator(patch.data))


def _ensure_picklable_operator(operator: Callable[[Any], Any]) -> None:
    try:
        pickle.dumps(operator)
    except Exception as exc:
        raise TypeError(
            "parallel_map(..., backend='process') requires a picklable operator; "
            "use a top-level function, use backend='thread', or wrap your "
            "operator with a cloudpickle-based runner."
        ) from exc


class _DeferredSelectField:
    """Stub field that pairs the real ``domain`` with a no-op ``select``.

    Driven through ``patcher.split`` so the patcher emits patches with
    real ``indices`` / ``anchor`` / ``weights`` but **without** issuing
    a per-patch I/O call. The real ``select_many`` runs later, in
    ``_bulk_select_patches``, against the full list of indices.
    """

    __slots__ = ("_real",)

    def __init__(self, real: Field) -> None:
        self._real = real

    @property
    def domain(self) -> Any:
        return self._real.domain

    def select(self, _indexer: Any) -> Any:
        # Sentinel — replaced by select_many output in _bulk_select_patches.
        return None

    def with_data(self, array: Any) -> Any:
        return self._real.with_data(array)


def _bulk_select_patches(
    *,
    patcher: SpatialPatcher,
    field: Field,
    batch_size: int,
    on_error: ErrorPolicy = "raise",
) -> list[Patch]:
    """Drive the patcher with a deferred-select stub, then ``select_many``.

    Splits the full patch list into chunks of ``batch_size`` and runs
    one ``field.select_many`` per chunk so peak memory stays bounded
    by the chunk's array volume rather than the full fan-out.

    Patch indices are passed through the same ``_unwrap_for_select``
    helper that the patcher's normal per-patch path uses, so geometries
    like ``SpatialPolygonIntersection`` that wrap their indices in a
    ``_MaskedWindow`` work correctly with batched fields.

    When ``on_error="skip"`` and ``field.select_many`` raises for a
    chunk, we fall back to per-patch ``field.select`` for that chunk so
    the patch-level skip semantics from the non-batched path are
    preserved (one bad tile shouldn't sink an entire chunk of N
    otherwise-good patches). When ``on_error="raise"`` the exception
    propagates as usual.
    """
    from geopatcher._src.spatial.patcher import _unwrap_for_select

    stub = _DeferredSelectField(field)
    patches = list(patcher.split(stub))
    if not patches:
        return patches
    out: list[Patch] = []
    for start in range(0, len(patches), batch_size):
        chunk = patches[start : start + batch_size]
        indices = [_unwrap_for_select(p.indices) for p in chunk]
        try:
            arrays = field.select_many(indices)
        except Exception:
            if on_error != "skip":
                raise
            # Fall back to per-patch reads so individual failures get
            # scoped (and skipped) rather than aborting the chunk.
            for patch, idx in zip(chunk, indices, strict=True):
                try:
                    data = field.select(idx)
                except Exception as exc:
                    warnings.warn(
                        f"parallel_map skipped patch at anchor {patch.anchor!r} "
                        f"after select error: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                out.append(replace(patch, data=data))
            continue
        if len(arrays) != len(chunk):
            raise RuntimeError(
                f"field.select_many returned {len(arrays)} arrays for "
                f"{len(chunk)} indices; expected one array per indexer."
            )
        for patch, data in zip(chunk, arrays, strict=True):
            out.append(replace(patch, data=data))
    return out
