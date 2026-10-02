"""Dask helpers for building patch-level task graphs.

Both helpers build **plain per-patch read tasks** around
`SpatialPatcher.patch_at`. The runner-level policies of
`SpatialPatcher.split` — ``on_error`` / retries, hooks, the journal,
``PatchCache`` and prefetch / backpressure — do **not** apply: a failing
patch fails its Dask task, and retries, logging and caching are the
scheduler's (or the caller's) concern.

The field and the patcher enter the graph **once** each, as
``delayed(obj, pure=True, traverse=False)`` handles every task depends on,
instead of being embedded in every task (which would serialise N copies
of the field on a distributed scheduler). ``traverse=False`` keeps Dask
from unpacking a dataclass field such as `DaskField` and computing its
whole backing array in one task — each task still reads only its own
window.
"""

from __future__ import annotations

from typing import Any


def _patch_at(patcher: Any, field: Any, anchor: Any) -> Any:
    return patcher.patch_at(field, anchor)


def _handles(patcher: Any, field: Any) -> tuple[Any, Any, Any]:
    try:
        from dask import delayed
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install geopatcher[dask] to use Dask helpers.") from exc
    return (
        delayed,
        delayed(patcher, pure=True, traverse=False),
        delayed(field, pure=True, traverse=False),
    )


def to_delayed(patcher: Any, field: Any, operator: Any | None = None) -> list[Any]:
    """Return one Dask delayed task per spatial patch.

    Each task computes ``patcher.patch_at(field, anchor)`` (then
    ``operator(patch)`` when given). The field is wrapped in a single
    delayed handle shared by every task. Runner-level policies
    (``on_error``, hooks, journal, cache, prefetch) do not apply — see
    the module docstring.

    Args:
        patcher: A `SpatialPatcher`.
        field: The `Field` to patch.
        operator: Optional callable mapped over each patch.

    Returns:
        ``list[dask.delayed.Delayed]`` in anchor order.
    """
    delayed, patcher_h, field_h = _handles(patcher, field)
    patch_at = delayed(_patch_at, pure=True)
    tasks = [patch_at(patcher_h, field_h, anchor) for anchor in patcher.anchors(field)]
    if operator is None:
        return tasks
    return [delayed(operator)(task) for task in tasks]


def to_dask_bag(patcher: Any, field: Any) -> Any:
    """Return a Dask bag with one element (and one partition) per patch.

    Built as ``db.from_sequence(anchors, partition_size=1).map(...)`` with
    the field and patcher passed as shared delayed handles, so — as in
    `to_delayed` — the field enters the graph once and runner-level
    policies do not apply.
    """
    try:
        import dask.bag as db
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install geopatcher[dask] to use Dask bag helpers.") from exc

    _, patcher_h, field_h = _handles(patcher, field)
    anchors = list(patcher.anchors(field))
    bag = db.from_sequence(anchors, partition_size=1)
    return bag.map(_swap_patch_at, patcher_h, field_h)


def _swap_patch_at(anchor: Any, patcher: Any, field: Any) -> Any:
    # `Bag.map` passes the bag element first.
    return patcher.patch_at(field, anchor)
