"""Dask helpers for building patch-level task graphs."""

from __future__ import annotations

from typing import Any


def to_delayed(patcher: Any, field: Any, operator: Any | None = None) -> list[Any]:
    """Return one Dask delayed task per spatial patch."""
    try:
        from dask import delayed
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install geopatcher[dask] to use Dask helpers.") from exc

    tasks = [
        delayed(patcher.patch_at)(field, anchor) for anchor in patcher.anchors(field)
    ]
    if operator is None:
        return tasks
    return [delayed(operator)(task) for task in tasks]


def to_dask_bag(patcher: Any, field: Any) -> Any:
    """Return a Dask bag with one element per patch."""
    try:
        import dask.bag as db
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install geopatcher[dask] to use Dask bag helpers.") from exc

    return db.from_delayed(to_delayed(patcher, field))
