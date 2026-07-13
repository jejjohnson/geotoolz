"""JAX-friendly batched patch splitting utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.patch import Patch


try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = np


@dataclass(eq=False)
class BatchedPatch:
    """A leading-axis batch of spatial patches.

    Produced by `batch_split`, which stacks the ``data`` payload of up to
    ``batch_size`` patches along a new leading axis so the whole batch can
    be fed to a jitted / vmapped model in one call. Positions ``i`` of the
    per-patch lists line up with ``data[i]`` and ``valid[i]``; padding
    entries (present only in a final padded batch) carry ``None`` metadata
    and all-zeros data.

    Attributes:
        data: Array of shape ``(batch, *patch_shape)`` — the patches'
            payloads stacked along a new leading axis (`jax.numpy` array
            when JAX is installed, `numpy` otherwise).
        anchors: Per-patch anchors, length ``batch``; ``None`` for
            padding entries.
        valid: Boolean array of shape ``(batch,)``; ``True`` where the
            entry is a real patch, ``False`` where it is zero-padding.
        indices: Per-patch index payloads (as on `Patch.indices`), length
            ``batch``; ``None`` for padding entries.
        weights: Per-patch window weights (as on `Patch.weights`), length
            ``batch``; ``None`` for padding entries.
    """

    data: Any
    anchors: list[Any]
    valid: Any
    indices: list[Any]
    weights: list[Any]


def batch_split(
    patcher: Any,
    field: Any,
    *,
    batch_size: int,
    pad_last: bool = True,
) -> Iterator[BatchedPatch]:
    """Yield `BatchedPatch` objects with data stacked on a leading axis.

    Consumes ``patcher.split(field)`` and groups consecutive patches into
    batches of exactly ``batch_size``. Only the final batch can be short;
    ``pad_last`` controls what happens to it.

    Args:
        patcher: Any patcher exposing ``split(field) -> Iterator[Patch]``
            (e.g. `SpatialPatcher`).
        field: The field to split; passed through to ``patcher.split``.
        batch_size: Number of patches per batch. Must be positive.
        pad_last: If ``True`` (default), a short final batch is zero-padded
            up to ``batch_size`` — padding entries have all-zeros ``data``,
            ``valid == False``, and ``None`` anchors/indices/weights — so
            every yielded batch has the same leading-axis length (no JIT
            recompilation on the last batch). If ``False``, the final batch
            is yielded ragged, with a leading axis equal to the number of
            remaining patches and all entries valid.

    Yields:
        One `BatchedPatch` per group of ``batch_size`` patches (full
        batches are never padded), plus a final short batch — padded or
        ragged per ``pad_last`` — when the patch count is not a multiple
        of ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch = []
    for patch in patcher.split(field):
        batch.append(patch)
        if len(batch) == batch_size:
            yield _batch(batch, batch_size, pad=False)
            batch = []
    if batch:
        yield _batch(batch, batch_size, pad=pad_last)


def unbatch(batch: BatchedPatch, data: Any | None = None) -> list[Patch]:
    """Convert a `BatchedPatch` back to ordinary `Patch` objects.

    Entries whose ``batch.valid`` flag is ``False`` (zero-padding added by
    ``batch_split(..., pad_last=True)``) are dropped, so the result holds
    only real patches — typically fewer than the batch's leading-axis
    length for a padded final batch.

    Args:
        batch: The batch to unpack. Its ``anchors`` / ``indices`` /
            ``weights`` are carried onto the reconstructed patches.
        data: Optional replacement payload with the same leading-axis
            layout as ``batch.data`` — typically the model output for the
            batch. When given, patch ``i`` receives ``data[i]`` instead of
            ``batch.data[i]``; the padding rows of ``data`` are ignored.

    Returns:
        One `Patch` per valid entry, in batch order.
    """
    arrays = batch.data if data is None else data
    valid = np.asarray(batch.valid, dtype=bool)
    patches = []
    for i, is_valid in enumerate(valid):
        if is_valid:
            patches.append(
                Patch(
                    data=arrays[i],
                    anchor=batch.anchors[i],
                    indices=batch.indices[i],
                    weights=batch.weights[i],
                )
            )
    return patches


def _batch(patches: list[Patch], batch_size: int, *, pad: bool) -> BatchedPatch:
    arrays = [jnp.asarray(p.data) for p in patches]
    valid = [True] * len(patches)
    anchors = [p.anchor for p in patches]
    indices = [p.indices for p in patches]
    weights = [p.weights for p in patches]
    if pad:
        for _ in range(batch_size - len(patches)):
            arrays.append(jnp.zeros_like(arrays[0]))
            valid.append(False)
            anchors.append(None)
            indices.append(None)
            weights.append(None)
    return BatchedPatch(
        data=jnp.stack(arrays, axis=0),
        anchors=anchors,
        valid=jnp.asarray(valid, dtype=bool),
        indices=indices,
        weights=weights,
    )
