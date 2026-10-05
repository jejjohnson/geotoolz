"""JAX-friendly batched patch splitting utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geopatcher._src.patch import Patch


try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
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
        carriers: Per-patch carrier templates (`GeoTensor` / `DataArray`
            metadata with no pixel data), used by `unbatch` to rebuild the
            original carrier type; ``None`` for plain arrays and padding.

    `BatchedPatch` is registered as a JAX pytree: ``data`` and ``valid``
    are the leaves, the per-patch metadata is static auxiliary data. Tree
    utilities (``jax.tree.map``, ``jax.device_put``) therefore work on a
    whole batch. The metadata compares by identity, so a jitted function
    *taking a whole batch* retraces for every batch — pass ``batch.data``
    to the jitted model instead.
    """

    data: Any
    anchors: list[Any]
    valid: Any
    indices: list[Any]
    weights: list[Any]
    carriers: list[Any] = field(default_factory=list)


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

    Note:
        With JAX's default ``jax_enable_x64=False``, stacking casts 64-bit
        payloads down: ``float64`` chips become ``float32`` and ``int64``
        chips ``int32`` (JAX warns once). Enable x64 (``jax.config.update(
        "jax_enable_x64", True)``) or cast the field first if the precision
        matters. `unbatch` rebuilds carriers from the (downcast) batch data.

    Yields:
        One `BatchedPatch` per group of ``batch_size`` patches (full
        batches are never padded), plus a final short batch — padded or
        ragged per ``pad_last`` — when the patch count is not a multiple
        of ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is not positive, or if the patches of
            one batch differ in shape (``boundary="shrink"`` edge chips,
            for instance) — use ``boundary="pad"`` / ``"reflect"`` /
            ``"drop"`` to get equal-sized chips.
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
        One `Patch` per valid entry, in batch order. A patch that came from
        a `GeoTensor` or `DataArray` chip comes back as the same carrier
        (transform / CRS / nodata / coords restored, values as a numpy
        array) whenever its row keeps the chip's spatial shape — e.g. a
        per-pixel model output with a different band count; otherwise
        the row is returned as a bare array.

    Examples:
        Round-trip a batch through a model and merge::

            outs = [p for b in batches for p in unbatch(b, model(b.data))]
            merged = patcher.merge(outs, field.domain)
    """
    arrays = batch.data if data is None else data
    valid = np.asarray(batch.valid, dtype=bool)
    carriers = batch.carriers or [None] * len(valid)
    patches = []
    for i, is_valid in enumerate(valid):
        if is_valid:
            patches.append(
                Patch(
                    data=_rebuild(carriers[i], arrays[i]),
                    anchor=batch.anchors[i],
                    indices=batch.indices[i],
                    weights=batch.weights[i],
                )
            )
    return patches


def _batch(patches: list[Patch], batch_size: int, *, pad: bool) -> BatchedPatch:
    shapes = [np.shape(p.data) for p in patches]
    if len(set(shapes)) > 1:
        detail = ", ".join(
            f"{p.anchor!r}: {shape}" for p, shape in zip(patches, shapes, strict=True)
        )
        raise ValueError(
            "batch_split needs equal-shaped patches to stack, got "
            f"{detail}. Edge chips shrink under boundary='shrink'; use "
            "boundary='pad', 'reflect' or 'drop' for fixed-size chips."
        )
    arrays = [jnp.asarray(np.asarray(p.data)) for p in patches]
    valid = [True] * len(patches)
    anchors = [p.anchor for p in patches]
    indices = [p.indices for p in patches]
    weights = [p.weights for p in patches]
    carriers = [_template(p.data) for p in patches]
    if pad:
        for _ in range(batch_size - len(patches)):
            arrays.append(jnp.zeros_like(arrays[0]))
            valid.append(False)
            anchors.append(None)
            indices.append(None)
            weights.append(None)
            carriers.append(None)
    return BatchedPatch(
        data=jnp.stack(arrays, axis=0),
        anchors=anchors,
        valid=jnp.asarray(valid, dtype=bool),
        indices=indices,
        weights=weights,
        carriers=carriers,
    )


def _template(data: Any) -> Any:
    """A pixel-free copy of ``data``'s carrier metadata (``None`` for arrays).

    The template holds a zero-stride broadcast view instead of the chip's
    pixels, so a batch does not keep a second copy of every chip alive.
    """
    from georeader.geotensor import GeoTensor

    if isinstance(data, GeoTensor):
        return GeoTensor(
            values=np.broadcast_to(np.zeros((), dtype=data.dtype), data.shape),
            transform=data.transform,
            crs=data.crs,
            fill_value_default=data.fill_value_default,
            attrs=dict(data.attrs or {}),
        )
    if _is_dataarray(data):
        placeholder = np.broadcast_to(np.zeros((), dtype=data.dtype), data.shape)
        return data.copy(deep=False, data=placeholder)
    return None


def _is_dataarray(data: Any) -> bool:
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is an optional extra
        return False
    return isinstance(data, xr.DataArray)


def _rebuild(template: Any, row: Any) -> Any:
    """Wrap ``row`` in ``template``'s carrier when the spatial shape matches."""
    if template is None:
        return row
    values = np.asarray(row)
    if _is_dataarray(template):
        if values.shape != template.shape:
            return row
        return template.copy(deep=False, data=values)
    if values.shape[-2:] != template.shape[-2:]:
        return row
    from georeader.geotensor import GeoTensor

    return GeoTensor(
        values=values,
        transform=template.transform,
        crs=template.crs,
        fill_value_default=template.fill_value_default,
        attrs=dict(template.attrs or {}),
    )


def _flatten(batch: BatchedPatch) -> tuple[tuple[Any, Any], _StaticMeta]:
    meta = _StaticMeta((batch.anchors, batch.indices, batch.weights, batch.carriers))
    return (batch.data, batch.valid), meta


def _unflatten(meta: _StaticMeta, children: tuple[Any, Any]) -> BatchedPatch:
    anchors, indices, weights, carriers = meta.value
    data, valid = children
    return BatchedPatch(
        data=data,
        anchors=anchors,
        valid=valid,
        indices=indices,
        weights=weights,
        carriers=carriers,
    )


class _StaticMeta:
    """Pytree aux data compared by identity.

    The per-patch metadata holds numpy weight arrays, whose ``==`` is
    elementwise — unusable for JAX's treedef comparison. Identity is the
    only safe equality: a structural "always equal" would let a cached jit
    trace hand back a stale batch's anchors.
    """

    __slots__ = ("value",)

    def __init__(self, value: tuple[Any, ...]) -> None:
        self.value = value


if jax is not None:
    jax.tree_util.register_pytree_node(BatchedPatch, _flatten, _unflatten)
