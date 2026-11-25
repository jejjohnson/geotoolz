"""Fixed-size batches of patches as one `GeoArray` with a ``patch`` dim (ADR-013).

`batch_split` groups a patcher's patches into `PatchBatch` tuples: the
patches as one `GeoArray` (``(patch, *dims)`` data, ``(patch, 6)``
transforms) and a ``(patch,)`` validity mask for the padding of the last
batch. The batch is NumPy; the caller moves it to JAX once, with
``batch.patches.to_jax()``. `unbatch` turns a batch (or a model's output
for it) back into `Patch` objects that ``patcher.merge`` places, from the
transform rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
from geoarray import DIMS, GeoArray, GeoArrayError, isel

from geopatcher._src.carrier import raster_grid
from geopatcher._src.patch import Patch


__all__ = ["PatchBatch", "batch_split", "unbatch"]

PATCH = "patch"
"""The batch dim of a `PatchBatch`: one place on the ground per item."""


class PatchBatch(NamedTuple):
    """A batch of patches: one carrier and a mask of its real entries.

    A `NamedTuple`, so JAX treats it as a pytree: jitting a step on a
    whole batch compiles once per patch shape, whatever the patches'
    positions or the mask.

    Attributes:
        patches: The patches as one `GeoArray` with dims
            ``("patch", *patch_dims)``, a ``(patch, 6)`` transform (one row
            per patch) and the field's CRS, fill, coordinates and attrs.
            NumPy until the caller calls ``patches.to_jax()``.
        valid: ``(patch,)`` bool, ``True`` for real patches. Padding rows
            of a final batch copy its last real patch (data and transform
            row), so they are finite and look like real data.

    Example:
        >>> for patches, valid in batch_split(patcher, field, batch_size=64):
        ...     patches = patches.to_jax()  # (64, 4, 256, 256), once
        ...     scores = geoarray.map_over(patches, model, "patch")
        ...     loss = masked_mean(scores.data, valid)
    """

    patches: GeoArray
    valid: np.ndarray


def batch_split(
    patcher: Any,
    field: Any,
    *,
    batch_size: int,
    pad_last: bool = True,
) -> Iterator[PatchBatch]:
    """Yield the patches of ``patcher.split(field)`` as fixed-size batches.

    Groups consecutive patches into batches of exactly ``batch_size``; only
    the final batch can be short, and ``pad_last`` decides what happens to
    it. The grouping, padding and mask are `pipekit_array.batch`'s (with
    `geoarray.stack_items` stacking the transform rows), the rule every
    padded batch in the stack follows. Every patch must be a `GeoArray`
    (any raster field adapter) or a ``DataArray`` that reads as a grid
    (`GeoArray.from_xarray`), of one shape.

    Args:
        patcher: Any patcher exposing ``split(field) -> Iterator[Patch]``
            (e.g. `SpatialPatcher`).
        field: The field adapter to split; passed through to
            ``patcher.split``.
        batch_size: Number of patches per batch. Must be positive.
        pad_last: If ``True`` (default), a short final batch is padded to
            ``batch_size`` with copies of its last real patch and
            ``valid == False``, so every batch has one shape (no
            recompilation on the last batch). If ``False``, the final batch
            is yielded short, all valid.

    Yields:
        One `PatchBatch` per ``batch_size`` patches, plus a final short
        batch (padded or not, per ``pad_last``) when the patch count is not
        a multiple of ``batch_size``. The batch is NumPy whatever is
        installed.

    Raises:
        ValueError: ``batch_size`` is not positive; the patches of one
            batch differ in shape (``boundary="shrink"`` edge chips) or in
            fill, coordinates or attrs (`geoarray.stack_items`). A float
            patch with no fill takes NaN, so a float field's interior chips
            and the NaN its ``boundary="pad"`` edge chips declare agree.
        geoarray.GridMismatchError: The patches of one batch differ in CRS.
        TypeError: A patch is not a raster carrier (a plain array, a
            ``DataArray`` with no grid): `stack_patches` stacks those.

    Example:
        >>> batches = list(batch_split(patcher, field, batch_size=16))
        >>> batches[0].patches.dims
        ('patch', 'band', 'y', 'x')
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    from pipekit_array import batch

    items = _batch_items(patcher.split(field), batch_size)
    pad = "last" if pad_last else "none"
    for padded in batch(items, size=batch_size, dim=PATCH, pad=pad):
        yield PatchBatch(padded.value, np.asarray(padded.valid, dtype=bool))


def unbatch(
    batch: PatchBatch,
    data: Any | None = None,
    *,
    patcher: Any,
    domain: Any,
) -> list[Patch]:
    """Turn a batch, or a model's output for it, back into `Patch` objects.

    Padding entries (``valid == False``) are dropped, so a padded final
    batch gives fewer patches than its length. Each patch is placed from its
    transform row against ``domain``'s grid: its anchor and indices are what
    ``patcher.geometry`` gives that pixel offset, and its weights are
    ``patcher.window``'s, so
    ``patcher.merge(unbatch(...), domain)`` reassembles the field.

    Args:
        batch: A `PatchBatch` from `batch_split`.
        data: Optional replacement payload, typically the model output for
            the batch: a `GeoArray` with the ``patch`` dim (from
            `geoarray.map_over`) or an array whose rows line up with the
            batch. Padding rows are ignored.
        patcher: The patcher that split the field (its geometry must be
            `spatial.geometry.Rectangular`).
        domain: The field's domain (``field.domain``, or its bare
            `geoarray.Grid`).

    Returns:
        One `Patch` per valid entry, in batch order, with NumPy data. A row
        that keeps the patch's spatial shape comes back as a `GeoArray` on
        the patch's grid (a per-pixel output with a new band count drops
        the stale per-band attrs); any other row stays a bare array.

    Raises:
        ValueError: The patcher's geometry is not
            `spatial.geometry.Rectangular`, the patches were clipped by
            ``boundary="shrink"``, ``domain`` is not a raster domain, or a
            transform row is not on the domain's pixel grid (another pixel
            size or rotation, naming the patch, or an origin between pixels).

    Example:
        >>> place = dict(patcher=patcher, domain=field.domain)
        >>> outs = [
        ...     p
        ...     for b in batch_split(patcher, field, batch_size=16)
        ...     for p in unbatch(b, model(b.patches), **place)
        ... ]
        >>> merged = patcher.merge(outs, field.domain)
    """
    from geopatcher._src.spatial.geometry import Rectangular
    from geopatcher._src.spatial.patcher import _safe_base_weights

    geometry = patcher.geometry
    if not isinstance(geometry, Rectangular):
        raise ValueError(
            "unbatch places a PatchBatch with a Rectangular geometry, got "
            f"{type(geometry).__name__}"
        )
    size = tuple(int(n) for n in geometry.size[-2:])
    if tuple(batch.patches.shape[-2:]) != size:
        # a chip clipped by boundary="shrink": its anchor and window weights
        # cannot be recovered from the transform row alone
        raise ValueError(
            f"unbatch places full {size} patches, got {batch.patches.shape[-2:]} "
            "ones: patches clipped by boundary='shrink' cannot be placed from "
            "their transforms; use boundary='pad', 'reflect' or 'drop'"
        )
    grid = raster_grid(domain)
    if grid is None:
        raise ValueError(
            "unbatch places a PatchBatch on a raster domain (a geoarray.Grid or an "
            f"object with a .grid), got {type(domain).__name__}"
        )
    _check_resolution(batch.patches, grid)
    to_pixel = ~grid.affine
    weights = _safe_base_weights(patcher.window, geometry)
    patches = []
    for i in np.flatnonzero(np.asarray(batch.valid, dtype=bool)):
        item = isel(batch.patches, **{PATCH: int(i)}).to_numpy()
        anchor = _pixel_offset(to_pixel, item)
        if data is None:
            out: Any = item
        elif isinstance(data, GeoArray) and PATCH in data.dims:
            out = isel(data, **{PATCH: int(i)}).to_numpy()
        else:
            out = _rebuild(item, data[i])
        patches.append(
            Patch(
                data=out,
                anchor=anchor,
                indices=geometry.neighborhood(domain, anchor),
                weights=weights,
            )
        )
    return patches


def _batch_items(patches: Iterator[Patch], size: int) -> Iterator[GeoArray]:
    """Each patch as the carrier `pipekit_array.batch` stacks, lazily.

    Patches of one batch (``size`` consecutive ones) must share a shape; the
    error names their anchors, which the stacked carriers no longer know. A
    float patch with no fill takes NaN: geoarray already reads non-finite
    pixels as nodata, and the ``boundary="pad"`` edge chips of such a field
    declare NaN, so one batch keeps one fill.
    """
    first: tuple[Any, tuple[int, ...]] | None = None
    for k, patch in enumerate(patches):
        shape = tuple(np.shape(patch.data))
        if k % size == 0:
            first = (patch.anchor, shape)
        elif first is not None and shape != first[1]:
            raise ValueError(
                "batch_split needs equal-shaped patches to stack, got "
                f"{first[0]!r}: {first[1]} and {patch.anchor!r}: {shape}. Edge "
                "chips shrink under boundary='shrink'; use boundary='pad', "
                "'reflect' or 'drop' for fixed-size chips."
            )
        item = _as_geoarray(patch.data)
        if item.fill_value is None and np.issubdtype(item.dtype, np.floating):
            item = item.with_data(item.data, fill_value=np.nan)
        yield item


def _as_geoarray(data: Any) -> GeoArray:
    """A patch as a raster carrier, or a `TypeError`."""
    if isinstance(data, GeoArray) and not data.batch_shape:
        return data
    if _is_dataarray(data):
        try:
            carrier = GeoArray.from_xarray(data)
        except (ValueError, GeoArrayError) as exc:
            raise _not_a_raster(data) from exc
        if isinstance(carrier, GeoArray):
            return carrier
    raise _not_a_raster(data)


def _not_a_raster(data: Any) -> TypeError:
    return TypeError(
        "batch_split batches raster patches (a GeoArray, or a DataArray on an "
        f"affine grid), got {type(data).__name__}. "
        "geopatcher.run.stack_patches stacks patches of any other kind."
    )


def _check_resolution(patches: GeoArray, grid: Any) -> None:
    """Every transform row has the target grid's pixel size and rotation."""
    rows = np.asarray(patches.transform, dtype=np.float64)[:, [0, 1, 3, 4]]
    target = np.asarray(grid.transform, dtype=np.float64)[[0, 1, 3, 4]]
    scale = float(np.abs(target).max())
    bad = ~np.isclose(rows, target, rtol=1e-9, atol=1e-12 * scale).all(axis=1)
    if bad.any():
        p = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"patch {p} of the batch has pixel size / rotation (a, b, d, e) = "
            f"{tuple(rows[p].tolist())}, but the target grid has "
            f"{tuple(target.tolist())}: patches are placed on the grid they "
            "were split from, never resampled"
        )


def expand_batches(patches: Any, patcher: Any, domain: Any) -> Iterable[Any]:
    """``patches`` with every batch unpacked into `Patch` objects, lazily.

    A batch is a `PatchBatch` or a `GeoArray` whose batch dim is ``patch``
    (every row real); `SpatialPatcher.merge` and its variants accept either
    in place of patches, one or a stream of them, mixed with `Patch`
    objects. A sequence with no batch in it is returned as it is, so the
    merge hooks still see its length.
    """
    if _is_batch(patches):
        patches = [patches]
    elif isinstance(patches, Sequence) and not any(_is_batch(p) for p in patches):
        return patches
    return _expanded(patches, patcher, domain)


def aexpand_batches(
    patches: AsyncIterable[Any] | Iterable[Any], patcher: Any, domain: Any
) -> AsyncIterable[Any] | Iterable[Any]:
    """`expand_batches` for a sync or an async stream (`SpatialPatcher.amerge`)."""
    if not isinstance(patches, AsyncIterable):
        return expand_batches(patches, patcher, domain)
    return _aexpanded(patches, patcher, domain)


def _expanded(patches: Iterable[Any], patcher: Any, domain: Any) -> Iterator[Any]:
    # closing this generator (a merge that stops early) closes the producer
    iterator = iter(patches)
    try:
        for item in iterator:
            if _is_batch(item):
                batch = _as_patch_batch(item)
                yield from unbatch(batch, patcher=patcher, domain=domain)
            else:
                yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


async def _aexpanded(
    patches: AsyncIterable[Any], patcher: Any, domain: Any
) -> AsyncIterator[Any]:
    # amerge closes its input when the aggregation fails: forward that to
    # the producer, so its cleanup runs before amerge returns
    iterator = aiter(patches)
    try:
        async for item in iterator:
            if _is_batch(item):
                batch = _as_patch_batch(item)
                for patch in unbatch(batch, patcher=patcher, domain=domain):
                    yield patch
            else:
                yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def _is_batch(item: Any) -> bool:
    if isinstance(item, PatchBatch):
        return True
    return isinstance(item, GeoArray) and item.batch_dims == (PATCH,)


def _as_patch_batch(item: Any) -> PatchBatch:
    if isinstance(item, PatchBatch):
        return item
    return PatchBatch(item, np.ones(item.batch_shape[0], dtype=bool))


def _pixel_offset(to_pixel: Any, item: GeoArray) -> tuple[int, int]:
    """``item``'s top-left pixel on the domain grid, as ``(row, col)``."""
    t = item.transform
    col, row = to_pixel * (float(t[2]), float(t[5]))
    r, c = round(row), round(col)
    if abs(row - r) > 1e-6 or abs(col - c) > 1e-6:
        raise ValueError(
            f"a patch's origin falls at pixel ({row}, {col}) of the domain, "
            "not on its grid: unbatch needs the domain of the field that was split"
        )
    return int(r), int(c)


def _rebuild(template: GeoArray, row: Any) -> Any:
    """Wrap ``row`` in ``template``'s carrier when the spatial shape matches."""
    values = np.asarray(row)
    if values.shape[-2:] != template.shape[-2:] or not 2 <= values.ndim <= 4:
        return row
    dims = template.dims if values.ndim == template.ndim else DIMS[-values.ndim :]
    return template.with_data(values, dims=dims)


def _is_dataarray(data: Any) -> bool:
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is an optional extra
        return False
    return isinstance(data, xr.DataArray)
